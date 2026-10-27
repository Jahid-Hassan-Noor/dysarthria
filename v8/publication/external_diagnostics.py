"""Frozen-model UASpeech content/channel and objective background diagnostics.

External labels are used only for roster validation and saved evaluation metrics.
No optimizer, calibration, threshold selection, or arm/channel selection is used.
New WavLM representations live exclusively in the publication cache. Encoding is
resumable per recording, with fail-closed source/configuration/content signatures.
"""
from __future__ import annotations

import gc
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import (REFERENCE_NAME, canonical_hash, external_features, frozen_model,
                     publish_result, read_json, sha256, stage_status, validate_features,
                     write_csv, write_json)

CHANNELS = tuple(f'M{i}' for i in range(2, 9))
COMMON_CODE = re.compile(r'(?:C\d+|CW\d+|D\d+|L[A-Z])\Z')
MLF_NAME = re.compile(r'(?P<speaker>C?[FM]\d+)_(?P<block>B[123])_(?P<word>[A-Z]+\d*|L[A-Z])_(?P<channel>M[2-8])\.lab\Z')
SILENCE_PROTOCOL = dict(version='v8_publication_background_edges_v1', frame_seconds=.02,
                        relative_peak_db=-35., minimum_seconds=.1, maximum_seconds=1.,
                        eligible_clips_per_speaker=50, normalization='peak_normalize_selected_real_segment',
                        selection='ascending_SHA256_relative_path_first_50_eligible',
                        region='longest_contiguous_low_energy_leading_or_trailing_edge',
                        trim=False, gate=False, labels_used_for_selection=False)


def validate_roster(ctx, meta):
    """Diagnosis comes from the official physical-speaker roster, never word IDs."""
    expected = {s: label for label, ids in ctx.pipeline.UASPEECH_ROSTER.items() for s in ids}
    for speaker, part in meta.groupby('physical_speaker_id'):
        if speaker not in expected or set(part.label.astype(int)) != {expected[speaker]}:
            raise RuntimeError(f'Official UASpeech roster/label mismatch: {speaker}')


def read_word_transcripts(root):
    """Parse documented per-speaker word MLFs and reject contradictory mappings."""
    mapping, files = {}, []
    for release in ['UASpeech_normalized_C', 'UASpeech_normalized_FM']:
        directory = Path(root) / release / 'mlf'
        for path in sorted(directory.glob('*/*_word.mlf')):
            files.append({'path': str(path), 'sha256': sha256(path)})
            current, words = None, []
            for line in path.read_text(errors='strict').splitlines():
                line = line.strip()
                if not line or line == '#!MLF!#':
                    continue
                if line.startswith('"') and line.endswith('"'):
                    if current is not None:
                        raise RuntimeError(f'Unterminated MLF entry in {path}')
                    match = MLF_NAME.fullmatch(Path(line.strip('"')).name)
                    current = tuple(match.group(k) for k in ['speaker', 'block', 'word', 'channel']) if match else None
                    words = []
                elif line == '.':
                    if current is not None:
                        if len(words) != 1:
                            raise RuntimeError(f'MLF does not identify one lexical item: {path}, {current}')
                        lexical = words[0].upper()
                        if current in mapping and mapping[current] != lexical:
                            raise RuntimeError(f'Conflicting official transcript for {current}')
                        mapping[current] = lexical
                    current, words = None, []
                elif current is not None:
                    # Word MLFs contain one orthographic token; do not infer words
                    # from numeric alignments or pronunciation strings.
                    if len(line.split()) != 1:
                        raise RuntimeError(f'Unexpected timed/multitoken word MLF: {path}')
                    words.append(line)
    if not mapping:
        raise RuntimeError('No official per-speaker word transcript MLF files found')
    return mapping, files


def matched_content_subset(meta, transcripts):
    """Common lexical material and identical block/code set across every speaker.

    Uncommon (UW) IDs are intentionally excluded. The common code must map to
    exactly one verified orthographic word across the whole selected population.
    Selection is independent of diagnosis values and model predictions.
    """
    if meta.empty:
        raise ValueError('No external recordings')
    result = meta.copy()
    result['_feature_row'] = np.arange(len(meta))
    rows, rejected = [], []
    for row in result.itertuples(index=False):
        code = str(row.word_code)
        if not COMMON_CODE.fullmatch(code):
            rejected.append(dict(file_path=row.file_path, reason='excluded_noncommon_word_code'))
            continue
        key = (str(row.physical_speaker_id), str(row.block), code, str(row.microphone))
        lexical = transcripts.get(key)
        if lexical is None:
            rejected.append(dict(file_path=row.file_path, reason='missing_official_word_transcript'))
            continue
        rows.append((row.Index if hasattr(row, 'Index') else None, key, lexical))
    if not rows:
        raise ValueError('No common recordings with verified official transcripts')
    lexicon = {key: word for _, key, word in rows}
    keep, lexical = [], []
    for index, row in result.iterrows():
        key = (str(row.physical_speaker_id), str(row.block), str(row.word_code), str(row.microphone))
        if key in lexicon:
            keep.append(index); lexical.append(lexicon[key])
    result = result.loc[keep].copy()
    result['lexical_item'] = lexical
    ambiguous = set(result.groupby('word_code').lexical_item.nunique().loc[lambda x: x != 1].index)
    if ambiguous:
        rejected.extend(dict(file_path=r.file_path, reason='ambiguous_common_word_transcript')
                        for r in result[result.word_code.isin(ambiguous)].itertuples(index=False))
        result = result[~result.word_code.isin(ambiguous)].copy()
    keys = ['physical_speaker_id', 'block', 'word_code']
    if result.duplicated(keys).any():
        raise RuntimeError('Duplicate physical-speaker/block/common-word recording')
    # Three release speakers have no verified word MLF entries. Exclude them
    # by transcript availability alone, then require exactly the same prompt
    # set for every retained speaker. Diagnosis never chooses the cohort.
    speakers = set(result.physical_speaker_id)
    available = [set(zip(p.block, p.word_code)) for _, p in result.groupby('physical_speaker_id')]
    common = set.intersection(*available) if available else set()
    selected = result[[tuple(x) in common for x in result[['block', 'word_code']].to_numpy()]].copy()
    if selected.empty:
        raise ValueError('No verified block/common-word intersection across speakers with official transcripts')
    if selected.label.nunique() != 2:
        raise ValueError('Verified matched-content cohort does not contain both diagnoses')
    for _, part in selected.groupby('physical_speaker_id'):
        if set(zip(part.block, part.word_code)) != common:
            raise AssertionError('Matched-content cohort is not identical across speakers')
    summary = dict(n_physical_speakers=len(speakers), n_utterances=len(selected),
                   n_matched_prompts=len(common), n_unique_lexical_items=int(selected.lexical_item.nunique()),
                   n_common_word_codes=int(selected.word_code.nunique()),
                   same_block_and_word_set_for_every_speaker=True,
                   n_source_physical_speakers=int(meta.physical_speaker_id.nunique()),
                   speakers_excluded_missing_verified_transcripts=sorted(set(meta.physical_speaker_id)-speakers),
                   uncommon_words_excluded=True, labels_used_for_subset_selection=False,
                   n_source_recordings=len(meta), n_excluded_recordings=len(meta)-len(selected))
    return selected.reset_index(drop=True), pd.DataFrame(rejected), summary


def _predict_external(ctx, meta, speech, acoustic, *, channel, subset, source_manifest=None):
    validate_roster(ctx, meta)
    frames = []
    for seed in ctx.config.seeds:
        model, store, lineage = frozen_model(ctx, seed, meta=meta, speech=speech, acoustic=acoustic)
        frame = ctx.pipeline.predict_speakers(model, store, set(meta.physical_speaker_id), ctx.config.max_utterances_per_bag)
        for key, value in lineage.items():
            frame[key] = value
        frame['corpus'] = 'UASpeech'; frame['channel'] = channel; frame['subset'] = subset
        if source_manifest is not None:
            frame['feature_manifest'] = str(source_manifest)
            frame['feature_manifest_sha256'] = sha256(source_manifest)
        frames.append(frame)
        del model, store
    return pd.concat(frames, ignore_index=True)


def run_matched_content(ctx):
    stage_status(ctx, 'matched_content', 'running')
    directory = ctx.output_dir / 'diagnostics/matched_content'; directory.mkdir(parents=True, exist_ok=True)
    meta, speech, acoustic = external_features(ctx, 'M3')
    validate_roster(ctx, meta)
    try:
        transcripts, transcript_files = read_word_transcripts(ctx.pipeline.resolve_uaspeech_root())
        matched, excluded, summary = matched_content_subset(meta, transcripts)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        status = {'state': 'not_estimable', 'message': 'MATCHED-CONTENT ANALYSIS NOT ESTIMABLE', 'reason': str(exc)}
        write_json(directory / 'audit.json', status)
        stage_status(ctx, 'matched_content', 'not_estimable', str(exc))
        print(status['message'] + ': ' + str(exc), flush=True)
        return status
    write_csv(directory / 'matched_recordings.csv', matched)
    write_csv(directory / 'excluded_recordings.csv', excluded)
    summary.update(transcript_files=transcript_files, source_signature=ctx.pipeline.source_signature(meta),
                   matched_recordings_sha256=sha256(directory / 'matched_recordings.csv'),
                   run_signature=ctx.run_signature, external_model_selection=False)
    write_json(directory / 'audit.json', summary)
    indices = matched['_feature_row'].to_numpy(dtype=int)
    predictions = _predict_external(ctx, matched, speech[indices], acoustic[indices], channel='M3',
                                    subset='verified_common_content_all_speakers', source_manifest=directory/'audit.json')
    for name in ['n_matched_prompts', 'n_unique_lexical_items']:
        predictions[name] = summary[name]
    publish_result(ctx, 'matched_content', predictions)
    return summary


def external_feature_spec(ctx):
    return dict(ctx.manifest['feature_spec'], external_preprocessing='normalized_UASpeech_mic_unknown_trim25_no_gate_v1')


def load_frozen_encoder(ctx):
    """Construct architecture locally and load verified snapshot weights; no URL API."""
    torchaudio = ctx.pipeline.torchaudio
    bundle = torchaudio.pipelines.WAVLM_BASE
    if bundle._normalize_waveform or bundle._model_type != 'WavLM':
        raise RuntimeError('Unexpected WavLM Base bundle configuration')
    path = ctx.output_dir / 'frozen_reference/pretrained/wavlm_base.pth'
    if sha256(path) != ctx.manifest['pretrained_weight_sha256']:
        raise RuntimeError('Frozen pretrained WavLM checksum mismatch')
    model = torchaudio.models.wavlm_model(**bundle._params)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model.eval().to(ctx.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def cache_identity(row, feature_spec, *, corpus_source_signature, audio_sha256):
    return dict(relative_path=str(row['relative_path']), file_path=str(row['file_path']),
                source_id=str(row['source_id']), source_signature=corpus_source_signature,
                source_size=int(row['source_size']), source_mtime_ns=int(row['source_mtime_ns']),
                audio_sha256=audio_sha256, preprocessing_signature=canonical_hash(feature_spec),
                pretrained_weight_sha256=feature_spec['pretrained_weight_sha256'],
                encoder_configuration={'name': feature_spec['encoder'], 'selected_layers':feature_spec['selected_layers'],
                                       'statistic':'valid_frame_mean_and_population_SD', 'feature_spec':feature_spec})


def validate_shard(path, identity, acoustic_dim):
    """Fail on stale, corrupt, or mis-shaped shards; never silently rebuild them."""
    with np.load(path, allow_pickle=False) as saved:
        if str(saved['cache_key'].item()) != canonical_hash(identity) or json.loads(str(saved['identity_json'].item())) != identity:
            raise RuntimeError(f'Cached representation signature mismatch: {path}')
        embedding, acoustic = saved['embedding'], saved['acoustic']
        if embedding.shape != (4, 1536) or acoustic.shape != (acoustic_dim,) or not np.isfinite(embedding).all() or not np.isfinite(acoustic).all():
            raise RuntimeError(f'Cached representation shape/value mismatch: {path}')
        digest = canonical_hash({'embedding': __import__('hashlib').sha256(embedding.tobytes()).hexdigest(),
                                 'acoustic': __import__('hashlib').sha256(acoustic.tobytes()).hexdigest()})
        if str(saved['array_sha256'].item()) != digest:
            raise RuntimeError(f'Cached representation payload checksum mismatch: {path}')
        return embedding.copy(), acoustic.copy()


def _save_shard(ctx, path, identity, embedding, acoustic):
    digest = canonical_hash({'embedding': __import__('hashlib').sha256(embedding.tobytes()).hexdigest(),
                             'acoustic': __import__('hashlib').sha256(acoustic.tobytes()).hexdigest()})
    ctx.pipeline._atomic_npz(path, embedding=embedding, acoustic=acoustic,
                             cache_key=np.asarray(canonical_hash(identity)),
                             identity_json=np.asarray(json.dumps(identity, sort_keys=True)),
                             array_sha256=np.asarray(digest))


def cached_extract(ctx, meta, directory, feature_spec, *, waveform_loader=None, encoder=None):
    """Extract/reuse every declared recording, keyed by content and full provenance.

    Each source keeps one fixed shard location. A different key at that location
    is an error, including raw content changes with artificially preserved mtime.
    The aggregate is assembled only after every shard verifies successfully.
    """
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    meta = meta.reset_index(drop=True)
    source_signature = ctx.pipeline.source_signature(meta)
    protocol = dict(source_signature=source_signature, feature_spec=feature_spec,
                    feature_signature=canonical_hash(feature_spec), n_recordings=len(meta),
                    raw_content_hashed=True, external_model_selection=False)
    protocol_path = directory / 'cache_protocol.json'
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise RuntimeError(f'Cached representation dataset/preprocessing mismatch: {directory}')
    write_json(protocol_path, protocol)
    acoustic_dim = len(ctx.pipeline.ACOUSTIC_NAMES)
    identities, paths = [], []
    owned_encoder = False
    waveform_loader = waveform_loader or (lambda row:ctx.pipeline.preprocess_waveform(Path(row.file_path), row.mic_type, ctx.config))
    try:
        for index, row in meta.iterrows():
            raw = Path(row.file_path); current = raw.stat()
            if current.st_size != int(row.source_size) or current.st_mtime_ns != int(row.source_mtime_ns):
                raise RuntimeError(f'Audio source changed before extraction: {raw}')
            identity = cache_identity(row, feature_spec, corpus_source_signature=source_signature, audio_sha256=sha256(raw))
            identity['selection_metadata'] = {key: row[key] for key in ['segment_start_sample','segment_end_sample'] if key in row}
            # Convert numpy scalar types for exact JSON round-trip comparison.
            identity = json.loads(json.dumps(identity, default=lambda x:x.item()))
            source_key = canonical_hash({'relative_path':row.relative_path, 'source_id':row.source_id})
            shard = directory / 'embedding_shards' / source_key[:2] / (source_key + '.npz')
            if shard.exists():
                validate_shard(shard, identity, acoustic_dim)
            else:
                if encoder is None:
                    encoder = load_frozen_encoder(ctx); owned_encoder = True
                audio = waveform_loader(row)
                # Match the frozen notebook's 1024-sample minimum. Some valid
                # UASpeech clips trim below 100 ms but were padded by V8.
                if len(audio) < 1024 or not np.isfinite(audio).all():
                    raise RuntimeError(f'Invalid diagnostic waveform: {raw}')
                embedding = ctx.pipeline._encode_batch(encoder, [audio], ctx.config, ctx.device)[0].astype(np.float32)
                acoustic = ctx.pipeline.acoustic_features(audio, ctx.config.sample_rate).astype(np.float32)
                _save_shard(ctx, shard, identity, embedding, acoustic)
                validate_shard(shard, identity, acoustic_dim)
            identities.append(identity); paths.append(shard)
            if (index+1) % 500 == 0 or index+1 == len(meta):
                print(f'V8 publication features {directory.name}: {index+1}/{len(meta)} verified', flush=True)
    finally:
        if owned_encoder:
            del encoder
            gc.collect()
            # The process releases its encoder on exit. Clearing thousands of
            # length-specific MPS graphs here can stall for minutes and mask
            # the actual extraction error; leave allocator cleanup to exit.
    aggregate = directory / 'features'; aggregate.mkdir(exist_ok=True)
    speech_tmp = aggregate / 'speech_embeddings.partial.npy'; acoustic_tmp = aggregate / 'acoustic_features.partial.npy'
    speech = np.lib.format.open_memmap(speech_tmp, mode='w+', dtype='float32', shape=(len(meta), 4, 1536))
    acoustic = np.lib.format.open_memmap(acoustic_tmp, mode='w+', dtype='float32', shape=(len(meta), acoustic_dim))
    for index, (path, identity) in enumerate(zip(paths, identities)):
        speech[index], acoustic[index] = validate_shard(path, identity, acoustic_dim)
    speech.flush(); acoustic.flush(); del speech, acoustic
    os.replace(speech_tmp, aggregate/'speech_embeddings.npy'); os.replace(acoustic_tmp, aggregate/'acoustic_features.npy')
    write_csv(aggregate/'usable_metadata.csv', meta)
    write_json(aggregate/'source_content_hashes.json', identities)
    write_json(aggregate/'manifest.json', dict(complete=True, feature_spec=feature_spec,
               feature_signature=canonical_hash(feature_spec), source_signature=source_signature,
               pipeline_signature=ctx.manifest['pipeline_signature'], n_usable=len(meta),
               embedding_shape=[len(meta),4,1536], acoustic_names=ctx.pipeline.ACOUSTIC_NAMES,
               extraction_errors=[], source_content_signature=canonical_hash(identities),
               source_content_hashes_sha256=sha256(aggregate/'source_content_hashes.json'),
               speech_embeddings_sha256=sha256(aggregate/'speech_embeddings.npy'),
               acoustic_features_sha256=sha256(aggregate/'acoustic_features.npy')))
    return validate_features(ctx.pipeline, ctx.config, aggregate, feature_spec, source_signature)


def matched_channel_indices(channel_metadata):
    """Match the SAME speaker/block/word recording across all compared channels.

    UW is safe here because speaker identity is retained in the matching key; it
    is never interpreted as the same lexical item across different speakers.
    """
    key_columns = ['physical_speaker_id','block','word_code']
    key_sets = []
    for channel, meta in channel_metadata.items():
        if meta.duplicated(key_columns).any():
            raise RuntimeError(f'Duplicate speaker/block/word recording in channel {channel}')
        key_sets.append(set(map(tuple, meta[key_columns].astype(str).to_numpy())))
    common = set.intersection(*key_sets) if key_sets else set()
    if not common:
        raise ValueError('No common physical-speaker/block/word recordings across compared channels')
    indices = {channel:np.flatnonzero([tuple(k) in common for k in meta[key_columns].astype(str).to_numpy()])
               for channel, meta in channel_metadata.items()}
    # Invariant also prevents channel-dependent cohort changes after matching.
    for channel, meta in channel_metadata.items():
        actual = set(map(tuple, meta.iloc[indices[channel]][key_columns].astype(str).to_numpy()))
        if actual != common:
            raise AssertionError('Matched channel content mismatch')
    return indices, sorted(common)


def run_channels(ctx, channels=CHANNELS):
    channels = tuple(channels)
    if 'M3' not in channels or len(set(channels)) != len(channels) or any(c not in CHANNELS for c in channels):
        raise ValueError('Channels must be unique M2–M8 including prespecified primary M3')
    stage_status(ctx, 'channels', 'running', channels=list(channels))
    features, frames, channel_counts = {}, [], []
    root = ctx.pipeline.resolve_uaspeech_root()
    for channel in channels:
        if channel == 'M3':
            features[channel] = external_features(ctx,'M3')
            feature_manifest = ctx.run_dir/'external_uaspeech/M3/features/manifest.json'
            inventory_count = len(features[channel][0]); excluded_count = 39
        else:
            directory = ctx.output_dir/'external_cache'/channel
            meta, issues = ctx.pipeline.inventory_uaspeech(root, channel, expected_full=False)
            validate_roster(ctx,meta)
            write_csv(directory/'raw_inventory.csv',meta); write_csv(directory/'excluded_audio.csv',issues)
            features[channel] = cached_extract(ctx,meta,directory,external_feature_spec(ctx))
            feature_manifest = directory/'features/manifest.json'
            inventory_count = len(meta); excluded_count = len(issues)
        meta,speech,acoustic = features[channel]
        frame = _predict_external(ctx,meta,speech,acoustic,channel=channel,subset='all_available',source_manifest=feature_manifest)
        frame['primary_channel'] = channel=='M3'
        frames.append(frame)
        channel_counts.append(dict(channel=channel,n_physical_speakers=int(meta.physical_speaker_id.nunique()),
                                   n_recordings=len(meta),n_valid_inventory=inventory_count,n_excluded=excluded_count,
                                   prespecified_primary=channel=='M3'))
        # Publish partial progress with explicit per-channel names, so a resumed
        # extraction retains already completed descriptive evaluations.
        publish_result(ctx,'channel_'+channel,frame)
    all_available = pd.concat(frames,ignore_index=True)
    publish_result(ctx,'channels',all_available)
    directory=ctx.output_dir/'diagnostics/channels';write_csv(directory/'all_available_counts.csv',pd.DataFrame(channel_counts))
    indices, common = matched_channel_indices({c:f[0] for c,f in features.items()})
    write_csv(directory/'matched_recording_keys.csv',pd.DataFrame(common,columns=['physical_speaker_id','block','word_code']))
    matched_frames=[];matched_counts=[]
    for channel,(meta,speech,acoustic) in features.items():
        rows=indices[channel];selected=meta.iloc[rows].reset_index(drop=True)
        frame=_predict_external(ctx,selected,speech[rows],acoustic[rows],channel=channel,subset='matched_across_channels',
                                source_manifest=directory/'matched_recording_keys.csv')
        frame['primary_channel']=channel=='M3';matched_frames.append(frame)
        matched_counts.append(dict(channel=channel,n_physical_speakers=int(selected.physical_speaker_id.nunique()),n_recordings=len(selected),
                                   n_matched_recording_keys=len(common),prespecified_primary=channel=='M3'))
    publish_result(ctx,'channels_matched',pd.concat(matched_frames,ignore_index=True))
    write_csv(directory/'matched_counts.csv',pd.DataFrame(matched_counts))
    audit=dict(primary_channel='M3',primary_channel_prespecified=True,channels=list(channels),
               n_matched_recording_keys=len(common),external_model_selection=False,
               external_threshold_selection=False,external_calibration=False,
               comparison='descriptive_frozen_model_cross_corpus',complete=set(channels)==set(CHANNELS))
    write_json(directory/'audit.json',audit)
    if set(channels)!=set(CHANNELS):stage_status(ctx,'channels','partial','Only a subset of M2–M8 was evaluated',channels=list(channels))
    return audit


def edge_background_segment(audio, sample_rate, protocol=SILENCE_PROTOCOL):
    """Select a real contiguous quiet edge without labels, VAD fitting, or padding."""
    audio=np.asarray(audio,dtype=np.float32)
    if audio.ndim!=1 or not len(audio) or not np.isfinite(audio).all():raise ValueError('Invalid raw mono audio')
    frame=max(1,round(protocol['frame_seconds']*sample_rate))
    n_frames=len(audio)//frame
    if n_frames==0:return None
    frames=audio[:n_frames*frame].reshape(n_frames,frame)
    rms=np.sqrt(np.mean(frames.astype(np.float64)**2,axis=1))
    limit=float(np.max(np.abs(audio)))*10**(protocol['relative_peak_db']/20)
    low=rms<=limit
    leading=next((i for i,v in enumerate(low) if not v),n_frames)
    trailing=next((i for i,v in enumerate(low[::-1]) if not v),n_frames)
    candidates=[]
    if leading:candidates.append((0,leading*frame))
    if trailing:candidates.append(((n_frames-trailing)*frame,n_frames*frame))
    if not candidates:return None
    start,end=max(candidates,key=lambda x:(x[1]-x[0],-x[0]))
    end=min(end,start+int(protocol['maximum_seconds']*sample_rate))
    if end-start < int(protocol['minimum_seconds']*sample_rate):return None
    return int(start),int(end)


def _raw_mono(ctx,path):
    audio,sr=ctx.pipeline.sf.read(path,dtype='float32',always_2d=True);audio=audio.mean(axis=1)
    if sr!=ctx.config.sample_rate:
        audio=ctx.pipeline.librosa.resample(audio,orig_sr=sr,target_sr=ctx.config.sample_rate)
    return np.asarray(audio,dtype=np.float32)


def select_background_recordings(ctx,meta,corpus,protocol=SILENCE_PROTOCOL):
    """Scan a hash-ordered list until 50 eligible clips per speaker, independent of labels."""
    selected,rows=[],[]
    for speaker,part in meta.groupby('physical_speaker_id',sort=True):
        part=part.assign(_order=part.relative_path.map(lambda p:canonical_hash({'background_source':p}))).sort_values('_order')
        n_selected=0
        for _,row in part.iterrows():
            base=dict(corpus=corpus,physical_speaker_id=speaker,file_path=row.file_path,relative_path=row.relative_path)
            if n_selected>=protocol['eligible_clips_per_speaker']:
                rows.append(base|{'status':'not_scanned_cap_reached'});continue
            stat=Path(row.file_path).stat()
            if stat.st_size!=int(row.source_size) or stat.st_mtime_ns!=int(row.source_mtime_ns):
                raise RuntimeError(f'Audio source changed before background selection: {row.file_path}')
            try:
                audio=_raw_mono(ctx,row.file_path);segment=edge_background_segment(audio,ctx.config.sample_rate,protocol)
            except (OSError,ValueError) as exc:
                rows.append(base|{'status':'decode_failed','reason':str(exc)});continue
            if segment is None:
                rows.append(base|{'status':'no_eligible_edge'});continue
            start,end=segment
            data=row.drop(labels=['_order']).to_dict();data.update(segment_start_sample=start,segment_end_sample=end,
                         background_seconds=(end-start)/ctx.config.sample_rate)
            selected.append(data);n_selected+=1
            rows.append(base|{'status':'selected','background_seconds':(end-start)/ctx.config.sample_rate})
    return pd.DataFrame(selected),pd.DataFrame(rows)


def run_silence(ctx,protocol=None):
    protocol=dict(SILENCE_PROTOCOL if protocol is None else protocol)
    directory=ctx.output_dir/'diagnostics/silence';directory.mkdir(parents=True,exist_ok=True)
    protocol_path=directory/'protocol.json'
    if protocol_path.exists() and read_json(protocol_path)!=protocol:raise RuntimeError('Prespecified background protocol changed')
    write_json(protocol_path,protocol)
    stage_status(ctx,'silence','running')
    frames=[];counts=[];not_estimable=[]
    external_meta,_,_=external_features(ctx,'M3')
    for corpus,meta in [('TORGO',ctx.meta),('UASpeech',external_meta)]:
        selected,selection=select_background_recordings(ctx,meta,corpus,protocol)
        write_csv(directory/f'{corpus}_selection_audit.csv',selection)
        write_csv(directory/f'{corpus}_selected_recordings.csv',selected)
        for speaker,part in selection.groupby('physical_speaker_id'):
            counts.append(dict(corpus=corpus,physical_speaker_id=speaker,n_available_recordings=len(part),
                               n_selected=int(part.status.eq('selected').sum()),n_no_eligible_edge=int(part.status.eq('no_eligible_edge').sum()),
                               n_decode_failed=int(part.status.eq('decode_failed').sum()),n_unscanned=int(part.status.eq('not_scanned_cap_reached').sum())))
        if selected.empty or selected.label.nunique()!=2:
            not_estimable.append(f'{corpus}: fewer than two diagnosis classes have eligible real background recordings');continue
        spec=dict(ctx.manifest['feature_spec'],background_protocol=protocol)
        if corpus=='UASpeech':spec['external_preprocessing']='objective_background_only_v1'
        def load_segment(row):
            audio=_raw_mono(ctx,row.file_path)[int(row.segment_start_sample):int(row.segment_end_sample)]
            peak=float(np.max(np.abs(audio)))
            return (audio/peak if peak>0 else audio).astype(np.float32)
        cache=ctx.output_dir/'external_cache'/('silence_'+corpus)
        usable,speech,acoustic=cached_extract(ctx,selected,cache,spec,waveform_loader=load_segment)
        feature_manifest=cache/'features/manifest.json'
        if corpus=='UASpeech':
            prediction=_predict_external(ctx,usable,speech,acoustic,channel='M3',subset='objective_background_only',source_manifest=feature_manifest)
            prediction['diagnostic']='silence_background';frames.append(prediction)
        else:
            for seed in ctx.config.seeds:
                seen=set()
                for outer in range(ctx.config.outer_folds):
                    audit=read_json(ctx.run_dir/'folds/speech_only'/f'seed{seed}_outer{outer}'/'fold_audit.json')
                    speakers=set(audit['test_physical_speakers'])&set(usable.physical_speaker_id)
                    if not speakers:continue
                    if speakers&set(audit['train_physical_speakers']) or speakers&seen:raise AssertionError('TORGO background OOF leakage')
                    seen|=speakers
                    model,store,lineage=frozen_model(ctx,seed,outer,meta=usable,speech=speech,acoustic=acoustic)
                    prediction=ctx.pipeline.predict_speakers(model,store,speakers,ctx.config.max_utterances_per_bag)
                    for key,value in lineage.items():prediction[key]=value
                    prediction['corpus']='TORGO';prediction['channel']='original_TORGO';prediction['subset']='objective_background_only'
                    prediction['diagnostic']='silence_background';prediction['feature_manifest']=str(feature_manifest)
                    prediction['feature_manifest_sha256']=sha256(feature_manifest);frames.append(prediction)
                    del model,store
                if seen!=set(usable.physical_speaker_id):raise AssertionError('Background OOF predictions omitted eligible TORGO speakers')
    write_csv(directory/'eligible_speaker_counts.csv',pd.DataFrame(counts))
    if frames:
        predictions=pd.concat(frames,ignore_index=True)
        publish_result(ctx,'silence',predictions)
        metrics=pd.read_csv(ctx.output_dir/'results/silence/per_seed_metrics.csv')
        findings=[]
        for corpus,part in metrics.groupby('corpus'):
            mean=float(part.ROC_AUC.mean())
            findings.append(dict(corpus=corpus,mean_ROC_AUC=mean,
                possible_recording_confound=mean>=.7,
                interpretation='Potential recording-artifact concern; diagnostic only, no model adjustment' if mean>=.7 else
                'Background-only discrimination below the prespecified descriptive flag; this does not exclude other confounds'))
        write_json(directory/'audit.json',dict(protocol=protocol,not_estimable=not_estimable,findings=findings,
                   flag_rule='descriptive mean ROC-AUC >= 0.70; not a significance test',
                   caveat='Objective low-energy edges are not guaranteed speech-free; selected background is peak-normalized as predeclared.',
                   labels_used_for_selection=False,model_or_threshold_changed=False))
        if not_estimable:stage_status(ctx,'silence','partial','; '.join(not_estimable))
        return {'complete':not not_estimable,'findings':findings,'not_estimable':not_estimable}
    reason='; '.join(not_estimable)
    write_json(directory/'audit.json',dict(state='not_estimable',reason=reason,protocol=protocol))
    stage_status(ctx,'silence','not_estimable',reason)
    return {'complete':False,'not_estimable':not_estimable}


def run(ctx,stages=('matched_content','channels','silence')):
    results={}
    for name in stages:
        if name=='matched_content':results[name]=run_matched_content(ctx)
        elif name=='channels':results[name]=run_channels(ctx)
        elif name=='silence':results[name]=run_silence(ctx)
        else:raise ValueError(f'Unknown external diagnostic stage: {name}')
    return results
