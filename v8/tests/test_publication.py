"""Scientific-control checks for saved-result methodology."""
from __future__ import annotations
import unittest
import tempfile
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from sklearn.metrics import average_precision_score, roc_auc_score
from v8.publication.ablations import _metric_values, make_ablation_head
from v8.publication.budgets import project_utterances, projected_probabilities, sampling_seed
from v8.publication.common import load_pipeline
from v8.publication.external_diagnostics import cache_identity, cached_extract, matched_content_subset

class PublicationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.pipeline=load_pipeline()

    def test_mean_pool_preserves_classifier_width_and_reference_layer_inputs(self):
        torch.manual_seed(4)
        speech=torch.randn(6,4,1536);base=self.pipeline.SpeakerAttentionClassifier(4,1536,1,False).eval()
        A1=make_ablation_head(self.pipeline.SpeakerAttentionClassifier,'A1')
        torch.manual_seed(4);mean=A1(4,1536,1,False).eval()
        self.assertEqual(base.classifier[0].weight.shape,mean.classifier[0].weight.shape)
        self.assertFalse(mean.layer_logits.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in mean.attention.parameters()))
        with torch.no_grad():
            original=mean(speech,None)
            altered=speech.clone();altered[:,:-1,:]+=1000
            self.assertTrue(torch.allclose(original,mean(altered,None),atol=1e-7))
            last=altered.clone();last[:,-1,:]+=1000
            self.assertFalse(torch.allclose(original,mean(last,None)))

    def test_budget_projection_and_fixed_sampling_match_frozen_head(self):
        torch.manual_seed(5)
        model=self.pipeline.SpeakerAttentionClassifier(4,1536,1,False).eval()
        speech=np.random.default_rng(5).normal(size=(11,4,1536)).astype('float32')
        projected=project_utterances(model,speech)
        for budget in (1,5,10):
            selection=np.random.default_rng(sampling_seed('TORGO','D1',42,7)).choice(11,budget,replace=False)
            p=projected_probabilities(model,projected,selection[None,:])[0]
            with torch.no_grad():direct=torch.sigmoid(model(torch.from_numpy(speech[selection]))).item()
            self.assertAlmostEqual(float(p),direct,places=6)
        self.assertEqual(sampling_seed('TORGO','D1',42,7),sampling_seed('TORGO','D1',42,7))
        self.assertNotEqual(sampling_seed('TORGO','D1',42,7),sampling_seed('TORGO','D1',42,8))

    def test_paired_bootstrap_metric_parity_with_ties(self):
        y=np.array([[0,0,1,1,1],[0,0,1,1,1]])
        p=np.array([[.1,.5,.5,.8,.9],[.9,.1,.2,.2,.2]])
        threshold=np.full(p.shape,.5)
        values=_metric_values(y,p,threshold)
        for i in range(2):
            self.assertAlmostEqual(values[i,5],roc_auc_score(y[i],p[i]),places=12)
            self.assertAlmostEqual(values[i,6],average_precision_score(y[i],p[i]),places=12)

    def test_matched_content_excludes_missing_mlf_speakers_without_using_labels(self):
        rows=[];transcripts={}
        for speaker,label in [('C1',0),('C2',0),('D1',1),('D2',1),('Cmissing',0)]:
            for word in ('CW1','CW2'):
                rows.append(dict(file_path=f'{speaker}_{word}.wav',physical_speaker_id=speaker,
                                 label=label,block='B1',word_code=word,microphone='M3'))
                if speaker!='Cmissing':transcripts[(speaker,'B1',word,'M3')]=word
        chosen,_,summary=matched_content_subset(pd.DataFrame(rows),transcripts)
        self.assertEqual(summary['n_matched_prompts'],2)
        self.assertEqual(summary['n_physical_speakers'],4)
        self.assertEqual(set(chosen.label),{0,1})
        self.assertEqual(summary['speakers_excluded_missing_verified_transcripts'],['Cmissing'])

    def test_external_cache_key_changes_with_audio_or_weights(self):
        row=pd.Series(dict(relative_path='F02/a.wav',file_path='/tmp/a.wav',source_id='source',source_size=7,source_mtime_ns=12))
        spec=dict(encoder='WAVLM_BASE',selected_layers=4,pretrained_weight_sha256='weight-a')
        a=cache_identity(row,spec,corpus_source_signature='source-a',audio_sha256='audio-a')
        b=cache_identity(row,spec,corpus_source_signature='source-a',audio_sha256='audio-b')
        c=cache_identity(row,dict(spec,pretrained_weight_sha256='weight-b'),corpus_source_signature='source-a',audio_sha256='audio-a')
        self.assertNotEqual(a,b);self.assertNotEqual(a,c)

    def test_external_cache_rejects_changed_audio_even_if_stat_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);wave=root/'one.wav'
            sf.write(wave,np.sin(np.arange(3200)*.03).astype('float32'),16000)
            stat=wave.stat()
            meta=pd.DataFrame([dict(relative_path='one.wav',file_path=str(wave),source_id='frozen-source',
                                    source_size=stat.st_size,source_mtime_ns=stat.st_mtime_ns,
                                    physical_speaker_id='C1',label=0,mic_type='external')])
            ctx=SimpleNamespace(pipeline=self.pipeline,config=self.pipeline.Config(),device=torch.device('cpu'),
                                manifest={'pipeline_signature':self.pipeline.PIPELINE_SIGNATURE})
            spec=self.pipeline._feature_spec(ctx.config,'frozen-weight')
            with (patch.object(self.pipeline,'_encode_batch',return_value=np.zeros((1,4,1536),dtype='float32')),
                  patch.object(self.pipeline,'acoustic_features',return_value=np.zeros(len(self.pipeline.ACOUSTIC_NAMES),dtype='float32'))):
                cached_extract(ctx,meta,root/'cache',spec,encoder=object(),
                               waveform_loader=lambda row:np.ones(1257,dtype='float32'))
                sf.write(wave,np.cos(np.arange(3200)*.03).astype('float32'),16000)
                self.assertEqual(wave.stat().st_size,stat.st_size)
                os.utime(wave,ns=(stat.st_atime_ns,stat.st_mtime_ns))
                with self.assertRaisesRegex(RuntimeError,'Cached representation signature mismatch'):
                    cached_extract(ctx,meta,root/'cache',spec,encoder=object(),
                                   waveform_loader=lambda row:np.ones(1257,dtype='float32'))

if __name__=='__main__':unittest.main()
