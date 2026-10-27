"""Build one channel's signed WavLM cache before frozen-model evaluation."""
from __future__ import annotations
import argparse
from .common import make_context,write_csv
from .external_diagnostics import CHANNELS,cached_extract,external_feature_spec,validate_roster

def prefetch(channel,device='mps'):
    if channel not in CHANNELS or channel=='M3':
        raise ValueError('Choose one secondary channel from M2,M4,M5,M6,M7,M8')
    ctx=make_context(verify=False,device=device)
    meta,issues=ctx.pipeline.inventory_uaspeech(ctx.pipeline.resolve_uaspeech_root(),channel,expected_full=False)
    validate_roster(ctx,meta)
    directory=ctx.output_dir/'external_cache'/channel
    write_csv(directory/'raw_inventory.csv',meta)
    write_csv(directory/'excluded_audio.csv',issues)
    cached_extract(ctx,meta,directory,external_feature_spec(ctx))
    print(f'PREFETCH {channel} COMPLETE {len(meta)}',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('channel',choices=tuple(x for x in CHANNELS if x!='M3'))
    parser.add_argument('--device',default='mps')
    options=parser.parse_args()
    prefetch(options.channel,options.device)
if __name__=='__main__':main()
