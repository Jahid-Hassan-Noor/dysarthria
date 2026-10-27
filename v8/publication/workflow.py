"""Resume the V8 publication-completion protocol using frozen artifacts."""
from __future__ import annotations
import argparse
from pathlib import Path
from .common import PUBLICATION_ROOT, make_context, register_primary, read_json

STAGES=('primary','ablations','budgets','baselines','matched_content','silence','channels','report')

def run_stage(name,ctx):
    if name=='primary':return register_primary(ctx)
    if name=='ablations':
        if (ctx.output_dir/'results/ablations/provenance.json').exists():return {'reused':'ablations'}
        from .ablations import run
        return run(ctx)
    if name=='budgets':
        from .budgets import run_corpus
        return {corpus:run_corpus(ctx,corpus) for corpus in ('TORGO','UASpeech')}
    if name=='baselines':
        from .baselines import run_kind
        return {kind:run_kind(ctx,kind) for kind in ('duration','artifact')}
    if name in ('matched_content','silence','channels'):
        from . import external_diagnostics as diagnostics
        return {'matched_content':diagnostics.run_matched_content,
                'silence':diagnostics.run_silence,'channels':diagnostics.run_channels}[name](ctx)
    if name=='report':
        from .reporting import generate_publication_report
        return generate_publication_report(ctx.output_dir)
    raise ValueError(name)

def run_publication_completion(stages=STAGES,*,publication_root=PUBLICATION_ROOT,device=None):
    ctx=make_context(publication_root,device=device)
    results={}
    for name in stages:
        if name not in STAGES:raise ValueError(f'Unknown stage: {name}')
        print(f'V8 publication stage: {name}',flush=True)
        results[name]=run_stage(name,ctx)
    return results

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stages',nargs='*',choices=STAGES,default=['report'])
    parser.add_argument('--device',default=None)
    parser.add_argument('--publication-root',type=Path,default=PUBLICATION_ROOT)
    options=parser.parse_args()
    run_publication_completion(options.stages,publication_root=options.publication_root,device=options.device)
if __name__=='__main__':main()
