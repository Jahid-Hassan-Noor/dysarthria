"""Complete locked channel scoring and the publication report after cache builders finish."""
from __future__ import annotations
import json
import time
from .common import PUBLICATION_ROOT
from .workflow import run_publication_completion

CHANNELS=('M2','M4','M5','M6','M7','M8')

def ready():
    for channel in CHANNELS:
        path=PUBLICATION_ROOT/'external_cache'/channel/'features/manifest.json'
        if not path.exists():return False
        try:
            if json.loads(path.read_text()).get('complete') is not True:return False
        except (OSError,ValueError):return False
    return True

def main():
    while not ready():
        time.sleep(30)
    print('All six secondary channel caches complete; scoring frozen models.',flush=True)
    results=run_publication_completion(stages=('channels','report'),device='mps')
    audit=results['report']['audit']
    if not audit['complete']:
        raise RuntimeError('Final V8 publication audit is incomplete; inspect publication_audit.txt')
    print('V8 PUBLICATION COMPLETION AUDIT PASS',flush=True)

if __name__=='__main__':main()
