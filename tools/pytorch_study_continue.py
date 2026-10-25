"""Continue the owned sequential study after its FT screening worker finishes."""
from pathlib import Path
import json,time
from tools.pytorch_study_pool import pool
from tools.pytorch_study_repeat import repeat

def main():
    marker=Path('outputs/pytorch_study_ft42/bike_watchdog.json')
    while not marker.is_file():time.sleep(2)
    for name in ['regression','adult','bank','bike']:
        q=json.loads((marker.parent/(name+'_watchdog.json')).read_text())
        if q['returncode']!=0:print('Additional screening incomplete:',name,q,flush=True)
    pool('outputs/pytorch_study_final_screen','outputs/pytorch_study_ft42','outputs/pytorch_study_pooled42')
    repeat('outputs/pytorch_study_pooled42','outputs/pytorch_study_repeats')

if __name__=='__main__':main()
