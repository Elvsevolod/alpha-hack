"""Read-only admission decision from frozen three-seed development evidence."""
from pathlib import Path
import argparse,json
import numpy as np
from automl_agent.util import atomic_json,file_hash


def assess(screen,repeats):
    screen,repeats=Path(screen),Path(repeats)
    freeze=json.loads((repeats/'selection_freeze.json').read_text());cases=[];pending=[]
    for name,models in freeze['selected'].items():
        path=screen/name/'canonical_analysis.json'
        if file_hash(path)!=freeze['screen_hashes'][name]:raise ValueError('Frozen screening evidence changed: '+name)
        sources={42:json.loads(path.read_text())}
        for seed in [17,2026]:
            p=repeats/str(seed)/name/'canonical_analysis.json'
            if p.is_file():sources[seed]=json.loads(p.read_text())
        for model in models:
            rows=[]
            for seed in [17,42,2026]:
                result=next((r for r in sources.get(seed,{}).get('results',[]) if r['model']==model),None)
                if result:rows.append(result)
            if len(rows)!=3:
                pending.append(dict(task=name,model=model,completed_seeds=[r['seed'] for r in rows]));continue
            matched=[r['signed_gain'] for r in rows];conservative=[r['gain_vs_reference_envelope'] for r in rows]
            passed=bool(np.mean(conservative)>0 and sum(g>0 for g in conservative)>=2)
            cases.append(dict(task=name,model=model,seeds=[r['seed'] for r in rows],
                baseline_mean=float(np.mean([r['baseline'] for r in rows])),score_mean=float(np.mean([r['score'] for r in rows])),
                matched_gains=matched,mean_matched_gain=float(np.mean(matched)),
                gains_vs_reference_envelope=conservative,mean_conservative_gain=float(np.mean(conservative)),
                positive_seeds=sum(g>0 for g in conservative),passed=passed,weights=[r['weight'] for r in rows]))
    return dict(passed_any=any(r['passed'] for r in cases),cases=cases,pending=pending,
        selection_freeze_sha256=file_hash(repeats/'selection_freeze.json'),control_used=False,
        rule='positive average development gain and >=2/3 positive seeds; conservative check also exceeds prespecified baseline argmax/fit-only-offset envelope',
        uncertainty='Same validation rows across seeds; neither independent datasets nor a statistical guarantee.',api_calls=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--screen',type=Path,required=True);p.add_argument('--repeats',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();q=assess(a.screen,a.repeats);atomic_json(a.output,q);print(json.dumps(q,ensure_ascii=False))
