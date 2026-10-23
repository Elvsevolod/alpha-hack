"""Version-2 deterministic finalist and closed-control selection policy."""
from .util import digest

POLICY_VERSION = 2


def select_finalists(candidates, baseline, leader, policy, windows):
    """Candidates already have immutable recipe identity and pre-control costs."""
    by_id = {row['id']: row for row in candidates}
    selected, identities = [], set()
    for identifier in (baseline, leader):
        if identifier in by_id and by_id[identifier]['identity'] not in identities:
            selected.append(identifier);identities.add(by_id[identifier]['identity'])
    pool = [row for row in candidates if row['identity'] not in identities]
    unique = {}
    for row in sorted(pool, key=lambda row:(row['created_index'],row['id'])):
        unique.setdefault(row['identity'], row)
    pool = list(unique.values())
    complete = lambda row: bool(windows) and set(row['windows']) == set(windows)
    anchors = [by_id[identifier] for identifier in selected if complete(by_id[identifier])]
    audit = {'version':POLICY_VERSION, 'windows':windows, 'anchors':selected[:], 'pool':[], 'mode':'limited_validation'}
    better = lambda a,b: a>b if policy.maximize else a<b
    no_worse = lambda a,b: a>=b if policy.maximize else a<=b
    if len(windows)>=2 and len(anchors)==len(selected) and selected:
        eligible=[]
        for row in pool:
            if not complete(row):continue
            dominated=any(all(no_worse(anchor['windows'][w],row['windows'][w]) for w in windows)
                and no_worse(anchor['score'],row['score'])
                and (better(anchor['score'],row['score']) or any(better(anchor['windows'][w],row['windows'][w]) for w in windows))
                for anchor in anchors)
            if not dominated:eligible.append(row)
        pool=eligible;audit['mode']='common_temporal_windows'
        for row in pool:
            ranks=[1+sum(better(other['windows'][w],row['windows'][w]) for other in pool) for w in windows]
            row={**row,'key':(max(ranks),sum(ranks)/len(ranks),-row['score'] if policy.maximize else row['score'],row['cost'],row['created_index'],row['id'])}
            audit['pool'].append(row)
    else:
        best=(max if policy.maximize else min)((row['score'] for row in candidates),default=None)
        for row in pool:
            if best is not None and policy.tied(row['score'],best):
                audit['pool'].append({**row,'key':(-row['score'] if policy.maximize else row['score'],row['cost'],row['created_index'],row['id'])})
    if audit['pool']:
        selected.append(min(audit['pool'],key=lambda row:row['key'])['id'])
    audit['selected']=selected
    audit['audit_hash']=digest(audit)
    return selected, audit


def choose_control(scores, baseline, policy, frozen_costs, created):
    best=(max if policy.maximize else min)(scores.values())
    tied=[identifier for identifier,score in scores.items() if policy.tied(score,best)]
    winner=baseline if baseline in tied else min(tied,key=lambda identifier:(frozen_costs[identifier],created[identifier],identifier))
    return winner, {'version':POLICY_VERSION,'best_control':best,'tie_set':sorted(tied),
        'reference':'exact_best_control','frozen_refit_costs':frozen_costs}
