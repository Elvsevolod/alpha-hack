"""Version-2 deterministic finalist and closed-control selection policy."""
from .util import digest

POLICY_VERSION = 3


def component_audit(c, finalists, policies):
    """Audit actual frozen recipes and every parent edge, before opening control."""
    from .contracts import PipelineSpec
    from .recipe_patch import removed_components
    from .decision import frozen_decision_valid
    warnings, identities = [], {}
    for finalist in finalists:
        pending, seen = [finalist], set()
        while pending:
            identifier = pending.pop()
            if identifier in seen:
                continue
            seen.add(identifier)
            spec, _ = c.lookup(identifier)
            identities[identifier] = spec.hash
            frozen = spec.pipeline
            if identifier in policies:
                frozen = PipelineSpec.model_validate(policies[identifier]["frozen_pipeline"])
                missing = removed_components(spec.pipeline, frozen, frozen=True)
                if spec.pipeline.prediction_policy == "class_offsets" and not frozen_decision_valid(frozen, c.task):
                    missing.append("class_offsets")
                if missing:
                    warnings.append({"finalist_id": finalist, "experiment_id": identifier,
                        "source": "frozen_pipeline", "missing_fields": sorted(set(missing))})
            descendant, acknowledged, ancestry = spec, set(), set()
            while descendant.parent_id and descendant.experiment_id not in ancestry:
                ancestry.add(descendant.experiment_id)
                acknowledged.update(descendant.hypothesis.changed_fields)
                parent, _ = c.lookup(descendant.parent_id)
                identities[parent.experiment_id] = parent.hash
                still_missing = removed_components(parent.pipeline, frozen, frozen=True)
                missing = [name for name in removed_components(parent.pipeline, descendant.pipeline)
                    if name not in acknowledged and name in still_missing]
                if missing:
                    warnings.append({"finalist_id": finalist, "experiment_id": descendant.experiment_id,
                        "parent_id": parent.experiment_id, "source": "unacknowledged_parent_removal",
                        "missing_fields": missing})
                descendant = parent
            pending.extend(spec.pipeline.members)
    audit = {"protocol_hash": c.plan.hash, "leader_id": c.journal.get("leader_id"),
        "finalists": finalists, "recipes": identities,
        "frozen_hashes": {key: value["pipeline_hash"] for key, value in policies.items()}, "warnings": warnings}
    return {**audit, "component_audit_hash": digest(audit)}


def affordable_incumbent(candidates, available, maximize):
    eligible = [row for row in candidates if row['seconds'] <= available]
    return min(eligible, key=lambda row: (-row['score'] if maximize else row['score'],
        row['seconds'], row['id']))['id'] if eligible else None


def select_finalists(candidates, baseline, leader, policy, windows, limit=3):
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
    requested_windows = list(windows)
    anchors_all = [by_id[identifier] for identifier in selected]
    hashes = {row.get('windows_hash') for row in anchors_all}
    protocols = {row.get('protocol') for row in anchors_all}
    common_identity = bool(anchors_all and len(hashes) == len(protocols) == 1 and None not in hashes
        and None not in protocols and '' not in hashes and '' not in protocols)
    windows_hash = next(iter(hashes)) if common_identity else None
    protocol = next(iter(protocols)) if common_identity else None
    windows = sorted(set(windows).intersection(*(set(row['windows']) for row in anchors_all))) if common_identity else []
    complete = lambda row: (bool(windows) and row.get('windows_hash') == windows_hash
        and row.get('protocol') == protocol and set(windows) <= set(row['windows']))
    anchors = [by_id[identifier] for identifier in selected if complete(by_id[identifier])]
    audit = {'version':POLICY_VERSION, 'windows':windows, 'requested_windows':requested_windows,
        'coverage':{row['id']:sorted(row['windows']) for row in candidates},
        'windows_hash':windows_hash, 'protocol':protocol,
        'completeness':{row['id']:row.get('window_status','unavailable') for row in candidates},
        'anchors':selected[:], 'pool':[], 'mode':'limited_validation'}
    better = lambda a,b: a>b if policy.maximize else a<b
    no_worse = lambda a,b: a>=b if policy.maximize else a<=b
    if windows and len(anchors)==len(selected) and selected:
        eligible=[]
        for row in pool:
            if not complete(row):continue
            dominated=len(windows) >= 2 and any(all(no_worse(anchor['windows'][w],row['windows'][w]) for w in windows)
                and no_worse(anchor['score'],row['score'])
                and (better(anchor['score'],row['score']) or any(better(anchor['windows'][w],row['windows'][w]) for w in windows))
                for anchor in anchors)
            if not dominated:eligible.append(row)
        pool=eligible;audit['mode']='common_temporal_windows' if len(windows) >= 2 else 'partial_common_window'
        for row in pool:
            ranks=[1+sum(better(other['windows'][w],row['windows'][w]) for other in pool) for w in windows]
            row={**row,'key':(max(ranks),sum(ranks)/len(ranks),-row['score'] if policy.maximize else row['score'],row['cost'],row['created_index'],row['id'])}
            audit['pool'].append(row)
    else:
        best=(max if policy.maximize else min)((row['score'] for row in candidates),default=None)
        for row in pool:
            if best is not None and policy.tied(row['score'],best):
                audit['pool'].append({**row,'key':(-row['score'] if policy.maximize else row['score'],row['cost'],row['created_index'],row['id'])})
    if audit['pool'] and len(selected) < limit:
        selected.append(min(audit['pool'],key=lambda row:row['key'])['id'])
    # The closed control is the only future-period guard. Always admit the
    # strongest distinct recipes, even when validation ranks them below the
    # incumbent; otherwise a near-period winner can lock out the better model.
    top_up = []
    if len(selected) < limit:
        ranked = sorted((row for row in candidates if row['id'] not in selected),
            key=lambda row: ((-row['score'] if policy.maximize else row['score']),
                row.get('cost', 0.), row.get('created_index', 0), row['id']))
        seen = {by_id[identifier]['identity'] for identifier in selected if identifier in by_id}
        for row in ranked:
            if len(selected) >= limit:
                break
            if row['identity'] in seen:
                continue
            selected.append(row['id']); seen.add(row['identity']); top_up.append(row['id'])
    audit['control_top_up'] = top_up
    audit['limit'] = limit
    audit['selected']=selected
    audit['audit_hash']=digest(audit)
    return selected, audit


def choose_control(scores, baseline, policy, frozen_costs, created):
    best=(max if policy.maximize else min)(scores.values())
    tied=[identifier for identifier,score in scores.items() if policy.tied(score,best)]
    winner=baseline if baseline in tied else min(tied,key=lambda identifier:(frozen_costs[identifier],created[identifier],identifier))
    return winner, {'version':POLICY_VERSION,'best_control':best,'tie_set':sorted(tied),
        'reference':'exact_best_control','frozen_refit_costs':frozen_costs}
