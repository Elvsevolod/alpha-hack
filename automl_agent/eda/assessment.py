"""Feature-only protocol diagnostics and compact fit-target facts for EDA decisions."""
from ..validation import ordered_periods


def target_overview(profile):
    result={}
    for name,item in list(profile.get('targets',{}).items())[:3]:
        if item.get('semantic_type')=='categorical':
            counts=[entry['rows'] for entry in item.get('counts',[])]
            positive=[n for n in counts if n>0]
            result[name]={'classes':len(counts),'unobserved_declared_classes':sum(n==0 for n in counts),
                'minimum_class_rows':min(counts,default=0),'maximum_class_rows':max(counts,default=0),
                'observed_count_ratio':max(positive)/min(positive) if positive else None}
        else:
            result[name]={k:item[k] for k in ['median','q01','q99','zero_fraction','negative_fraction'] if k in item}
    return result


def protocol_assessment(dataset,plan):
    parts={name:list(getattr(plan,name)) for name in ['fit','validation','control']}
    memberships={name:set(rows) for name,rows in parts.items()}
    pairs=[('fit','validation'),('fit','control'),('validation','control')]
    result={'protocol_hash':plan.hash,'strategy':plan.strategy,'rows':{k:len(v) for k,v in parts.items()},
        'partitions_disjoint':all(not memberships[a]&memberships[b] for a,b in pairs),
        'external_test_rows':len(dataset.test),'warnings':[]}
    if len(parts['validation'])<30:
        result['warnings'].append({'code':'low_validation_support','observation':'Fewer than 30 validation rows.',
            'limitation':'One small aggregate is unstable; engineering threshold, not a confidence guarantee.'})
    if dataset.task.time_column:
        column=dataset.task.time_column
        periods={name:ordered_periods(dataset.train.iloc[rows][column]) for name,rows in parts.items()}
        test=ordered_periods(dataset.test[column])
        counts={name:int(values.nunique()) for name,values in periods.items()}
        result['periods']={**counts,'external_test':int(test.nunique())}
        result['chronological']=all(periods[a].max()<periods[b].min() for a,b in pairs)
        result['test_follows_train']=bool(test.min()>ordered_periods(dataset.train[column]).max())
        if result['test_follows_train'] and result['periods']['external_test']>counts['validation']:
            result['warnings'].append({'code':'temporal_horizon_mismatch',
                'observation':f"Validation covers {counts['validation']} periods, external transfer covers {result['periods']['external_test']}.",
                'limitation':'Short-horizon errors do not establish long-horizon transfer. Protocol is unchanged.'})
    if dataset.task.group_column:
        column=dataset.task.group_column
        groups={name:set(dataset.train.iloc[rows][column]) for name,rows in parts.items()}
        result['groups']={name:len(values) for name,values in groups.items()}
        result['groups']['external_test']=int(dataset.test[column].nunique())
        result['groups_disjoint']=all(not groups[a]&groups[b] for a,b in pairs)
        result['external_groups_disjoint']=not bool(set(dataset.train[column])&set(dataset.test[column]))
    return result


def research_boundary_context(protocol):
    """The EDA model never scores outer validation; do not confuse it with fit-inner support."""
    return {key:protocol[key] for key in ('protocol_hash','strategy','partitions_disjoint',
        'external_test_rows','chronological','test_follows_train','groups_disjoint','external_groups_disjoint') if key in protocol} | {
        'scored_scope':'fit_inner_holdout_only', 'outer_validation_and_control':'closed_and_not_scored',
        'support_source':'Use each experiment evaluation_rows/evaluation_periods, not outer partition sizes.',
        'external_test_periods':protocol.get('periods',{}).get('external_test')}
