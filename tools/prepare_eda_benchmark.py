"""Reproducible development fixtures from UCI originals; never included in submission."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import httpx
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from automl_agent.util import atomic_json, file_hash

DATASETS = {"adult": (2,"adult"), "bank": (222,"bank+marketing"),
    "bike": (275,"bike+sharing+dataset"), "har": (240,"human+activity+recognition+using+smartphones")}


def download(url, path):
    if not path.exists():
        response = httpx.get(url, timeout=90, follow_redirects=True)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path.read_bytes()


def prepare(root):
    root = root.resolve()
    sources = root / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    originals, manifest = {}, {}
    for name, (number, slug) in DATASETS.items():
        directory = sources / name
        directory.mkdir(exist_ok=True)
        url = f"https://archive.ics.uci.edu/static/public/{number}/{slug}.zip"
        data = download(url, directory / "original.zip")
        page = f"https://archive.ics.uci.edu/dataset/{number}/{slug}"
        download(page, directory / "uci.html")
        # Preserve primary license text with attribution page and originals.
        license_url = "https://creativecommons.org/licenses/by/4.0/legalcode.txt"
        download(license_url, directory / "LICENSE.txt")
        archive = zipfile.ZipFile(io.BytesIO(data))
        originals[name] = archive
        for member in archive.namelist():
            if member.lower().endswith(('.names','.txt')) and 'inertial' not in member.lower():
                # Preserve descriptions safely without extracting untrusted archive paths.
                safe = member.replace('/', '__').replace('\\','__')
                (directory / safe).write_bytes(archive.read(member))
        manifest[name] = {"url": url, "page": page, "license": "CC BY 4.0", "license_url": license_url,
            "sha256": hashlib.sha256(data).hexdigest(), "members": archive.namelist()}
    adult_cols = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation',
        'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
    a = originals['adult']
    adult = pd.concat([pd.read_csv(io.BytesIO(a.read(member)), names=adult_cols, skipinitialspace=True,
        comment='|', na_values=['?']) for member in ['adult.data','adult.test']], ignore_index=True)
    adult.income = adult.income.str.rstrip('.')
    adult_cats = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
    bank_archive = zipfile.ZipFile(io.BytesIO(originals['bank'].read('bank.zip')))
    bank = pd.read_csv(io.BytesIO(bank_archive.read('bank-full.csv')), sep=';')
    (sources/'bank'/'bank-names.txt').write_bytes(bank_archive.read('bank-names.txt'))
    bike = pd.read_csv(io.BytesIO(originals['bike'].read('hour.csv')))
    bike = bike.rename(columns={'instant':'id'})
    # Keep original date as structural service field, model inputs are documented calendar/weather fields.
    definitions = {
        'adult': (adult,'income',adult_cats,[],None,None,
            'Predict annual income class (<=50K or >50K) for census records. Primary metric: Macro F1. Rows are independent census observations.'),
        'bank': (bank,'y',list(bank.select_dtypes('object').columns.drop('y')),['duration'],None,None,
            'Before the current marketing phone call, predict subscription y (yes/no). Primary metric: Macro F1. duration is the final duration of that call and is unavailable at decision time; it is not supplied in the input files. pdays=-1 means no prior contact. Unknown is a recorded category.'),
        'bike': (bike,'cnt',['season','yr','mnth','hr','holiday','weekday','workingday','weathersit'],
            ['dteday','casual','registered'],'dteday',None,
            'Predict hourly total rentals cnt in later calendar periods. Primary metric: RMSE. dteday is the calendar day used for temporal splitting. casual and registered are post-event components whose sum is cnt; they are not supplied in the input files. Calendar codes are categorical; temp, atemp, hum, windspeed are numeric.'),
    }
    h = originals['har']
    if not any(n.endswith('/features.txt') for n in h.namelist()):
        h = zipfile.ZipFile(io.BytesIO(h.read(next(n for n in h.namelist() if n.lower().endswith('.zip')))))
    prefix = next(n[:-len('features.txt')] for n in h.namelist() if n.endswith('/features.txt'))
    raw_names = [line.split(maxsplit=1)[1] for line in h.read(prefix+'features.txt').decode().splitlines()]
    names = [f'f{i+1:03d}_{name}' for i,name in enumerate(raw_names)]
    def har(part):
        X = pd.read_csv(io.BytesIO(h.read(prefix+f'{part}/X_{part}.txt')), sep=r'\s+', header=None, names=names)
        X['subject'] = pd.read_csv(io.BytesIO(h.read(prefix+f'{part}/subject_{part}.txt')), header=None)[0]
        X['activity'] = pd.read_csv(io.BytesIO(h.read(prefix+f'{part}/y_{part}.txt')), header=None)[0].astype(str)
        return X
    har_train, har_test = har('train'), har('test')
    definitions['har'] = (har_train,'activity',[],['subject'],None,'subject',
        'Classify activity of new volunteers from smartphone sensor windows. Primary metric: Macro F1. Activity labels: 1 WALKING, 2 WALKING_UPSTAIRS, 3 WALKING_DOWNSTAIRS, 4 SITTING, 5 STANDING, 6 LAYING. subject identifies a participant and is a group/service column, never a predictive input. Internal validation must use disjoint subjects. All 561 f-prefixed sensor summaries are numerical features. External test retains the official participant split.')
    for name,(frame,target,cats,service,time_col,group_col,description) in definitions.items():
        frame = frame.copy()
        if 'id' not in frame:
            frame.insert(0,'id',np.arange(len(frame)))
        if name in {'adult','bank'}:
            train_idx,test_idx = train_test_split(np.arange(len(frame)),test_size=.2,random_state=42,stratify=frame[target])
            train,test = frame.iloc[np.sort(train_idx)].copy(),frame.iloc[np.sort(test_idx)].copy()
            rule = 'stratified outer 80/20 seed=42; train cap stratified seed=42'
            if len(train)>20000:
                selected,_=train_test_split(np.arange(len(train)),train_size=20000,random_state=42,stratify=train[target])
                train=train.iloc[np.sort(selected)].copy()
        elif name=='bike':
            days=sorted(frame.dteday.unique()); cutoff=days[-math.ceil(.2*len(days))]
            train=frame[frame.dteday<cutoff].copy();test=frame[frame.dteday>=cutoff].copy()
            rule=f'last ceil(20% distinct days) external test; first test day={cutoff}; no train cap needed'
        else:
            train,test=frame,har_test.copy();test.insert(0,'id',np.arange(len(test))+len(train))
            assert not set(train.subject)&set(test.subject)
            rule='official disjoint volunteer split; no train cap needed'
        directory=root/'tasks'/name;directory.mkdir(parents=True,exist_ok=True)
        sealed=root/'sealed'/name;sealed.mkdir(parents=True,exist_ok=True)
        test[['id',target]].to_csv(sealed/'labels.csv',index=False)
        # Unavailable post-event fields are absent altogether in agent files.
        remove=[c for c in service if c not in {time_col,group_col}]
        train=train.drop(columns=remove);test=test.drop(columns=remove)
        service=[c for c in service if c not in remove]
        features=[c for c in train if c not in ['id',target]+service]
        train.to_csv(directory/'train.csv',index=False)
        test.drop(columns=[target]).to_csv(directory/'test.csv',index=False)
        sample=test[['id']].copy();sample[target]=0 if name=='bike' else str(sorted(train[target].unique())[0]);sample.to_csv(directory/'sample_submission.csv',index=False)
        descriptions=[]
        for c in train:
            role='target' if c==target else 'identifier' if c=='id' else 'service' if c in service else 'feature'
            dtype='categorical' if c in cats or c==target and name!='bike' else 'numeric'
            if c==time_col:dtype='date'
            desc=f'{role}; {dtype}'
            if c=='subject':desc+='; participant group for disjoint-subject validation'
            if c=='dteday':desc+='; calendar day for chronological validation'
            descriptions.append({'column':c,'description':desc,'role':role,'dtype':dtype})
        pd.DataFrame(descriptions).to_csv(directory/'data_dictionary.csv',index=False)
        (directory/'task.md').write_text(description+'\n\nFiles: train.csv contains labels, test.csv has no labels. data_dictionary.csv documents columns. sample_submission.csv fixes ordered output: id, '+target+'. Use labels for classification and values for regression. Source: '+manifest[name]['page']+' (CC BY 4.0).\n')
        reference={'kind':'regression' if name=='bike' else 'classification','targets':[target],'id_columns':['id'],
            'feature_columns':features,'categorical_columns':cats,'service_columns':service,'time_column':time_col,'group_column':group_col,
            'metric':{'name':'rmse' if name=='bike' else 'f1','average':'macro'},
            'output':{'columns':['id',target],'prediction_type':'values' if name=='bike' else 'labels'},
            'explicit_validation':'temporal' if time_col else 'group' if group_col else 'random'}
        atomic_json(sealed/'reference_contract.json',reference)
        manifest[name].update(train_rows=len(train),test_rows=len(test),subsampling=rule,
            prepared_sha256={p.name:file_hash(p) for p in directory.iterdir() if p.is_file()},
            expected_checks={'contract':reference,'tools_not_prescribed':True})
    atomic_json(root/'manifest.json',manifest)
    return {name:{k:item[k] for k in ['train_rows','test_rows','sha256']} for name,item in manifest.items()}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(prepare(args.output),indent=2))
