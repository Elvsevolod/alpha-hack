"""Resource measurements on one CPU; results retain environment and corpus identity."""
import argparse
import gc
import importlib.metadata
from pathlib import Path
import platform
import resource
import sqlite3
import statistics
import time

from .core import canonical, read_json, requirements, write_json
from .runtime import ReferenceArgs, ReferenceStore


def percentile(values,p):
    values=sorted(values);return values[max(0,__import__('math').ceil(len(values)*p)-1)]


def benchmark(corpus, queries, output, requirement_file=None):
    import psutil
    process=psutil.Process();gc.collect();baseline=process.memory_info().rss
    start=time.perf_counter();store=ReferenceStore(corpus)
    first=store.query(query=queries[0],max_chars=6000)
    cold=time.perf_counter()-start
    resident=process.memory_info().rss-baseline
    timings={}
    for fts,name,ceiling in ((True,'fts5',.5),(False,'lexical',2.)):
        store.use_fts=fts;seconds=[];empty=0
        for i in range(max(100,len(queries))):
            query=queries[i%len(queries)]
            began=time.perf_counter();hits=store.search(ReferenceArgs(query=query));seconds.append(time.perf_counter()-began)
            empty+=not bool(hits)
        timings[name]={'queries':len(seconds),'p50_seconds':statistics.median(seconds),'p95_seconds':percentile(seconds,.95),'max_seconds':max(seconds),'empty':empty,'budget_passed':percentile(seconds,.95)<=ceiling}
    store.close();gc.collect()
    versions={}
    if requirement_file:
        for package,version in requirements(requirement_file).items():
            try:actual=importlib.metadata.version(package).split('+')[0]
            except importlib.metadata.PackageNotFoundError:actual=None
            versions[package]={'expected':version,'actual':actual,'matches':actual==version}
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1024 if platform.system()=='Linux' else 1)
    total=sum(path.stat().st_size for path in Path(corpus).rglob('*') if path.is_file())
    report={'corpus':read_json(Path(corpus)/'knowledge.lock.json')['corpus'],'environment':{'python':platform.python_version(),'sqlite':sqlite3.sqlite_version,'platform':platform.platform(),'cpu_affinity':process.cpu_affinity() if hasattr(process,'cpu_affinity') else None},
            'installed_versions':versions,'all_required_versions_exact':bool(versions) and all(r['matches'] for r in versions.values()),
            'open_and_first_search_seconds':cold,'restoration_included':False,'resident_increase_bytes':resident,'peak_process_rss_bytes':peak,
            'corpus_bytes':total,'queries':timings,'budgets':{'disk':total<=1024**3,'resident':resident<=256*1024**2,'peak':peak<=1024**3,'open_first_search':cold<=30},
            'first_status':first['status']}
    write_json(output,report);print(canonical({k:v for k,v in report.items() if k not in {'installed_versions'}}))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--root',type=Path,default=Path(__file__).parent)
    a=p.parse_args();q=[x[0] for x in read_json(a.root/'evaluation/legacy40.json')];benchmark(a.corpus,q,a.output,a.root/'requirements.txt')
