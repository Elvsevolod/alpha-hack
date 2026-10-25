"""Development-only CPU cost probe for a small FT-style Transformer.

Measures random tensors, not quality. It does not reproduce the tuned paper.
"""
from pathlib import Path
import sys,time,json,resource
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from automl_agent.util import atomic_json

def run(path):
    torch.set_num_threads(1);torch.manual_seed(42);results=[]
    for features,rows in [(21,20000),(96,52903),(1123,4862)]:
        batch=32;width=32
        scales=nn.Parameter(torch.randn(features,width)*.01)
        model=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,2,64,.1,batch_first=True),1,enable_nested_tensor=False)
        head=nn.Linear(width,6)
        optimizer=torch.optim.AdamW([scales,*model.parameters(),*head.parameters()],lr=.001)
        elapsed=[]
        for step in range(4):
            x=torch.randn(batch,features);begin=time.monotonic();optimizer.zero_grad()
            z=model(x[:,:,None]*scales[None,:,:]);loss=head(z.mean(1)).square().mean();loss.backward();optimizer.step()
            if step:elapsed.append(time.monotonic()-begin)
        cost=float(np.median(elapsed));epoch=cost*np.ceil(rows/batch)
        results.append(dict(features=features,fit_rows=rows,batch_size=batch,width=width,layers=1,
            median_training_step_seconds=cost,estimated_epoch_seconds=epoch,
            estimated_30_epoch_fit_seconds=30*epoch,estimated_60_epoch_fit_seconds=60*epoch,estimated_two_fit_seconds=120*epoch,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            quality_measured=False,decision='eligible_for_30_epoch_screen' if 30*epoch<90 and 120*epoch<1000 else 'defer_exceeds_screen_fit_or_worker_budget',
            limitation='Tiny numeric token cost probe; preprocessing, categorical tokens, validation and inference add cost. No paper-recipe equivalence.'))
        print(results[-1],flush=True)
    atomic_json(Path(path),dict(results=results,threads=1,api_calls=0))

if __name__=='__main__':run(sys.argv[1])
