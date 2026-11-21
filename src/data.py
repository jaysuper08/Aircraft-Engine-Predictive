"""NASA FD001 loading and causal windows. No test-derived preprocessing."""
from pathlib import Path
import hashlib, io, urllib.request, zipfile
import numpy as np
import pandas as pd
URL='https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip'
COLS=['engine','cycle']+[f'op{i}' for i in range(1,4)]+[f's{i}' for i in range(1,22)]
def download(root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    if all((root/n).exists() for n in ['train_FD001.txt','test_FD001.txt','RUL_FD001.txt','readme.txt']): return
    outer=zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL,timeout=120).read()))
    nested=next(n for n in outer.namelist() if n.endswith('CMAPSSData.zip'))
    inner=zipfile.ZipFile(io.BytesIO(outer.read(nested)))
    for name in ['train_FD001.txt','test_FD001.txt','RUL_FD001.txt','readme.txt']:
        (root/name).write_bytes(inner.read(name))
def load(root):
    root=Path(root)
    def read(name):
        d=pd.read_csv(root/name,sep=r'\s+',header=None,names=COLS)
        assert d.shape[1]==26 and not d.isna().any().any()
        assert not d.duplicated(['engine','cycle']).any()
        return d
    train,test=read('train_FD001.txt'),read('test_FD001.txt')
    train['rul']=train.groupby('engine').cycle.transform('max')-train.cycle
    truth=np.loadtxt(root/'RUL_FD001.txt')
    assert len(truth)==test.engine.nunique()
    return train,test,truth
def windows(frame,sensors,scaler,length=30,cap=125,stride=1,endpoints=False):
    xs,ys,ids,cycles=[],[],[],[]
    for engine,g in frame.groupby('engine',sort=True):
        g=g.sort_values('cycle'); a=scaler.transform(g[sensors]).astype('float32')
        ends=[len(g)-1] if endpoints else range(0,len(g),stride)
        for j in ends:
            w=a[max(0,j-length+1):j+1]
            if len(w)<length: w=np.concatenate([np.repeat(w[:1],length-len(w),axis=0),w])
            xs.append(w);ids.append(int(engine));cycles.append(int(g.iloc[j].cycle))
            ys.append(min(float(g.iloc[j].rul),cap) if 'rul' in g else np.nan)
    return np.asarray(xs),np.asarray(ys,dtype='float32'),np.asarray(ids),np.asarray(cycles)
def features(x,history=True):
    if not history:return x[:,-1,:]
    t=np.arange(x.shape[1],dtype='float32');t-=t.mean()
    slope=np.einsum('ntf,t->nf',x,t)/(t@t)
    return np.concatenate([x[:,-1],x.mean(1),x.std(1),slope],axis=1)
def hashes(root):
    return {n:hashlib.sha256((Path(root)/n).read_bytes()).hexdigest() for n in ['train_FD001.txt','test_FD001.txt','RUL_FD001.txt']}
