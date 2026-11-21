import numpy as np
def evaluate(y,p):
    e=np.asarray(p)-np.asarray(y)
    near=np.asarray(y)<=30
    return dict(mae=float(np.abs(e).mean()),rmse=float(np.sqrt((e**2).mean())),bias=float(e.mean()),overestimate_pct=float(100*(e>0).mean()),near_n=int(near.sum()),near_mae=float(np.abs(e[near]).mean()) if near.any() else None)
def bootstrap_rmse(y,p,seed=2026,repeats=2000):
    rng=np.random.default_rng(seed); e=np.asarray(p)-y
    v=[np.sqrt(np.mean(e[rng.integers(0,len(e),len(e))]**2)) for _ in range(repeats)]
    return list(map(float,np.percentile(v,[2.5,97.5])))
