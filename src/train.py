import json,time,platform,sys,copy
from pathlib import Path
import numpy as np,pandas as pd,sklearn,torch,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from .data import download,load,windows,features,hashes
from .metrics import evaluate,bootstrap_rmse
from .models import LSTMRegressor
ROOT=Path(__file__).resolve().parents[1]
def main():
    cfg=json.loads((ROOT/'config.json').read_text());out=ROOT/'results';out.mkdir(exist_ok=True);models=out/'models';models.mkdir(exist_ok=True)
    raw=ROOT/'data/raw';download(raw);tr,te,truth=load(raw)
    rng=np.random.default_rng(cfg['split_seed']);ids=rng.permutation(tr.engine.unique());vi=sorted(ids[:cfg['validation_engines']].tolist());ti=sorted(ids[cfg['validation_engines']:].tolist())
    a,b=tr[tr.engine.isin(ti)],tr[tr.engine.isin(vi)]
    sensors=[f's{i}' for i in range(1,22) if a[f's{i}'].std()>1e-8]
    scale=StandardScaler().fit(a[sensors]);joblib.dump(scale,models/'scaler.joblib')
    kw=dict(sensors=sensors,scaler=scale,length=cfg['window'],cap=cfg['cap'])
    x,y,_,_=windows(a,**kw,stride=cfg['stride'])
    # Fixed pseudo-censoring of held-out engines, independent of test labels.
    vrng=np.random.default_rng(2026);cuts={int(e):int(vrng.integers(max(1,int(len(g)*.35)),max(2,int(len(g)*.9)))) for e,g in b.groupby('engine')}
    vb=pd.concat([g[g.cycle<=cuts[int(e)]] for e,g in b.groupby('engine')]);xv,yv,_,_=windows(vb,**kw,endpoints=True)
    xt,_,test_ids,_=windows(te,**kw,endpoints=True)
    manifest=dict(config=cfg,train_engines=ti,validation_engines=vi,validation_cuts=cuts,sensors=sensors,train_rows=len(tr),test_rows=len(te),training_windows=len(x),hashes=hashes(raw),python=sys.version,platform=platform.platform(),processor=platform.processor(),versions={'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__},target='min(raw remaining cycles,125)',test_metrics='also reported against uncapped official truth')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[];predictions=pd.DataFrame({'engine':test_ids,'truth_raw':truth,'truth_capped':np.minimum(truth,cfg['cap'])});tuning=[]
    def record(name,model,trainx,valx,testx,seed,elapsed):
        vp=np.clip(model.predict(valx),0,cfg['cap']);p=np.clip(model.predict(testx),0,cfg['cap'])
        m=evaluate(np.minimum(truth,cfg['cap']),p);rawm=evaluate(truth,p)
        row={'model':name,'seed':seed,'fit_seconds':elapsed,'val_rmse':evaluate(yv,vp)['rmse'],**m,'raw_mae':rawm['mae'],'raw_rmse':rawm['rmse'],'rmse_ci_low':bootstrap_rmse(np.minimum(truth,cfg['cap']),p)[0],'rmse_ci_high':bootstrap_rmse(np.minimum(truth,cfg['cap']),p)[1]}
        rows.append(row);predictions[f'{name}_{seed}']=p;print(json.dumps(row),flush=True)
    for hist in [False,True]:
        xx,vv,tt=[features(z,hist) for z in [x,xv,xt]];name='Linear_history' if hist else 'Linear_current'
        start=time.perf_counter();m=LinearRegression().fit(xx,y);elapsed=time.perf_counter()-start
        record(name,m,xx,vv,tt,42,elapsed);joblib.dump(m,models/f'{name}.joblib')
    bests={}
    for hist in [False,True]:
        xx,vv,tt=[features(z,hist) for z in [x,xv,xt]];name='RF_history' if hist else 'RF_current';candidates=[]
        for params in cfg['forest_candidates']:
            m=RandomForestRegressor(**params,random_state=42,n_jobs=2);start=time.perf_counter();m.fit(xx,y);elapsed=time.perf_counter()-start;score=evaluate(yv,np.clip(m.predict(vv),0,cfg['cap']))['rmse'];tuning.append(dict(model=name,params=params,val_rmse=score));candidates.append((score,params,m,elapsed))
        _,params,m,elapsed=min(candidates,key=lambda z:z[0]);bests[name]=params
        record(name,m,xx,vv,tt,42,elapsed);joblib.dump(m,models/f'{name}_42.joblib')
        if hist:
            for seed in cfg['training_seeds'][1:]:
                start=time.perf_counter();m=RandomForestRegressor(**params,random_state=seed,n_jobs=2).fit(xx,y);elapsed=time.perf_counter()-start;record(name,m,xx,vv,tt,seed,elapsed);joblib.dump(m,models/f'{name}_{seed}.joblib')
    candidates=[]
    for params in cfg['lstm_candidates']:
        m=LSTMRegressor(**params,epochs=cfg['epochs'],patience=cfg['patience'],batch=cfg['batch']);start=time.perf_counter();m.fit(x,y,xv,yv);elapsed=time.perf_counter()-start;score=evaluate(yv,np.clip(m.predict(xv),0,cfg['cap']))['rmse'];tuning.append(dict(model='LSTM',params=params,val_rmse=score,best_epoch=m.best_epoch));candidates.append((score,params,m,elapsed))
    _,params,m,elapsed=min(candidates,key=lambda z:z[0]);bests['LSTM']=params
    for seed in cfg['training_seeds']:
        if seed!=42:
            start=time.perf_counter();m=LSTMRegressor(**params,epochs=cfg['epochs'],patience=cfg['patience'],batch=cfg['batch'],seed=seed).fit(x,y,xv,yv);elapsed=time.perf_counter()-start
        record('LSTM',m,x,xv,xt,seed,elapsed);torch.save(m.net.state_dict(),models/f'LSTM_{seed}.pt');pd.DataFrame(m.history).to_csv(out/f'lstm_history_{seed}.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);predictions.to_csv(out/'predictions.csv',index=False);(out/'tuning.json').write_text(json.dumps({'candidates':tuning,'selected':bests},indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()
