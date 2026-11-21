"""Independent checks of persisted metrics, data identity and fitted inference."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,joblib,torch
from src.data import load,windows,features,hashes
from src.metrics import evaluate
from src.models import LSTMNet
root=Path(__file__).resolve().parents[1];m=pd.read_csv(root/'results/metrics.csv');p=pd.read_csv(root/'results/predictions.csv');manifest=json.loads((root/'results/manifest.json').read_text());cfg=manifest['config']
assert hashes(root/'data/raw')==manifest['hashes']
assert len(p)==100 and not p.engine.duplicated().any()
for _,r in m.iterrows():
    pred=p[f'{r.model}_{int(r.seed)}'].to_numpy();score=evaluate(p.truth_capped.to_numpy(),pred)
    for k in ['mae','rmse','bias','overestimate_pct','near_mae']:np.testing.assert_allclose(score[k],r[k],rtol=1e-6,atol=1e-6)
    raw=evaluate(p.truth_raw.to_numpy(),pred);np.testing.assert_allclose(raw['rmse'],r.raw_rmse)
tr,te,truth=load(root/'data/raw');sc=joblib.load(root/'results/models/scaler.joblib');x,_,ids,_=windows(te,manifest['sensors'],sc,length=cfg['window'],cap=cfg['cap'],endpoints=True)
for name in ['Linear_history','RF_history']:
    suffix='_42' if name=='RF_history' else '';model=joblib.load(root/f'results/models/{name}{suffix}.joblib');pred=np.clip(model.predict(features(x)),0,125);np.testing.assert_allclose(pred,p[f'{name}_42'],rtol=1e-5,atol=1e-5)
params=json.loads((root/'results/tuning.json').read_text())['selected']['LSTM'];net=LSTMNet(len(manifest['sensors']),params['hidden']);net.load_state_dict(torch.load(root/'results/models/LSTM_42.pt',weights_only=True));net.eval()
with torch.no_grad():pred=np.clip(net(torch.from_numpy(x)).numpy()*125,0,125)
np.testing.assert_allclose(pred,p.LSTM_42,rtol=1e-5,atol=1e-4)
assert not tr.isna().any().any() and not te.isna().any().any()
print('PASS: all nine metric rows recomputed; hashes match; saved linear, forest and LSTM inference matches; 100 unique test engines.')
