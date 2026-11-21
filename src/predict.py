"""Predict using a saved model and a 26-column NASA-format input file."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,joblib,torch
from .data import COLS,windows,features
from .models import LSTMNet
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--model',choices=['Linear_history','RF_history','LSTM'],default='RF_history');ap.add_argument('--output',default='predictions_new.csv');args=ap.parse_args()
    d=pd.read_csv(args.input,sep=r'\s+',header=None,names=COLS);m=json.loads((ROOT/'results/manifest.json').read_text());cfg=m['config'];scale=joblib.load(ROOT/'results/models/scaler.joblib');x,_,ids,_=windows(d,m['sensors'],scale,length=cfg['window'],cap=cfg['cap'],endpoints=True)
    if args.model=='LSTM':
        selected=json.loads((ROOT/'results/tuning.json').read_text())['selected']['LSTM'];net=LSTMNet(len(m['sensors']),selected['hidden']);net.load_state_dict(torch.load(ROOT/'results/models/LSTM_42.pt',weights_only=True));net.eval()
        with torch.no_grad():p=net(torch.from_numpy(x)).numpy()*cfg['cap']
    else:
        suffix='_42' if args.model=='RF_history' else '';model=joblib.load(ROOT/f'results/models/{args.model}{suffix}.joblib');p=model.predict(features(x))
    pd.DataFrame({'engine':ids,'predicted_capped_rul':np.clip(p,0,cfg['cap'])}).to_csv(args.output,index=False)
if __name__=='__main__':main()
