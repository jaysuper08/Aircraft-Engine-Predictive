"""Regenerate all figures from recorded results and original data."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import load
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
def main():
    out=ROOT/'results';fig=out/'figures';fig.mkdir(exist_ok=True)
    tr,te,_=load(ROOT/'data/raw');metrics=pd.read_csv(out/'metrics.csv');p=pd.read_csv(out/'predictions.csv')
    f,axs=plt.subplots(1,2,figsize=(10,3.4));axs[0].hist(tr.groupby('engine').cycle.max(),bins=15,color='#205b80');axs[0].set(xlabel='Training engine lifetime (cycles)',ylabel='Engines')
    for e in [1,2,3]:
        g=tr[tr.engine==e];axs[1].plot(g.cycle,g.s11.rolling(10,min_periods=1).mean(),label=f'Engine {e}')
    axs[1].set(xlabel='Operating cycle',ylabel='Sensor 11 (10-cycle mean)');axs[1].legend();f.tight_layout();f.savefig(fig/'eda.pdf');plt.close(f)
    main=metrics[(metrics.seed==42)&metrics.model.isin(['Linear_history','RF_history','LSTM'])]
    f,axs=plt.subplots(1,2,figsize=(10,3.5));x=np.arange(len(main));axs[0].bar(x,main.rmse,color=['#8b9da7','#205b80','#e79b3a']);axs[0].set(xticks=x,xticklabels=['Linear','Forest','LSTM'],ylabel='Capped-target test RMSE (cycles)')
    for name in main.model:
        axs[1].scatter(p.truth_capped,p[f'{name}_42'],s=13,alpha=.65,label=name.replace('_history',''))
    axs[1].plot([0,125],[0,125],'k--',lw=1);axs[1].set(xlabel='True capped RUL',ylabel='Predicted RUL');axs[1].legend(fontsize=8);f.tight_layout();f.savefig(fig/'comparison.pdf');plt.close(f)
    f,axs=plt.subplots(1,2,figsize=(10,3.5));
    for name in main.model:
        err=p[f'{name}_42']-p.truth_capped;axs[0].scatter(p.truth_capped,err,s=13,alpha=.6,label=name.replace('_history',''))
    axs[0].axhline(0,c='k',ls='--',lw=1);axs[0].set(xlabel='True capped RUL',ylabel='Prediction error (cycles)');axs[0].legend(fontsize=8)
    h=pd.read_csv(out/'lstm_history_42.csv');axs[1].plot(h.epoch,np.sqrt(h.train_mse)*125,label='Train window RMSE');axs[1].plot(h.epoch,h.val_rmse,label='Validation endpoint RMSE');axs[1].set(xlabel='Epoch',ylabel='RMSE (cycles)');axs[1].legend(fontsize=8);f.tight_layout();f.savefig(fig/'diagnostics.pdf');plt.close(f)
    features=['last','mean','std','slope'];import joblib
    m=joblib.load(out/'models/RF_history_42.joblib');sensors=json.loads((out/'manifest.json').read_text())['sensors'];names=[f'{s} {kind}' for kind in features for s in sensors];idx=np.argsort(m.feature_importances_)[-10:]
    f,ax=plt.subplots(figsize=(7,3.6));ax.barh(np.array(names)[idx],m.feature_importances_[idx],color='#205b80');ax.set(xlabel='Impurity-based importance (descriptive, not causal)');f.tight_layout();f.savefig(fig/'importance.pdf');plt.close(f)
if __name__=='__main__':main()
