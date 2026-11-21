"""Common fit/predict API; LSTM uses actual backpropagation through time."""
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
class LSTMNet(nn.Module):
    def __init__(self,inputs,hidden):
        super().__init__();self.lstm=nn.LSTM(inputs,hidden,batch_first=True);self.head=nn.Linear(hidden,1)
    def forward(self,x):
        z,_=self.lstm(x);return self.head(z[:,-1]).squeeze(-1)
class LSTMRegressor:
    def __init__(self,hidden=32,lr=.001,epochs=35,batch=256,patience=7,seed=42,cap=125):
        self.hidden,self.lr,self.epochs,self.batch,self.patience,self.seed,self.cap=hidden,lr,epochs,batch,patience,seed,cap
    def predict(self,x):
        self.net.eval()
        with torch.no_grad():return self.net(torch.from_numpy(x)).numpy()*self.cap
    def fit(self,x,y,xv,yv):
        torch.manual_seed(self.seed);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        self.net=LSTMNet(x.shape[-1],self.hidden)
        opt=torch.optim.Adam(self.net.parameters(),lr=self.lr,weight_decay=1e-5)
        gen=torch.Generator().manual_seed(self.seed)
        loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y/self.cap)),batch_size=self.batch,shuffle=True,generator=gen)
        best=float('inf');wait=0;self.history=[];state=None
        for epoch in range(self.epochs):
            self.net.train();total=0
            for xb,yb in loader:
                opt.zero_grad();loss=((self.net(xb)-yb)**2).mean();loss.backward();nn.utils.clip_grad_norm_(self.net.parameters(),1.0);opt.step();total+=loss.item()*len(xb)
            rmse=float(np.sqrt(np.mean((np.clip(self.predict(xv),0,self.cap)-yv)**2)))
            self.history.append({'epoch':epoch+1,'train_mse':total/len(x),'val_rmse':rmse})
            if rmse<best-1e-5:best=rmse;state=copy.deepcopy(self.net.state_dict());self.best_epoch=epoch+1;wait=0
            else:wait+=1
            if wait>=self.patience:break
        self.net.load_state_dict(state);return self
