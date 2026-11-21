"""Run with python -m unittest discover -s tests -v."""
import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.data import load,windows,features
from src.metrics import evaluate
class PipelineTests(unittest.TestCase):
    def setUp(self):self.tr,self.te,self.truth=load('data/raw')
    def test_counts_labels(self):
        self.assertEqual(self.tr.shape[0],20631);self.assertEqual(self.te.shape[0],13096)
        self.assertTrue((self.tr.groupby('engine').tail(1).rul==0).all());self.assertEqual(len(self.truth),100)
    def test_causality_and_padding(self):
        g=self.tr[self.tr.engine==1].copy();s=['s2','s3'];sc=StandardScaler().fit(g[s])
        x,y,ids,c=windows(g,s,sc,length=30)
        changed=g.copy();changed.loc[changed.cycle>20,'s2']+=10000
        xx,*_=windows(changed,s,sc,length=30)
        np.testing.assert_array_equal(x[:20],xx[:20]);self.assertEqual(x.shape[1:],(30,2))
        np.testing.assert_array_equal(x[0,0],x[0,-1]);self.assertTrue((y<=125).all())
    def test_features_and_metrics(self):
        x=np.tile(np.arange(30)[None,:,None],(2,1,3)).astype('float32');f=features(x)
        np.testing.assert_allclose(f[:,-3:],1,atol=1e-6)
        self.assertEqual(evaluate(np.array([0,2]),np.array([0,2]))['rmse'],0)
    def test_engine_split(self):
        import json
        from pathlib import Path
        m=json.loads(Path('results/manifest.json').read_text());self.assertFalse(set(m['train_engines'])&set(m['validation_engines']))
        self.assertEqual(len(m['train_engines'])+len(m['validation_engines']),100)
if __name__=='__main__':unittest.main()
