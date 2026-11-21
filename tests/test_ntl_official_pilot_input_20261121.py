import unittest
import tempfile
from pathlib import Path
import pandas as pd
from scripts.ntl_official_pilot_input_20261121 import load_official

MODELS=['claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
        'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash',
        'glm-5.3','glm-5.3-flash','gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra','kimi-k3','qwen3.8-max']


class TestLoader(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        rows=[dict(id=f'r{i}',prompt_id=f'p{i%5}',model=m,text=f'sentence {m}')
              for i,m in enumerate(MODELS)]
        self.tr=pd.DataFrame(rows)
        self.fo=self.tr[['id','prompt_id','model']].copy()
        self.fo['fold_id'] = list(range(5))*3+[0,1,2]
        self.tr.to_csv(self.root/'train.csv',index=False)
        self.fo.to_csv(self.root/'folds.csv',index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def load(self):
        return load_official(self.root/'train.csv',self.root/'folds.csv',
                             expected_rows=18,expected_prompts=5)

    def test_model_indices_and_real_provider_family(self):
        frame,folds,family,labels=self.load()
        self.assertEqual(len(labels),18)
        self.assertEqual(frame.label.nunique(),18)
        self.assertEqual(family[labels.index('claude-opus-5')],
                         family[labels.index('claude-fable-5')])
        self.assertNotEqual(family[labels.index('glm-5.3')],
                            family[labels.index('claude-fable-5')])
        self.assertEqual(folds['p0'],0)

    def test_reject_wrong_model_same_row(self):
        v=self.tr.copy();v.loc[0,'model']='gpt-5.5'
        v.to_csv(self.root/'train.csv',index=False)
        with self.assertRaises(ValueError):self.load()

    def test_reject_prompt_cross_fold(self):
        v=self.fo.copy();v.loc[0,'fold_id']=2
        v.to_csv(self.root/'folds.csv',index=False)
        with self.assertRaises(ValueError):self.load()

    def test_reject_missing_row(self):
        self.tr.iloc[:-1].to_csv(self.root/'train.csv',index=False)
        with self.assertRaises(ValueError):self.load()

    def test_reject_duplicate_row(self):
        v=self.tr.copy();v.loc[1,'id']=v.loc[0,'id']
        v.to_csv(self.root/'train.csv',index=False)
        with self.assertRaises(ValueError):self.load()

    def test_reject_blank_text(self):
        v=self.tr.copy();v.loc[0,'text']=''
        v.to_csv(self.root/'train.csv',index=False)
        with self.assertRaises(ValueError):self.load()

if __name__=='__main__':
    unittest.main()
