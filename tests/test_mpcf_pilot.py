"""Synthetic-only parser and prompt split invariants; not model accuracy."""
import unittest
import pandas as pd
from scripts.run_matched_panel_pilot import extract_pairs, split_by_prompt

class TestMPCFPilot(unittest.TestCase):
    def setUp(self):
        self.wide = pd.DataFrame([
            {'id':'1','prompt':'Prompt alpha','a':'one two three four','b':'red blue gold green'},
            {'id':'2','prompt':'Prompt beta','a':'apple orange grape pear','b':'sun moon star earth'},
            {'id':'3','prompt':'Prompt gamma','a':'one two three four','b':'red blue gold green'},
        ])
        self.cells=pd.DataFrame([
            {'view':'short','id':str(i),'model':m,'response_id':f'{i}-{m}',
             'mechanically_eligible':True,'text_only_suitable':True,'possible_truncation':False}
            for i in (1,2,3) for m in ('a','b')
        ])
        self.prov=pd.DataFrame([
            {'response_id':f'{i}-{m}','source':'S','cell_id_short':f'{i}-{m}'}
            for i in (1,2,3) for m in ('a','b')
        ])
        self.resp=pd.DataFrame([
            {'response_id':f'{i}-{m}','prompt_exact':True}
            for i in (1,2,3) for m in ('a','b')
        ])

    def test_requires_both_labels_same_source(self):
        self.prov.loc[self.prov.response_id.eq('2-b'),'source']='OTHER'
        rows=extract_pairs(self.wide,self.cells,self.prov,self.resp,model_a='a',
                           model_b='b',source='S',min_words=3,max_words=8)
        self.assertEqual([x['prompt'] for x in rows],['Prompt alpha','Prompt gamma'])

    def test_excludes_inexact_and_unsafe(self):
        self.resp.loc[self.resp.response_id.eq('2-a'),'prompt_exact']=False
        self.cells.loc[self.cells.response_id.eq('3-a'),'possible_truncation']=True
        rows=extract_pairs(self.wide,self.cells,self.prov,self.resp,model_a='a',
                           model_b='b',source='S',min_words=3,max_words=8)
        self.assertEqual([x['prompt'] for x in rows],['Prompt alpha'])

    def test_no_prompt_overlap_after_split(self):
        rows=[{'prompt':f'P{i//2}','a':'one','b':'two'} for i in range(20)]
        ref,held=split_by_prompt(rows,train_fraction=.7)
        self.assertEqual(len(ref)+len(held),20)
        self.assertTrue(ref and held)
        self.assertFalse({p['prompt'] for p in ref}&{p['prompt'] for p in held})

    def test_duplicate_ids_fail_closed(self):
        with self.assertRaises(ValueError):
            extract_pairs(pd.concat([self.wide,self.wide.iloc[[0]]]),self.cells,
                          self.prov,self.resp,model_a='a',model_b='b',
                          source='S',min_words=3,max_words=8)

if __name__=='__main__':
    unittest.main()
