import copy, unittest
from dream_rsi.new_task_suite import SPECS, TRANSFER, build_suite, oracle
from dream_rsi.external_judge import canonical
from dream_rsi.coding_tasks import extract_code

class NewTaskSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.suite=build_suite(count=24)
    def test_sizes(self):
        self.assertEqual([len(self.suite[k]) for k in ('train','dev','confirmation','transfer')],[10,10,10,6])
        self.assertTrue(all(len(r['cases'])==24 for k in ('train','dev','confirmation','transfer') for r in self.suite[k]))
    def test_reproducible(self): self.assertEqual(self.suite,build_suite(count=24))
    def test_input_disjointness(self):
        for i in range(10):
            sets=[{canonical(a) for a,b in self.suite[k][i]['cases']} for k in ('train','dev','confirmation')]
            self.assertTrue(all(not sets[i]&sets[j] for i in range(3) for j in range(i)))
    def test_transfer_family_exclusion(self): self.assertFalse(set(SPECS)&set(TRANSFER))
    def test_references_vs_oracles(self):
        # Only trusted author-controlled references are executed locally.
        for name,(description,source) in SPECS.items():
            ns={};exec(source,ns)
            for split in ('train','dev','confirmation'):
                row=next(x for x in self.suite[split] if x['family']==name)
                for arg,expected in row['cases']:
                    with self.subTest(name=name,split=split,arg=arg):
                        actual=ns['solve'](copy.deepcopy(arg))
                        self.assertEqual(type(actual),type(expected));self.assertEqual(actual,expected)
    def test_half_even(self):
        for arg,expected in [(['2.5',0],'2'),(['3.5',0],'4'),(['-0.005',2],'0.00'),(['-1.225',2],'-1.22')]:
            self.assertEqual(oracle('round_even',arg),expected)
    def test_pointer_escaping_and_array(self):
        for arg,expected in [([{'a/b':{'~':[7]}},'/a~1b/~0/0'],{'found':True,'value':7}),([['a'],'/00'],{'found':False}),([None,''],{'found':True,'value':None})]:
            self.assertEqual(oracle('pointer',arg),expected)
    def test_nonfile_standard_library_imports_allowed(self):
        for name in ('csv','decimal','ipaddress','fractions','json','re','shlex','posixpath','io'):
            self.assertIn('solve',extract_code(f'import {name}\ndef solve(data): return data'))
    def test_dangerous_imports_still_rejected(self):
        for name in ('os','sys','subprocess','socket','ctypes','pathlib'):
            with self.assertRaises(ValueError): extract_code(f'import {name}\ndef solve(data): return data')
    def test_edge_contracts(self):
        rows=[('csv_record','',['']),('ordering',[2,[[0,1],[1,0]]],[]),('paths','../../a/../b','../../b'),('ipv4',['1.2.3.4',[['0.0.0.0/0','default'],['1.2.3.4/32','host']]],'host'),('bucket',[0,2,[[1,1],[2,0]]],[False,True]),('shell_words',"a'' 'b c' d\\ e",['a','b c','d e'])]
        for name,arg,expected in rows: self.assertEqual(oracle(name,arg),expected)

if __name__=='__main__':unittest.main()

class ComparisonTests(unittest.TestCase):
    def row(self,key,passed,total=4):return {'task_id':key,'result':{'passed':passed,'total':total}}
    def test_partial_regression_not_hidden_by_aggregate(self):
        from dream_rsi.new_task_runtime import preserved
        self.assertFalse(preserved([self.row('a',2),self.row('b',0)],[self.row('a',1),self.row('b',4)]))
    def test_pairing_checks_tasks_and_counts(self):
        from dream_rsi.new_task_runtime import paired
        with self.assertRaises(ValueError):paired([self.row('a',1)],[self.row('b',1)])
        with self.assertRaises(ValueError):paired([self.row('a',1)],[self.row('a',1,5)])
    def test_complete_wins_and_losses(self):
        from dream_rsi.new_task_runtime import paired
        x=paired([self.row('a',4),self.row('b',2)],[self.row('a',3),self.row('b',4)])
        self.assertEqual((x['newly_complete'],x['lost_complete']),(1,1));self.assertEqual(x['case_gain'],.125)
    def test_missing_regression_cannot_pass(self):
        from dream_rsi.new_task_runtime import preserved
        self.assertFalse(preserved([self.row('a',4)],[]))
