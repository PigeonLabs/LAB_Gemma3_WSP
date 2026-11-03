import unittest
import math
import numpy as np
from wsp.v2.arithmetic import Calculator
from wsp.v2.observer import framed,decode,parse_object
from wsp.v2.calibration import summarize
from wsp.v2.data import allowed_subsets
from wsp.circuits import oracle

class V2Tests(unittest.TestCase):
    def setUp(self):
        self.table=[[.3,-.2,.8,.1,.5],[.2,.4,.1,.7,-.6],[.4,.1,.5,.4,.2],[.2,.5,-.5,.1,.8]]

    def test_complete_oracle_fits_budget_and_replays(self):
        calc=Calculator(self.table)
        def call(name,op,*args): calc.apply([{'id':name,'op':op,'args':list(args)}])
        for axis in ['X','Y']:
            call('g'+axis+'a','gelu','g'+axis)
            call('h'+axis,'mul','g'+axis+'a','u'+axis)
            call('w'+axis,'mul','h'+axis,'d')
            call('f'+axis,'sum','w'+axis)
        call('gs','add','gX','gY');call('us','add','uX','uY')
        call('ga','gelu','gs');call('hs','mul','ga','us');call('ws','mul','hs','d');call('fs','sum','ws')
        for axis,other in [('X','Y'),('Y','X')]:
            call('delta'+axis,'sub','fs','f'+other);call('a'+axis,'abs','delta'+axis)
            call('b'+axis,'abs','f'+axis);call('total'+axis,'add','a'+axis,'b'+axis)
            call('I'+axis,'div','total'+axis,2)
        expected=oracle(self.table)
        self.assertLessEqual(calc.attempts,32)
        for ax,old in [('X','A'),('Y','B')]: self.assertAlmostEqual(calc.scalar('I'+ax),expected['influence_'+old],places=14)
        edited=[r.copy() for r in self.table]; edited[0][2]*=8;edited[0][3]*=8;edited[0][4]/=8
        other=Calculator(edited);other.replay(calc.trace)
        self.assertNotEqual(calc.values['hX'][0],other.values['hX'][0])
        self.assertAlmostEqual(calc.scalar('IX'),other.scalar('IX'),places=14)

    def test_tools_reject_unsafe_and_budget(self):
        calc=Calculator(self.table,limit=1)
        calc.apply([{'id':'v','op':'sum','args':['d']}])
        with self.assertRaises(ValueError): calc.apply([{'id':'x','op':'sum','args':['d']}])
        for call in [{'id':'z','op':'python','args':['print(1)']},{'id':'z','op':'mul','args':['uX',8]},
                     {'id':'z','op':'take','args':['uX',[-1]]},{'id':'d','op':'sum','args':['uX']}]:
            with self.assertRaises(ValueError): Calculator(self.table).apply([call])

    def test_frames(self):
        for f in range(4):
            table,code=framed({'table':self.table},f)
            truth=oracle(table)['truth']; displayed='X' if truth=='A' else 'Y'
            self.assertEqual(decode(code[displayed],f),oracle(self.table)['truth'])

    def test_missing_invalid_cannot_pass(self):
        cs=[{'id':'a'}]
        rows=[{'circuit_id':'a','frame':f,'correct':False,'joint_correct':False,'decoded_answer':'INVALID',
               'seconds':1,'generated_tokens':1} for f in range(4)]
        self.assertEqual(summarize(rows,cs)['consistency'],0)
        self.assertEqual(summarize([],cs)['status'],'failed')

    def test_budget_matches_actual_full_tensor(self):
        c={'units':list(range(4)),'up_norm':[1,1.01,1.02,1.03],'down_norm':[2]*4}
        self.assertGreaterEqual(len(allowed_subsets(c)),2)

    def test_parse_never_evaluates_code(self):
        with self.assertRaises(ValueError): parse_object('__import__("os").getcwd()')
        self.assertEqual(parse_object('```json\n{"answer":"A"}\n```')['answer'],'A')

if __name__=='__main__': unittest.main()
