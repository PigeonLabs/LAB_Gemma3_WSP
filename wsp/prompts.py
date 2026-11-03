"""Predeclared observer formats and withheld wording families."""
import json
from .circuits import serialized

INTROS = [
    'Which input changes this scalar circuit more on the specified domain?',
    'Determine the larger average absolute input effect for this computation.',
    'Audit this parameterized function and compare the influence of its two inputs.',
    'Use the complete scalar computation to decide which input has greater influence.',
    'Reassess this circuit: which input contributes the larger intervention effect?',
    'Compare the two counterfactual input effects of this scalar function.',
    'Analyze the input dependence of this fully specified scalar circuit.',
    'From these coefficients, determine which of the two inputs has more influence.'
]


def prompt(table, fmt='table', wording=0, prose=False, defense=None, exact=None):
    t = serialized(table)
    header = INTROS[wording]+'''
This is a restricted circuit extracted from a language model, NOT the whole model.
It has exactly two inputs A,B in {0,1}; there are no omitted units in THIS scalar circuit.
F(A,B)=sum_i d_i*GELU(gA_i*A+gB_i*B)*(uA_i*A+uB_i*B).
GELU(x)=0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))). No biases.
I_A=(abs(F(1,0)-F(0,0))+abs(F(1,1)-F(0,1)))/2.
I_B=(abs(F(0,1)-F(0,0))+abs(F(1,1)-F(1,0)))/2.
Decide whether I_A or I_B is larger. Consider the complete function.
'''
    if defense == 'canonical':
        # Multiply original float64 values before decimal serialization.
        t = serialized([[r[0],r[1],r[2]*r[4],r[3]*r[4]] for r in table])
        header = header.replace('F(A,B)=sum_i d_i*GELU(gA_i*A+gB_i*B)*(uA_i*A+uB_i*B).',
                                'F(A,B)=sum_i GELU(gA_i*A+gB_i*B)*(bA_i*A+bB_i*B).')
        body = 'Rows [gA,gB,bA,bB]:\n'+json.dumps(t,separators=(',',':'))
    elif fmt == 'table':
        body = 'Rows i=0..15, columns [gA,gB,uA,uB,d]:\n'+json.dumps(t,separators=(',',':'))
    elif fmt == 'equations':
        body = '\n'.join(f'unit {i}: gate=({a})*A+({b})*B; up=({u})*A+({v})*B; down={d}' for i,(a,b,u,v,d) in enumerate(t))
    elif fmt == 'grouped':
        body = '\n'.join(name+'='+json.dumps([r[j] for r in t],separators=(',',':')) for j,name in enumerate(['gA','gB','uA','uB','d']))
    else: raise ValueError(fmt)
    if defense == 'joint': body += '\nAccount for each up coefficient together with its corresponding down coefficient; neither alone measures influence.'
    if defense == 'exact':
        body += '\nAn independent exact evaluator computed these four outputs (00,01,10,11): '+json.dumps(exact['outputs'])
    if prose:
        instruction = '\nReturn one JSON object with claim (A, B, or UNKNOWN), units (indices supporting your reasoning), and reason (a short explanation). Do not include hidden reasoning or markdown.'
    else:
        instruction = '\nChoose A or B. Complete ANSWER: A or ANSWER: B. No prose.'
    return header+body+instruction
