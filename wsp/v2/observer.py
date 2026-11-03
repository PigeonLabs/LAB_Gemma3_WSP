"""Gemma 3 only: fresh conversations and an explicit primitive-tool protocol."""
import json
import math
import time
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
from .arithmetic import Calculator
from .state import ROOT, config

def framed(c,frame):
    table=[row.copy() for row in c['table']]
    if frame&1: table=[[r[1],r[0],r[3],r[2],r[4]] for r in table]
    code={'X':'B' if frame&2 else 'A','Y':'A' if frame&2 else 'B'}
    return table,code

def decode(answer,frame):
    if answer not in ['A','B']: return 'INVALID'
    xcode='B' if frame&2 else 'A'
    axis='X' if answer==xcode else 'Y'
    return ('B' if axis=='X' else 'A') if frame&1 else ('A' if axis=='X' else 'B')

def parse_object(text):
    # Only whitespace / a single optional Markdown JSON fence is tolerated.
    text=text.strip()
    if text.startswith('```json') and text.endswith('```'): text=text[7:-3].strip()
    elif text.startswith('```') and text.endswith('```'): text=text[3:-3].strip()
    result=json.loads(text)
    if not isinstance(result,dict): raise ValueError('Expected object')
    return result

def prompt(c,frame,mode):
    table,code=framed(c,frame)
    intro=('Analyze this real gated MLP subcircuit. F(X,Y)=sum_i d[i]*GELU(gX[i]*X+gY[i]*Y)*(uX[i]*X+uY[i]*Y). '
           'GELU(z)=0.5*z*(1+tanh(sqrt(2/pi)*(z+0.044715*z^3))). '
           'IX=(abs(F(1,0)-F(0,0))+abs(F(1,1)-F(0,1)))/2. '
           'IY=(abs(F(0,1)-F(0,0))+abs(F(1,1)-F(1,0)))/2. '
           f'Which influence is larger? Answer code {code["X"]} means X, code {code["Y"]} means Y. '
           'Use the provided coefficients, including signs and cancellation.\n')
    values={k:[float(f'{r[i]:.12g}') for r in table] for i,k in enumerate(['gX','gY','uX','uY','d'])}
    if mode=='D3':
        y=c['oracle']['outputs'].copy()
        if frame&1: y[1],y[2]=y[2],y[1]
        intro+='Exact independently computed F00,F01,F10,F11='+json.dumps(y)+'\n'
    else: intro+='Coefficient handles and FP64 values: '+json.dumps(values,separators=(',',':'))+'\n'
    if mode=='D0': return intro+'Respond with only A or B.'
    if mode in ['D1','D3']:
        return intro+('Calculate internally. Respond with one JSON object: '
                      '{"answer":"A or B","IX":number,"IY":number,"reason":"calculation and explanation, at most 80 words"}. '
                      'No tools are available. No Markdown or text outside JSON.')
    return intro+('You can operate an FP64 calculator. You choose all operations; it will not solve the circuit automatically. '
        'Every turn must be one JSON object, without Markdown. Either request '
        '{"operations":[{"id":"new_result_name","op":"operation","args":[arguments]}]} or finish with '
        '{"answer":"A or B","IX":"scalar_result_handle","IY":"scalar_result_handle","reason":"at most 80 words"}. '
        'Operations: add/sub/mul/div take two scalar-or-vector arguments (elementwise, scalar broadcast); '
        'abs/gelu/sum take one. sum reduces a vector to a scalar. take takes [vector_handle,[zero_based_indices]]. '
        'Arguments must be string handles gX,gY,uX,uY,d or earlier result ids, or numeric constants -1,0,0.5,1,2. '
        'Never copy coefficient numbers into arguments. New result ids must be unique identifiers. '
        'Up to 4 operations per request, up to 32 operations total; later operations can reference earlier ids in the same request. '
        'For example {"operations":[{"id":"example","op":"add","args":[1,2]}]} returns 3. '
        'This example is only API syntax; do not call it unnecessarily. All generated replies together must fit 2048 tokens. '
        'The final IX and IY must reference scalar results from your calculations. Do not guess.')

class Observer:
    def __init__(self):
        torch.set_num_threads(8); torch.manual_seed(config()['seed'])
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        start=time.perf_counter(); path=str(ROOT/config()['target'])
        self.model=AutoModelForImageTextToText.from_pretrained(path,local_files_only=True,trust_remote_code=False,
             dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager').eval()
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        self.model.generation_config.pad_token_id=self.tokenizer.pad_token_id
        self.load_seconds=time.perf_counter()-start
        print('Gemma 3 observer ready; seconds',self.load_seconds,flush=True)

    def ids(self,messages):
        return self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
              return_tensors='pt',return_dict=False).to('cuda')

    @torch.inference_mode()
    def score(self,text):
        ids=self.ids([{'role':'user','content':text}])
        prefix=torch.tensor([self.tokenizer.encode('ANSWER:',add_special_tokens=False)],device='cuda')
        base=torch.cat([ids,prefix],dim=1)
        suffixes=[self.tokenizer.encode(' '+s,add_special_tokens=False) for s in ['A','B']]
        if len(suffixes[0])!=len(suffixes[1]): raise ValueError('Unequal score suffix length')
        suffix=torch.tensor(suffixes,device='cuda'); n=suffix.shape[1]
        logits=self.model(input_ids=torch.cat([base.expand(2,-1),suffix],dim=1),use_cache=False,logits_to_keep=n+1).logits
        scores=logits[:,-n-1:-1].float().log_softmax(-1).gather(-1,suffix[:,:,None]).sum((1,2)).tolist()
        gap=scores[0]-scores[1]
        return {'answer':'UNKNOWN' if abs(gap)<.05 else 'A' if gap>0 else 'B','scores':scores,'gap':gap,'input_tokens':int(ids.shape[1])}

    @torch.inference_mode()
    def generate(self,messages,budget):
        ids=self.ids(messages)
        output=self.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,
                max_new_tokens=budget,use_cache=True)
        response=self.tokenizer.decode(output[0,ids.shape[1]:],skip_special_tokens=True)
        return response,int(output.shape[1]-ids.shape[1]),int(ids.shape[1])

    def episode(self,c,frame,mode='D2'):
        start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        text=prompt(c,frame,mode); table,_=framed(c,frame)
        table=[[float(f'{v:.12g}') for v in r] for r in table]
        calc=Calculator(table); messages=[{'role':'user','content':text}]
        used=0; inputs=0; errors=[]; attempts=0; final=None; answer='INVALID'; numeric=None; turns=[]
        if mode=='D0':
            final=self.score(text); answer=final['answer']; inputs=final['input_tokens']
        else:
            while used<config()['observer_max_generated_tokens']:
                response,tokens,ntokens=self.generate(messages,config()['observer_max_generated_tokens']-used)
                used+=tokens; inputs+=ntokens; turns.append({'response':response,'generated_tokens':tokens,'input_tokens':ntokens})
                messages.append({'role':'assistant','content':response})
                try:
                    obj=parse_object(response)
                    if 'operations' in obj:
                        if mode!='D2' or set(obj)!={'operations'}: raise ValueError('Tools unavailable or request schema error')
                        feedback={'results':calc.apply(obj['operations']),'operations_remaining':calc.limit-calc.attempts}
                        messages.append({'role':'user','content':json.dumps(feedback,separators=(',',':'))})
                    else:
                        if set(obj)!={'answer','IX','IY','reason'} or obj['answer'] not in ['A','B'] or not isinstance(obj['reason'],str) or len(obj['reason'].split())>80:
                            raise ValueError('Final JSON schema or explanation length error')
                        if mode=='D2': numeric=[calc.scalar(obj['IX']),calc.scalar(obj['IY'])]
                        else:
                            if any(type(obj[k]) not in [int,float] for k in ['IX','IY']): raise ValueError('Numeric influences required')
                            numeric=[float(obj['IX']),float(obj['IY'])]
                        if not all(math.isfinite(v) for v in numeric): raise ValueError('Nonfinite final')
                        final=obj; answer=obj['answer']; break
                except (ValueError,TypeError,KeyError,IndexError,FloatingPointError) as exc:
                    errors.append(str(exc)); attempts+=1
                    if attempts>config()['observer_format_retries']: break
                    messages.append({'role':'user','content':'Protocol error: '+str(exc)+'. Correct the JSON format or reference; no answer feedback is provided. This is the sole retry.'})
                if tokens==0: break
        decoded=decode(answer,frame); truth=c['oracle']['truth']; correct=decoded==truth
        expected=[c['oracle']['influence_A'],c['oracle']['influence_B']]
        if frame&1: expected.reverse()
        numeric_ok=numeric is not None and all(math.isclose(a,b,abs_tol=config()['numeric_atol'],rel_tol=config()['numeric_rtol']) for a,b in zip(numeric,expected))
        return {'circuit_id':c['id'],'layer':c['layer'],'n':len(c['table']),'frame':frame,'mode':mode,
            'prompt':text,'answer':answer,'decoded_answer':decoded,'truth':truth,'correct':correct,
            'numeric_correct':numeric_ok,'joint_correct':correct and numeric_ok,'numeric':numeric,
            'final':final,'turns':turns,'trace':calc.trace,'operations_attempted':calc.attempts,
            'errors':errors,'generated_tokens':used,'input_tokens':inputs,'seconds':time.perf_counter()-start,
            'budget_exhausted':final is None and used>=config()['observer_max_generated_tokens'],
            'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()}
