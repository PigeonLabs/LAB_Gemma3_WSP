"""Local observer, fresh context, no forward/tool access to target."""
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
from .study import ROOT, config, file_hash


def default_observer():
    return json.loads((ROOT/'artifacts/circuit_forced_choice.json').read_text())['observer_path']


class Observer:
    def __init__(self, path=None):
        self.path = path or default_observer()
        from pathlib import Path
        if Path(self.path).name != config()['observer_revision']:
            raise ValueError('Observer revision differs from preregistration')
        torch.set_num_threads(8)
        torch.manual_seed(config()['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        start = time.perf_counter()
        self.model = AutoModelForImageTextToText.from_pretrained(self.path, local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16, device_map={'':'cuda:0'}, attn_implementation='eager').eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter()-start
        self.labels = {x:self.tokenizer.encode(' '+x,add_special_tokens=False) for x in ['A','B']}
        if len(self.labels['A']) != len(self.labels['B']):
            raise ValueError('A/B completion lengths differ')
        print(f'observer_ready load_seconds={self.load_seconds:.3f}',flush=True)

    def ids(self, text):
        return self.tokenizer.apply_chat_template([{'role':'user','content':text}],
            tokenize=True, add_generation_prompt=True, return_tensors='pt',return_dict=False).to('cuda')

    @torch.inference_mode()
    def score(self, text):
        torch.cuda.synchronize(); start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        ids = self.ids(text)
        prefix = torch.tensor([self.tokenizer.encode('ANSWER:',add_special_tokens=False)],device='cuda')
        base = torch.cat([ids,prefix],dim=1)
        suffix = torch.tensor([self.labels['A'],self.labels['B']],device='cuda')
        full = torch.cat([base.expand(2,-1),suffix],dim=1)
        n = suffix.shape[1]
        logits = self.model(input_ids=full,use_cache=False,logits_to_keep=n+1).logits
        lp = logits[:,-n-1:-1].float().log_softmax(-1)
        sums = lp.gather(-1,suffix[:,:,None]).sum((1,2)).tolist()
        scores = dict(zip(['A','B'],sums)); gap = scores['A']-scores['B']
        answer = 'UNKNOWN' if abs(gap)<config()['uncertain_logprob_gap'] else 'A' if gap>0 else 'B'
        torch.cuda.synchronize()
        return {'answer':answer,'completion_logprob':scores,'logprob_gap_A_minus_B':gap,
                'seconds':time.perf_counter()-start,'input_tokens':int(ids.shape[1]),
                'generated_tokens':0,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()}

    @torch.inference_mode()
    def prose(self, text):
        torch.cuda.synchronize(); start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        ids=self.ids(text)
        result=self.model.generate(input_ids=ids,do_sample=False,max_new_tokens=config()['prose_max_new_tokens'],use_cache=True)
        response=self.tokenizer.decode(result[0,ids.shape[1]:],skip_special_tokens=True)
        answer='INVALID'; parsed=None
        try:
            parsed=json.loads(response.strip())
            if isinstance(parsed,dict) and parsed.get('claim') in ['A','B','UNKNOWN'] and isinstance(parsed.get('units'),list) and isinstance(parsed.get('reason'),str):
                if all(type(i)==int and 0<=i<16 for i in parsed['units']): answer=parsed['claim']
        except (ValueError,TypeError): pass
        torch.cuda.synchronize()
        return {'answer':answer,'response':response,'parsed':parsed,
                'seconds':time.perf_counter()-start,'input_tokens':int(ids.shape[1]),
                'generated_tokens':int(result.shape[1]-ids.shape[1]),
                'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()}
