"""Independent BF16 verification for subset/semantic edits, with real extraction."""
from contextlib import nullcontext
import gc
import time
import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from .study import ROOT, config
from .edits import edited_mlp, text_layers
from .circuits import oracle, serialized
from .preservation import preservation_prompts, tensor_hash


class TargetChecks:
    def __init__(self):
        cfg=config(); torch.manual_seed(cfg['seed']); torch.set_num_threads(8)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        self.model=Gemma3ForConditionalGeneration.from_pretrained(ROOT/cfg['target'],dtype=torch.bfloat16,
            device_map={'':'cuda:0'},attn_implementation='eager',local_files_only=True,trust_remote_code=False).eval()
        self.tokenizer=AutoTokenizer.from_pretrained(ROOT/cfg['target'],local_files_only=True,trust_remote_code=False)
        self.layers=text_layers(self.model)
        self.inputs=[self.ids(p) for p in preservation_prompts()]
        with torch.inference_mode():
            self.baseline=[self.model(input_ids=x,use_cache=False).logits.cpu() for x in self.inputs]
            self.generations={i:self.model.generate(input_ids=self.inputs[i],do_sample=False,max_new_tokens=64).cpu() for i in range(0,32,4)}
            if not all(torch.equal(self.model(input_ids=x,use_cache=False).logits.cpu(),b) for x,b in zip(self.inputs,self.baseline)):
                raise RuntimeError('Nondeterministic identity in additional verification')

    def ids(self,text):
        return self.tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,
            return_tensors='pt',return_dict=False).to('cuda')

    @torch.inference_mode()
    def check(self,circuit,positions):
        start=time.perf_counter(); m=self.layers[circuit['layer']][1]
        units=[circuit['units'][i] for i in positions]
        before=tensor_hash(m.up_proj.weight),tensor_hash(m.down_proj.weight),tensor_hash(m.gate_proj.weight)
        extra=self.ids(circuit['semantic_prompt']) if 'semantic_prompt' in circuit else None
        extra_baseline=self.model(input_ids=extra,use_cache=False).logits.cpu() if extra is not None else None
        context=edited_mlp(m,{'units':units,'factors':[8.]*len(units)}) if units else nullcontext()
        with context:
            idx=torch.tensor(circuit['units'],device='cuda')
            if 'projection' in circuit:
                basis=torch.tensor(circuit['projection']['input_basis'],device='cuda',dtype=torch.float64)
                readout=torch.tensor(circuit['projection']['readout'],device='cuda',dtype=torch.float64)
                gg=16*m.gate_proj.weight[idx].double()@basis.T
                uu=16*m.up_proj.weight[idx].double()@basis.T
                dd=(readout@m.down_proj.weight[:,idx].double())[:,None]
            else:
                gg=16*m.gate_proj.weight[idx][:,circuit['axes']].double()
                uu=16*m.up_proj.weight[idx][:,circuit['axes']].double()
                dd=m.down_proj.weight[circuit['readout'],idx].double()[:,None]
            table=torch.cat([gg,uu,dd],dim=1).cpu().tolist()
            checks=[]
            for x,b in zip(self.inputs,self.baseline):
                actual=self.model(input_ids=x,use_cache=False).logits.cpu()
                checks.append({'equal':torch.equal(actual,b),'max_abs_diff':float((actual.float()-b.float()).abs().max())})
            generated=[torch.equal(self.model.generate(input_ids=self.inputs[i],do_sample=False,max_new_tokens=64).cpu(),b) for i,b in self.generations.items()]
            extra_equal=True if extra is None else torch.equal(self.model(input_ids=extra,use_cache=False).logits.cpu(),extra_baseline)
        after=tensor_hash(m.up_proj.weight),tensor_hash(m.down_proj.weight),tensor_hash(m.gate_proj.weight)
        original=oracle(circuit['table']); edited=oracle(table); wire=oracle(serialized(table))
        same=torch.allclose(torch.tensor(original['outputs'],dtype=torch.float64),torch.tensor(edited['outputs'],dtype=torch.float64),rtol=1e-12,atol=1e-14)
        passed=bool(all(x['equal'] for x in checks) and all(generated) and extra_equal and before==after and same and wire['truth']==original['truth'])
        torch.cuda.synchronize()
        return {'passed':passed,'positions':positions,'table':table,'checks':checks,'generations_equal':generated,
                'semantic_prompt_logits_equal':extra_equal,'restored':before==after,'circuit_equal':bool(same),
                'seconds':time.perf_counter()-start}

    def close(self):
        del self.model,self.inputs,self.baseline,self.generations
        gc.collect(); torch.cuda.empty_cache()
