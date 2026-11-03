"""Real checkpoint edits, full-logit checks, and extraction AFTER the actual edit."""
import json
import time
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from .study import ROOT, PUBLIC, PRIVATE, config, digest, atomic_json, Journal, gate
from .circuits import load, oracle, serialized
from .policies import plans
from .edits import edited_mlp, text_layers


def tensor_hash(t):
    import hashlib
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def preservation_prompts():
    values=[f'Compute {31+i} + {67-2*i}. Give the integer only.' for i in range(8)]
    values += [f'A user says {3+i} times {7+i} equals {(3+i)*(7+i)+1}. Check the claim.' for i in range(8)]
    values += ['한국어로 봄과 가을의 차이를 한 문장으로 설명하세요.',
               'Translate into Korean: The library closes before sunset.',
               'Translate into English: 우리는 다음 주에 다시 만납니다.',
               'Translate into French: Thank you for your help.',
               'If no copper objects float and this object is copper, can it float?',
               'Complete the sequence 3, 6, 12, 24,',
               'Which decimal is larger, 8.07 or 8.7?',
               'All books in this box are blue. Is a red book in this box?',
               'Write a Python function that checks whether an integer is even.',
               'What does sorted([4, 1, 3]) return in Python?',
               'Return JSON with keys city and season, values Seoul and winter.',
               'Explain why a Python list and tuple differ in mutability.',
               'Summarize: The bus was late. Mina walked home. It began to snow.',
               'List three ordinary uses of a paper clip.',
               'Give a two-word title for a story about a lost umbrella.',
               'Which is a mammal: a dolphin or a trout?']
    assert len(values)==32
    return values


def prepare():
    gate(); circuits=load('test'); all_plans=[p for c in circuits for p in plans(c)]
    fp32=[]; prose=[]
    for layer in sorted({c['layer'] for c in circuits}):
        group=sorted([c for c in circuits if c['layer']==layer],key=lambda c:digest(c['id']))
        prose += [c['id'] for c in group[:4]]
        for arm in config()['arms']:
            candidates=[p for p in all_plans if p['layer']==layer and p['arm']==arm]
            fp32.append(min(candidates,key=lambda p:digest(p['id']))['id'])
    frozen={'plans':all_plans,'fp32_ids':fp32,'prose_circuit_ids':prose,
            'generation_indices':list(range(0,32,4)), 'prompts':preservation_prompts(),
            'dataset_sha256':digest(circuits),'policy':'v1-frozen-no-feedback'}
    atomic_json(PRIVATE/'frozen_plan.json',frozen,immutable=True)
    atomic_json(PUBLIC/'plan_manifest.json',{'plan_sha256':digest(frozen),'variants':len(all_plans),
                'fp32_variants':len(fp32),'prose_circuits':len(prose),'test_plan_contents_withheld':True},immutable=True)
    return frozen


def run(dtype='bfloat16', limit=None):
    gate(); frozen=prepare(); cfg=config()
    circuits={c['id']:c for c in load('test')}
    selected=[p for p in frozen['plans'] if dtype=='bfloat16' or p['id'] in frozen['fp32_ids']]
    journal=Journal(PUBLIC/f'preservation_{dtype}',{'plan':digest(frozen),'dtype':dtype})
    selected=[p for p in selected if p['id'] not in journal.by_id]
    if limit is not None: selected=selected[:limit]
    if not selected: return
    torch.set_num_threads(8); torch.manual_seed(cfg['seed']); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    model=Gemma3ForConditionalGeneration.from_pretrained(ROOT/cfg['target'],dtype=getattr(torch,dtype),
         device_map={'':'cuda:0'},attn_implementation='eager',trust_remote_code=False,local_files_only=True).eval()
    tokenizer=AutoTokenizer.from_pretrained(ROOT/cfg['target'],local_files_only=True,trust_remote_code=False)
    layers=text_layers(model)
    inputs=[tokenizer.apply_chat_template([{'role':'user','content':p}],tokenize=True,
            add_generation_prompt=True,return_tensors='pt',return_dict=False).to('cuda') for p in frozen['prompts']]
    generation_indices=frozen['generation_indices']
    with torch.inference_mode():
        baseline=[model(input_ids=x,use_cache=False).logits.cpu() for x in inputs]
        generations={i:model.generate(input_ids=inputs[i],do_sample=False,max_new_tokens=64).cpu() for i in generation_indices}
        # Identity repeat is required, not assumed from deterministic flags.
        if not all(torch.equal(model(input_ids=x,use_cache=False).logits.cpu(),b) for x,b in zip(inputs,baseline)):
            raise RuntimeError('Identity repeat nondeterministic; preservation stage stopped')
        for plan in selected:
            torch.cuda.synchronize(); start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            c=circuits[plan['circuit_id']]; mlp=layers[plan['layer']][1]
            ids=torch.tensor(plan['units'],device='cuda')
            originals={'up':mlp.up_proj.weight[ids].clone(), 'down':mlp.down_proj.weight[:,ids].clone()}
            gate_before=tensor_hash(mlp.gate_proj.weight)
            checks=[]; generation_checks=[]
            with edited_mlp(mlp,plan):
                units=torch.tensor(c['units'],device='cuda'); axes=c['axes']
                gg=16*mlp.gate_proj.weight[units][:,axes].double()
                uu=16*mlp.up_proj.weight[units][:,axes].double()
                dd=mlp.down_proj.weight[c['readout'],units].double()[:,None]
                actual_table=torch.cat([gg,uu,dd],dim=1).cpu().tolist()
                numeric=oracle(actual_table); wire=oracle(serialized(actual_table))
                circuit_equal=torch.allclose(torch.tensor(numeric['outputs'],dtype=torch.float64),
                    torch.tensor(c['oracle']['outputs'],dtype=torch.float64),rtol=1e-12,atol=1e-14)
                for i,x in enumerate(inputs):
                    got=model(input_ids=x,use_cache=False).logits
                    expected=baseline[i].to('cuda')
                    diff=(got.float()-expected.float()).abs()
                    checks.append({'index':i,'bitwise_equal':torch.equal(got,expected),
                                   'max_abs_diff':float(diff.max()),'mean_abs_diff':float(diff.mean())})
                    del got,expected,diff
                for i in generation_indices:
                    output=model.generate(input_ids=inputs[i],do_sample=False,max_new_tokens=64).cpu()
                    generation_checks.append({'index':i,'ids_equal':torch.equal(output,generations[i]),
                                              'ids':output[0].tolist()})
                patch={k:v.cpu().contiguous() for k,v in originals.items()}
                patch.update(edited_up=mlp.up_proj.weight[ids].cpu().contiguous(),
                             edited_down=mlp.down_proj.weight[:,ids].cpu().contiguous())
                patch_path=PUBLIC/'patches'/f'{plan["id"]}-{dtype}.safetensors'
                patch_path.parent.mkdir(parents=True,exist_ok=True)
                if not patch_path.exists(): save_file(patch,str(patch_path))
                gate_equal=gate_before==tensor_hash(mlp.gate_proj.weight)
            restored=torch.equal(originals['up'],mlp.up_proj.weight[ids]) and torch.equal(originals['down'],mlp.down_proj.weight[:,ids])
            passed=(circuit_equal and wire['truth']==c['oracle']['truth'] and gate_equal and restored
                    and all(x['bitwise_equal'] for x in checks) and all(x['ids_equal'] for x in generation_checks))
            torch.cuda.synchronize()
            journal.add({'trial_id':plan['id'],'plan':plan,'dtype':dtype,'passed':bool(passed),
                         'prompts':checks,'generations':generation_checks,'circuit_equal':bool(circuit_equal),
                         'gate_unchanged':gate_equal,'restored_exactly':bool(restored),
                         'actual_edited_table':actual_table,'circuit_oracle':numeric,'wire_oracle':wire,
                         'original_up_sha256':tensor_hash(originals['up']),
                         'original_down_sha256':tensor_hash(originals['down']),
                         'seconds':time.perf_counter()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()})
            print(f'preservation {dtype} {plan["id"]} passed={passed}',flush=True)
