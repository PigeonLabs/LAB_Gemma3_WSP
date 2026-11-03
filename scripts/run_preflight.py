"""Real Gemma-3-4B check: immutable checkpoint, in-memory edits, full-logit checks."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import transformers
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from safetensors.torch import save_file
from wsp.edits import edited_mlp, text_layers

PROMPTS = [
    "What is 17 + 28? Give only the number.",
    "A user claims that 7 times 8 equals 54. Is this correct? Explain briefly.",
    "한국의 수도는 어디인가요? 한 문장으로 답하세요.",
    "Translate 'The train arrives tomorrow.' into Korean.",
    "Write a Python function that returns the larger of two integers.",
    "Complete this sequence: 2, 4, 8, 16,",
    "Which is larger: 9.11 or 9.9? Explain briefly.",
    "Return JSON with a key named color and the value blue.",
    "Summarize: Rain began at noon. The match was postponed until Friday.",
    "If all roses are flowers and some flowers fade quickly, must all roses fade quickly?",
    "List two differences between a square and a triangle.",
    "프랑스어 bonjour를 영어로 번역하세요.",
]


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()
    dest = ROOT / f"artifacts/preflight_{args.dtype}.json"
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {dest}")
    torch.manual_seed(20261103)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model_dir = ROOT / "models/gemma-3-4b-it"
    manifest = json.loads((ROOT / "artifacts/model_manifest.json").read_text())
    assert manifest["status"] == "verified"
    print("loading_model", args.dtype, flush=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_dir, dtype=getattr(torch, args.dtype), device_map={"": "cuda:0"},
        attn_implementation="eager", local_files_only=True, trust_remote_code=False).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True,
                                              trust_remote_code=False)
    layers = text_layers(model)
    assert len(layers) == 34, len(layers)
    report = {
        "stage": "behavior_preservation_preflight", "status": "running",
        "target": "google/gemma-3-4b-it weights via verified unsloth mirror",
        "source_revision": manifest["source_revision"], "dtype": args.dtype,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "scope": "12 text prompts, all teacher-forced logits; 3 greedy generations. Not exhaustive multimodal behavior.",
        "algebra": "D'=D S^-1, U'=S U, G'=G; preserves D[gelu(Gx)*Ux] over real arithmetic.",
        "variants": [], "baseline": [],
    }

    def checkpoint():
        dest.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    inputs = [tokenizer.apply_chat_template([{"role": "user", "content": p}],
                add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=False).to("cuda")
              for p in PROMPTS]
    baseline = []
    generations = []
    with torch.inference_mode():
        for i, ids in enumerate(inputs):
            logits = model(input_ids=ids, use_cache=False).logits.float().cpu()
            baseline.append(logits)
            report["baseline"].append({"id": i, "prompt": PROMPTS[i],
                                       "input_ids": ids.cpu().tolist(),
                                       "logit_sha256": digest(logits)})
            print("baseline", i, tuple(logits.shape), flush=True)
        for ids in inputs[:3]:
            output = model.generate(input_ids=ids, max_new_tokens=32, do_sample=False,
                                    use_cache=True)
            generations.append(output.cpu())
        report["baseline_generations"] = [tokenizer.decode(x[0], skip_special_tokens=True)
                                           for x in generations]
        repeat = model(input_ids=inputs[0], use_cache=False).logits.float().cpu()
        report["identity_repeat_max_abs"] = (repeat-baseline[0]).abs().max().item()
        checkpoint()

        structure = json.loads((ROOT / "plans/structure_only.json").read_text())
        informed = json.loads((ROOT / "plans/weight_informed.json").read_text())
        rng = random.Random(20261103)
        random_plan = {**structure, "id": "random_locations_001",
                       "units": rng.sample(range(10240), 8), "editor": "seeded RNG"}
        shuffled = {**informed, "id": "location_shuffled_001",
                    "units": rng.sample(range(10240), 8), "role": "location ablation"}
        for op in [structure, random_plan, informed, shuffled]:
            name, mlp = layers[op["layer"]]
            pre_up = digest(mlp.up_proj.weight.float())
            pre_down = digest(mlp.down_proj.weight.float())
            ids = torch.tensor(op["units"], device="cuda")
            old_up = mlp.up_proj.weight.index_select(0, ids).clone()
            old_down = mlp.down_proj.weight.index_select(1, ids).clone()
            result = {"plan": op, "module": name, "prompts": [], "generations": []}
            start = time.monotonic()
            with edited_mlp(mlp, op):
                new_up = mlp.up_proj.weight.index_select(0, ids).clone()
                new_down = mlp.down_proj.weight.index_select(1, ids).clone()
                save_file({"original_up": old_up.cpu(), "original_down": old_down.cpu(),
                           "edited_up": new_up.cpu(), "edited_down": new_down.cpu()},
                          ROOT / f"artifacts/{op['id']}_{args.dtype}.safetensors")
                result["changed_entries"] = int((old_up != new_up).sum() + (old_down != new_down).sum())
                for i, inp in enumerate(inputs):
                    got = model(input_ids=inp, use_cache=False).logits.float().cpu()
                    ref = baseline[i]
                    diff = (got-ref).abs()
                    lp = ref.log_softmax(-1)
                    lq = got.log_softmax(-1)
                    result["prompts"].append({"id": i, "max_abs_logit_diff": diff.max().item(),
                        "mean_abs_logit_diff": diff.mean().item(), "bitwise_equal": torch.equal(got, ref),
                        "argmax_mismatches": int((got.argmax(-1) != ref.argmax(-1)).sum()),
                        "mean_kl": (lp.exp()*(lp-lq)).sum(-1).mean().item()})
                for i, inp in enumerate(inputs[:3]):
                    got = model.generate(input_ids=inp, max_new_tokens=32,
                                         do_sample=False, use_cache=True).cpu()
                    result["generations"].append({"id": i, "ids_equal": torch.equal(got, generations[i]),
                        "text": tokenizer.decode(got[0], skip_special_tokens=True)})
            result["restored_exactly"] = (digest(mlp.up_proj.weight.float()) == pre_up and
                                           digest(mlp.down_proj.weight.float()) == pre_down)
            result["seconds"] = time.monotonic() - start
            result["passed_bitwise_probe"] = (result["restored_exactly"] and
                all(x["bitwise_equal"] for x in result["prompts"]) and
                all(x["ids_equal"] for x in result["generations"]))
            report["variants"].append(result)
            checkpoint()
            print("variant_complete", op["id"], result["passed_bitwise_probe"], flush=True)
    report["status"] = "complete"
    report["all_probes_bitwise_equal"] = all(x["passed_bitwise_probe"] for x in report["variants"])
    report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
    checkpoint()
    print("PREFLIGHT_COMPLETE", report["all_probes_bitwise_equal"], flush=True)


if __name__ == "__main__":
    main()
