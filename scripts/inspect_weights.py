"""Read actual model weights without running the target network."""
import hashlib
import json
import sys
from pathlib import Path
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/gemma-3-4b-it"


def main():
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    rows = []
    with torch.inference_mode():
        for layer in [5, 11, 17, 23, 29, 33]:
            matrices = {}
            for stem in ["gate_proj", "up_proj", "down_proj"]:
                names = [k for k in index if k.endswith(f"layers.{layer}.mlp.{stem}.weight")]
                assert len(names) == 1, names
                name = names[0]
                with safe_open(MODEL / index[name], framework="pt", device="cpu") as f:
                    matrices[stem] = f.get_tensor(name).float().to("cuda")
            g = matrices["gate_proj"].norm(dim=1)
            u = matrices["up_proj"].norm(dim=1)
            d = matrices["down_proj"].norm(dim=0)
            # This statistic is invariant to reciprocal up/down scaling. It is
            # a candidate-ranking heuristic, NOT an activation or causal effect.
            score = g * u * d
            ids = score.topk(16).indices.tolist()
            rows.append({"layer": layer, "intermediate_size": len(score),
                         "gate_norm_mean": g.mean().item(), "up_norm_mean": u.mean().item(),
                         "down_norm_mean": d.mean().item(),
                         "candidates": [{"unit": i, "gate_norm": g[i].item(),
                                         "up_norm": u[i].item(), "down_norm": d[i].item(),
                                         "invariant_norm_product": score[i].item()} for i in ids]})
            print("inspected_layer", layer, flush=True)
            del matrices
    output = {"method": "static gated-MLP weight norms; no target forward pass",
              "limitations": "Norm products do not establish semantic specialization or causal importance.",
              "layers": rows}
    dest = ROOT / "artifacts/weight_inspection.json"
    dest.write_text(json.dumps(output, indent=2))
    print("saved", str(dest), hashlib.sha256(dest.read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
