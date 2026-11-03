"""Static extraction of real, nonoverlapping Gemma MLP circuits."""
import json
import random
import math
import torch
from safetensors import safe_open
from .study import ROOT, PUBLIC, PRIVATE, config, digest, atomic_json, require_freeze


def tensor(name):
    folder = ROOT/config()['target']
    index = json.loads((folder/'model.safetensors.index.json').read_text())['weight_map']
    keys = [k for k in index if k.endswith(name)]
    if len(keys) != 1: raise ValueError(f'Ambiguous tensor: {name}')
    with safe_open(folder/index[keys[0]], framework='pt', device='cpu') as handle:
        return handle.get_tensor(keys[0])


def oracle(table):
    t = torch.tensor(table, dtype=torch.float64)
    x = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]], dtype=torch.float64)
    y = (torch.nn.functional.gelu(x@t[:,:2].T, approximate='tanh')*(x@t[:,2:4].T))@t[:,4]
    ia = float(((y[2]-y[0]).abs()+(y[3]-y[1]).abs())/2)
    ib = float(((y[1]-y[0]).abs()+(y[3]-y[2]).abs())/2)
    total = ia+ib
    return {'outputs': y.tolist(), 'influence_A': ia, 'influence_B': ib,
            'relative_margin': abs(ia-ib)/max(total, 1e-300),
            'truth': 'A' if ia > ib else 'B' if ib > ia else 'TIE'}


def serialized(table):
    return [[float(f'{v:.12g}') for v in row] for row in table]


def modified_table(circuit, selected):
    table = [row.copy() for row in circuit['table']]
    for i in selected:
        table[i][2] *= 8.; table[i][3] *= 8.; table[i][4] /= 8.
    return table


def extract():
    require_freeze()
    cfg = config()
    torch.set_num_threads(8)
    all_circuits = []
    counts = {}
    for layer in range(cfg['layer_count']):
        split = ('calibration' if layer in cfg['calibration_layers'] else
                 'development' if layer in cfg['development_layers'] else 'test')
        dest = PRIVATE/'circuits'/f'layer-{layer:02d}.json'
        if dest.exists():
            circuits = json.loads(dest.read_text())
            all_circuits.extend(circuits); counts[str(layer)] = len(circuits)
            continue
        g = tensor(f'layers.{layer}.mlp.gate_proj.weight').float()
        u = tensor(f'layers.{layer}.mlp.up_proj.weight').float()
        d = tensor(f'layers.{layer}.mlp.down_proj.weight').float()
        rng = random.Random(cfg['seed']+layer)
        indices = list(range(g.shape[0])); rng.shuffle(indices)
        circuits = []
        for attempt in range(min(cfg['candidate_groups_per_layer'], len(indices)//cfg['units_per_circuit'])):
            units = indices[attempt*16:(attempt+1)*16]
            gg, uu, dd = g[units].double(), u[units].double(), d[:,units].double()
            axes = (gg.square()+uu.square()).sum(0).topk(2).indices.tolist()
            # Exactly four swaps per layer, independent of truth or observer output.
            swapped = len(circuits) % 2 == 1
            if swapped: axes.reverse()
            readout = int(dd.square().sum(1).argmax())
            table = torch.cat([16*gg[:,axes], 16*uu[:,axes], dd[readout,:,None]], dim=1).tolist()
            truth = oracle(table)
            if truth['relative_margin'] < cfg['minimum_relative_margin'] or truth['influence_A']+truth['influence_B'] < cfg['minimum_total_influence']:
                continue
            wire = oracle(serialized(table))
            if wire['truth'] != truth['truth']: continue
            circuits.append({'id': f'L{layer:02d}C{len(circuits):02d}', 'layer': layer,
                             'split': split, 'units': units, 'axes': axes, 'readout': readout,
                             'axis_swapped': swapped, 'attempt': attempt, 'table': table,
                             'gate_norm': gg.norm(dim=1).tolist(), 'up_norm': uu.norm(dim=1).tolist(),
                             'down_norm': dd.norm(dim=0).tolist(), 'oracle': truth,
                             'serialized_oracle': wire})
            if len(circuits) == cfg['circuits_per_layer']: break
        atomic_json(dest, circuits, immutable=True)
        if split != 'test': atomic_json(PUBLIC/'circuits'/dest.name, circuits, immutable=True)
        all_circuits.extend(circuits); counts[str(layer)] = len(circuits)
        print(f'layer={layer} split={split} count={len(circuits)}', flush=True)
        del g, u, d
    summary = {'status': 'complete' if all(n == 8 for n in counts.values()) else 'insufficient_sample',
               'counts': counts, 'split_counts': {s: sum(c['split']==s for c in all_circuits) for s in ['calibration','development','test']},
               'dataset_sha256': digest(all_circuits),
               'test_contents_private_until_policy_freeze_and_evaluation': True}
    atomic_json(PUBLIC/'dataset_summary.json', summary, immutable=True)
    if summary['status'] != 'complete': raise RuntimeError('Insufficient circuits; thresholds remain fixed')


def load(split=None):
    values = [c for p in sorted((PRIVATE/'circuits').glob('layer-*.json')) for c in json.loads(p.read_text())]
    return [c for c in values if split is None or c['split'] == split]
