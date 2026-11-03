"""Frozen no-feedback selection, with capability-separated policy inputs."""
import itertools
import math
import random
import statistics
from .study import config, digest


def allowed_combinations(up_norm, down_norm, tolerance=.05):
    energies = [49*u*u + (7/8)**2*d*d for u,d in zip(up_norm,down_norm)]
    combos = list(itertools.combinations(range(len(energies)),4))
    norms = [math.sqrt(sum(energies[i] for i in ids)) for ids in combos]
    reference = statistics.median(norms)
    allowed = [(ids,norm) for ids,norm in zip(combos,norms) if abs(norm/reference-1) <= tolerance]
    if not allowed: raise ValueError('No feasible equal-budget edit')
    return allowed, reference


def raw_scores(table, target):
    t = 2 if target == 'A' else 3
    o = 3 if target == 'A' else 2
    return [(abs(row[t])-abs(row[o]))*math.hypot(row[0],row[1])*abs(row[4]) for row in table]


def stats_scores(norms):
    return [g*u*d for g,u,d in zip(norms['gate'],norms['up'],norms['down'])]


def choose(allowed, scores, seed, top=8):
    rng = random.Random(seed)
    if scores is None: return rng.choice(allowed)
    ranked = sorted(allowed, key=lambda x: (-sum(scores[i] for i in x[0]), x[0]))
    return rng.choice(ranked[:top])


def views(circuit, seed):
    table = circuit['table']
    allowed, reference = allowed_combinations(circuit['up_norm'],circuit['down_norm'])
    # Map within 4-neuron perturbation-energy strata, always a derangement.
    ordered = sorted(range(16), key=lambda i: 49*circuit['up_norm'][i]**2+(7/8)**2*circuit['down_norm'][i]**2)
    perm = list(range(16)); rng = random.Random(seed+70001)
    for start in range(0,16,4):
        group = ordered[start:start+4]; shift = rng.randrange(1,4)
        for i, index in enumerate(group): perm[index] = group[(i+shift)%4]
    return {'W': {'table': table},
            'S': {'norms': {'gate': circuit['gate_norm'], 'up': circuit['up_norm'], 'down': circuit['down_norm']}},
            'M': {'table': [table[i] for i in perm]}, 'R': {}}, allowed, reference, perm


def plans(circuit):
    cfg = config()
    target = 'B' if circuit['oracle']['truth']=='A' else 'A'
    result = []
    for seed in cfg['selection_seeds']:
        info, allowed, reference, perm = views(circuit, seed)
        for arm in cfg['arms']:
            scores = (raw_scores(info[arm]['table'], target) if arm in ['W','M'] else
                      stats_scores(info[arm]['norms']) if arm=='S' else None)
            ids, norm = choose(allowed, scores, seed, cfg['top_combinations'])
            result.append({'id': f"{circuit['id']}-{arm}-{seed}", 'circuit_id': circuit['id'],
                           'layer': circuit['layer'], 'arm': arm, 'seed': seed,
                           'positions': list(ids), 'units': [circuit['units'][i] for i in ids],
                           'factors': [8.]*4, 'target': target,
                           'edit_norm': norm, 'reference_norm': reference,
                           'allowed_combinations': len(allowed),
                           'policy_input_sha256': digest(info[arm]),
                           'mismatched_permutation': perm if arm=='M' else None})
    return result
