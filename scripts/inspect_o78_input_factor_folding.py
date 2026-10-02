#!/usr/bin/env python3
"""Read-only v74 gate: can full-K scale factors fit in existing INT8/INT4 inputs?

This tests exact mathematical representability, not a new quantizer/kernel,
FP32 epilogue acceptance, timing or MSE. GCD removal is deliberately optimistic:
even the smallest exact integer representation must fit the original type.
"""
import argparse
from collections import Counter
from functools import reduce
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from o78_fullk_integer_metadata import dyadic_scale

ROOT = Path(__file__).resolve().parents[1]


def inspect_operand(codes, minima, maxima, gcds, kind, bits, tile_rows):
    arrays = [np.asarray(x) for x in (codes, minima, maxima, gcds)]
    if (bits not in (4, 8) or tile_rows < 1 or arrays[0].ndim != 2 or
            not arrays[0].size or arrays[0].shape[0] % tile_rows or
            any(x.shape != arrays[0].shape or x.dtype.kind not in 'iu' for x in arrays)):
        raise ValueError('matching integer group arrays and aligned rows required')
    codes, minima, maxima, gcds = arrays
    rows = []
    for cs, lo, hi, gs in zip(codes, minima, maxima, gcds):
        terms = [dyadic_scale(int(c), kind) for c in cs]
        if any(int(a) > int(b) or int(g) < 0 or
               (g == 0 and (a != 0 or b != 0)) or
               (g > 0 and (int(a) % int(g) or int(b) % int(g)))
               for a, b, g in zip(lo, hi, gs)):
            raise ValueError('invalid exact payload group statistics')
        # Zero payload/scale groups impose no representation constraint.
        anchor = min((e for (m, e), g in zip(terms, gs) if m and g), default=0)
        factors = [m << (e - anchor) if m and g else 0 for (m, e), g in zip(terms, gs)]
        divisor = reduce(math.gcd, (int(g) * f for g, f in zip(gs, factors)), 0) or 1
        lower = min(int(a) * f for a, f in zip(lo, factors)) // divisor
        upper = max(int(b) * f for b, f in zip(hi, factors)) // divisor
        required = max(1, max(upper, 0).bit_length() + 1, max(-lower - 1, 0).bit_length() + 1)
        fits = -(1 << (bits - 1)) <= lower and upper < (1 << (bits - 1))
        rows.append(dict(anchor=anchor, removed_gcd=divisor, minimum=lower, maximum=upper,
                         minimum_signed_bits=required, fits=bool(fits)))
    fitting_rows = sum(r['fits'] for r in rows)
    tile_mask = [all(r['fits'] for r in rows[i:i + tile_rows]) for i in range(0, len(rows), tile_rows)]
    return dict(target_bits=bits, tile_rows=tile_rows, rows=rows, fitting_rows=fitting_rows,
                total_rows=len(rows), fitting_row_fraction=fitting_rows / len(rows),
                fitting_tiles=sum(tile_mask), total_tiles=len(tile_mask),
                fitting_tile_fraction=sum(tile_mask) / len(tile_mask), tile_mask=tile_mask,
                minimum_signed_bits_histogram=dict(sorted(Counter(r['minimum_signed_bits'] for r in rows).items())),
                interpretation='optimistic_exact_input_range_gate_not_runtime_or_performance_acceptance')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, choices=(4, 24), default=4)
    p.add_argument('--data', type=Path, default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    p.add_argument('--raw-data', type=Path, default=Path('data/raw/llama2_7b_prefill'))
    p.add_argument('--trace-config', type=Path, default=Path('configs/trace/llama2_7b_prefill.yaml'))
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT):
        p.error('fresh repository output required')
    import torch
    from adangel.quantization import mixed_formats as mf
    from adangel.trace.prepare import _load_and_validate_raw
    from adangel.trace.storage import sha256_file
    from benchmark_a100_mixed_trace import inspect_inputs, inspect_raw_inputs, source_identity, tensor_identity
    torch.cuda.init(); torch.set_num_threads(4)
    if torch.cuda.get_device_capability() != (8, 0):
        p.error('A100 SM80 required')
    prepared, ph = inspect_inputs(a.data)
    raw, rh = inspect_raw_inputs(a.raw_data, prepared, a.trace_config)
    a.output.mkdir(parents=True)
    result = dict(scope='exact_input_factor_folding_feasibility_not_new_kernel_timing_or_MSE',
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  script_sha256=sha256_file(Path(__file__)), prepared_manifest_sha256=ph,
                  raw_manifest_sha256=rh, gpu=torch.cuda.get_device_name(), torch=torch.__version__, samples=[])
    for entry in raw['samples'][:a.samples]:
        path = a.raw_data / entry['file']
        if sha256_file(path) != entry['sha256']:
            raise ValueError('raw sample SHA mismatch')
        record = _load_and_validate_raw(path, entry['layer'], entry['projection'])
        for variant, (wf, af) in mf.VARIANTS.items():
            row = dict(sample_id=entry['sample_id'], variant=variant, raw_sha256=entry['sha256'])
            for side, fmt, tensor, kind, bits, tile in (
                    ('a', af, record['activation_fp16'], 'ue8m0' if variant == 'o7' else 'e4m3', 8, 64),
                    ('w', wf, record['weight_fp16'], 'e4m3' if variant == 'o7' else 'e6m2', 4, 128)):
                source = mf.quantize_source(tensor.cuda(), fmt)
                q, _ = mf.to_fixed_reference(source)
                if q.dtype != torch.int8 or tuple(q.shape) != (4096, 4096):
                    raise ValueError('expected original fixed-point payload')
                values = q.cpu().numpy().astype(np.int16).reshape(4096, 32, 128)
                lo, hi = values.min(-1), values.max(-1)
                gs = np.gcd.reduce(values, axis=-1)
                codes = source['scale'].cpu().numpy()
                stats = inspect_operand(codes, lo, hi, gs, kind, bits, tile)
                detail = dict(kind=kind, source=source_identity(source), payload=tensor_identity(q),
                              codes=codes.tolist(), minima=lo.tolist(), maxima=hi.tolist(), gcds=gs.tolist(),
                              analysis=stats)
                name = f"{entry['sample_id']}_{variant}_{side}.json"
                (a.output / name).write_text(json.dumps(detail, separators=(',', ':'), allow_nan=False) + '\n')
                row[side] = {k: v for k, v in stats.items() if k not in ('rows', 'tile_mask')}
                row[side].update(file=name, sha256=sha256_file(a.output / name))
                del source, q, values, stats, detail
            row['ctas_fitting_both_operands'] = row['a']['fitting_tiles'] * row['w']['fitting_tiles']
            result['samples'].append(row)
            print(entry['sample_id'], variant, 'A row/tile fits=', row['a']['fitting_rows'], row['a']['fitting_tiles'],
                  'W row/tile fits=', row['w']['fitting_rows'], row['w']['fitting_tiles'], flush=True)
        del record
    if sha256_file(a.data / 'manifest.json') != ph or sha256_file(a.raw_data / 'trace_manifest.json') != rh:
        raise ValueError('manifest changed')
    (a.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('EXACT INPUT RANGE INSPECTION COMPLETE; no kernel or performance/MSE claim', flush=True)


if __name__ == '__main__':
    main()
