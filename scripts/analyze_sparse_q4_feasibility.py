#!/usr/bin/env python3
"""v103: read-only, full24 feasibility of lossless paired-4:8 Q4 sparsity.

This is NOT a sparse kernel, pruning, a benchmark, or a new quantizer. Use
current O3 prepared Q4 and replay existing O7/O8 source preparation, requiring
exact SHA agreement with the already measured v99 sources. Count the minimum
nonzero residual for each fixed eight-wide chunk after retaining two adjacent
pairs. Do not change K order, G128 scales, production dispatch, or the extension.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VALID_METADATA = {0x4, 0x8, 0x9, 0xC, 0xD, 0xE}
REFERENCE_PROVENANCE = Path(
    'docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78/source_provenance.jsonl')
PTX_SOURCE = ('https://docs.nvidia.com/cuda/archive/12.8.0/'
              'parallel-thread-execution/index.html#sparse-matrix-storage')


def paired_decomposition(q):
    """Keep two pairs with most nonzeros; ties use increasing pair index.

    A selected pair may itself contain numerical zeros. Counting these as
    storable makes this an optimistic feasibility estimate. Always emit two
    distinct, ordered metadata indices, including for all-zero chunks.
    """
    q = np.asarray(q)
    if (q.dtype != np.int8 or q.ndim != 2 or not q.shape[0] or not q.shape[1]
            or q.shape[1] % 128 or np.any(q < -8) or np.any(q > 7)):
        raise ValueError('requires nonempty int8 Q4 matrix with G128-aligned K')
    pairs = q.reshape(q.shape[0], -1, 4, 2)
    counts = np.count_nonzero(pairs, axis=-1)
    chosen = np.sort(np.argsort(-counts, axis=-1, kind='stable')[..., :2], axis=-1)
    keep = np.zeros(counts.shape, dtype=bool)
    np.put_along_axis(keep, chosen, True, axis=-1)
    main = np.where(keep[..., None], pairs, 0).reshape(q.shape).astype(np.int8)
    residual = (q.astype(np.int16) - main).astype(np.int8)
    metadata = (chosen[..., 0] | (chosen[..., 1] << 2)).astype(np.uint8)
    if not np.isin(metadata, list(VALID_METADATA)).all():
        raise AssertionError('invalid ordered pair metadata')
    if not np.array_equal(main.astype(np.int16) + residual, q):
        raise AssertionError('lossless decomposition failed')
    return main, residual, metadata, counts


def weight_statistics(q, chunk_rows=64):
    q = np.asarray(q)
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError('positive chunk row count required')
    active_hist = np.zeros(5, dtype=np.int64)
    residual_hist = np.zeros(5, dtype=np.int64)
    zero_count = residual_count = g128_without_residual = 0
    for offset in range(0, q.shape[0], chunk_rows):
        block = q[offset:offset + chunk_rows]
        _, residual, _, counts = paired_decomposition(block)
        active = np.count_nonzero(counts, axis=-1)
        residual_per8 = np.count_nonzero(residual.reshape(block.shape[0], -1, 8), axis=-1)
        active_hist += np.bincount(active.ravel(), minlength=5)
        residual_hist += np.bincount(residual_per8.ravel(), minlength=5)
        zero_count += int(np.count_nonzero(block == 0))
        residual_count += int(np.count_nonzero(residual))
        g128_without_residual += int(np.count_nonzero(
            residual_per8.reshape(block.shape[0], -1, 16).sum(-1) == 0))
    total = q.size
    chunks = total // 8
    if int(active_hist.sum()) != chunks or int(residual_hist.sum()) != chunks:
        raise AssertionError('chunk counts do not cover all weights')
    return dict(shape=list(q.shape), weights=total, q4_zero_count=zero_count,
        q4_zero_fraction=zero_count / total,
        active_pair_histogram=active_hist.tolist(),
        minimum_residual_nonzeros_per8_histogram=residual_hist.tolist(),
        minimum_residual_nonzeros=residual_count,
        minimum_residual_fraction=residual_count / total,
        chunks_requiring_residual=int(residual_hist[1:].sum()),
        chunks_requiring_residual_fraction=float(residual_hist[1:].sum()) / chunks,
        g128_without_residual=g128_without_residual,
        g128_without_residual_fraction=g128_without_residual / (total // 128),
        decomposition_exact=True, ordered_metadata_valid=True,
        scope='fixed_K_order_per_row_paired4of8_not_global_reordering',
        selected_pairs_may_contain_numerical_zeros=True)


def scalar_residual_model(weight_count, residual_count, m=4096,
                          sm_count=108, clock_mhz=1410, lanes_per_sm_cycle=128):
    """Optimistic *scalar* SIMT correction only, not a whole-kernel roof.

    One exact scalar IMAD for each retained weight/output-row MAC; no gathering,
    scales, addressing, loads, or synchronization charged. 128 lane operations
    is a declared generous 4 SMSP x 32 lanes/cycle issue model, not a measured
    IMAD throughput. Vector dot products / Tensor Cores are NOT this model.
    """
    if not 0 <= residual_count <= weight_count or min(m, sm_count, clock_mhz, lanes_per_sm_cycle) <= 0:
        raise ValueError('invalid scalar model counts/capacity')
    macs = m * residual_count
    cycles_per_ms = clock_mhz * 1000
    return dict(residual_scalar_macs=macs, assumed_sm_count=sm_count,
        assumed_clock_mhz=clock_mhz,
        assumed_lane_corrections_per_sm_cycle=lanes_per_sm_cycle,
        scalar_correction_only_optimistic_ms=macs / (sm_count * cycles_per_ms * lanes_per_sm_cycle),
        ideal_four_mac_vector_instruction_ms=macs / (4 * sm_count * cycles_per_ms * lanes_per_sm_cycle),
        vector_value_excludes_gather_packing_and_dependencies=True,
        mma_can_overlap_so_no_addition_to_mma_floor=True,
        measured_kernel_peak=False, measured_speedup=False)


def summarize(rows):
    if len(rows) != 72 or len({(r['sample_id'], r['variant']) for r in rows}) != 72:
        raise ValueError('exactly 24 x 3 unique records required')
    ids = {r['sample_id'] for r in rows}
    if len(ids) != 24 or {(s, v) for s, v in itertools.product(ids, ('o3', 'o7', 'o8'))} != {
            (r['sample_id'], r['variant']) for r in rows}:
        raise ValueError('incomplete variant coverage')
    fields = ('q4_zero_fraction', 'minimum_residual_fraction',
              'chunks_requiring_residual_fraction', 'g128_without_residual_fraction')
    result = []
    for variant in ('o3', 'o7', 'o8'):
        selected = [r for r in rows if r['variant'] == variant]
        result.append(dict(variant=variant, samples=24,
            statistics={key: dict(median=statistics.median(r['statistics'][key] for r in selected),
                                  min=min(r['statistics'][key] for r in selected),
                                  max=max(r['statistics'][key] for r in selected)) for key in fields},
            scalar_model_median_ms=statistics.median(
                r['model']['scalar_correction_only_optimistic_ms'] for r in selected),
            ideal_vector_model_median_ms=statistics.median(
                r['model']['ideal_four_mac_vector_instruction_ms'] for r in selected)))
    return dict(scope='read_only_full24_Q4_data_feasibility_not_GPU_performance',
        records=72, variants=result, production_default_changed=False,
        pruning_performed=False, new_MSE_measured=False, new_GEMM_measured=False,
        residual_format_or_kernel_implemented=False, ptx_rule=PTX_SOURCE)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    p.add_argument('--raw-data', type=Path, default=Path('data/raw/llama2_7b_prefill'))
    p.add_argument('--trace-config', type=Path, default=Path('configs/trace/llama2_7b_prefill.yaml'))
    p.add_argument('--source-provenance', type=Path, default=REFERENCE_PROVENANCE)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT):
        p.error('fresh repository output required')
    import torch
    from adangel.trace.storage import load_prepared, sha256_file
    from adangel.trace.prepare import _load_and_validate_raw
    from adangel.quantization.mxfp4 import unpack_int4_tensor, mxfp4_to_q4_packed
    from adangel.quantization import mixed_formats as mf
    from benchmark_a100_mixed_trace import inspect_inputs, inspect_raw_inputs, source_identity
    torch.set_num_threads(4)
    manifest, prepared_hash = inspect_inputs(args.data)
    raw, raw_hash = inspect_raw_inputs(args.raw_data, manifest, args.trace_config)
    expected_rows = [json.loads(line) for line in args.source_provenance.read_text().splitlines() if line.strip()]
    expected = {(r['sample_id'], r['variant']): r for r in expected_rows}
    if len(expected) != 48 or len(expected_rows) != 48:
        raise ValueError('exact v99 full24 O7/O8 source provenance required')
    raw_index = {r['sample_id']: r for r in raw['samples']}
    extensions = list((ROOT / 'python/adangel').glob('_sm80*.so'))
    if len(extensions) != 1:
        raise ValueError('one existing SM80 extension expected; no rebuild or import')
    extension_hash = sha256_file(extensions[0])
    args.output.mkdir(parents=True)
    environment = dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_hashes={name: sha256_file(ROOT / name) for name in
            ('scripts/analyze_sparse_q4_feasibility.py', 'python/adangel/quantization/mixed_formats.py',
             'python/adangel/quantization/mxfp4.py')},
        torch=torch.__version__, cuda=torch.version.cuda, source_device=args.device,
        prepared_manifest_sha256=prepared_hash, raw_manifest_sha256=raw_hash,
        v99_provenance_sha256=sha256_file(args.source_provenance),
        native_extension_path=str(extensions[0].relative_to(ROOT)), native_extension_sha256=extension_hash,
        no_production_GEMM_or_candidate_kernel_launched=True, no_performance_timing=True,
        existing_torch_source_reference_on_GPU=(args.device == 'cuda'),
        source_quantizer_reused=True, quantization_semantics_changed=False,
        model_assumptions=dict(sm_count=108, clock_mhz=1410, scalar_lane_ops_per_cycle=128),
        ptx_rule=PTX_SOURCE)
    (args.output / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    rows = []
    for entry in manifest['samples']:
        sid = entry['sample_id']
        re = raw_index[sid]
        prepared = load_prepared(args.data / entry['file'], device='cpu')
        if not torch.equal(prepared.W_q4, mxfp4_to_q4_packed(prepared.W_mxfp4_g128)):
            raise ValueError('existing O3 Q4 preparation mismatch: ' + sid)
        q3 = unpack_int4_tensor(prepared.W_q4).numpy()
        variants = [('o3', q3, dict(prepared_sha256=entry['sha256'], current_Q4_exact=True))]
        record = _load_and_validate_raw(args.raw_data / re['file'], re['layer'], re['projection'])
        for variant in ('o7', 'o8'):
            source = mf.quantize_source(record['weight_fp16'].to(args.device), mf.VARIANTS[variant][0])
            identity = source_identity(source)
            ref = expected[sid, variant]
            if ref['raw_sha256'] != re['sha256'] or identity != ref['weight']:
                raise ValueError('source differs from measured v99 source: ' + sid + '/' + variant)
            q, scale = mf.to_fixed_reference(source)
            variants.append((variant, q.cpu().numpy(), dict(v99_source_exact=True, weight_source=identity)))
            del source, q, scale
        for variant, q, provenance in variants:
            measured = weight_statistics(q)
            row = dict(sample_id=sid, variant=variant, raw_sha256=re['sha256'], provenance=provenance,
                fixed_weight_sha256=hashlib.sha256(q.tobytes()).hexdigest(), statistics=measured,
                model=scalar_residual_model(measured['weights'], measured['minimum_residual_nonzeros']))
            rows.append(row)
            with (args.output / 'results.jsonl').open('a') as out:
                out.write(json.dumps(row, allow_nan=False) + '\n')
        print(sid, 'exact sources and lossless pair statistics passed', flush=True)
        del record, prepared, variants, q3, q
    if sha256_file(extensions[0]) != extension_hash:
        raise AssertionError('production extension changed')
    result = summarize(rows)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
