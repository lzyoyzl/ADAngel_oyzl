#!/usr/bin/env python3
"""v122 full24 exact INT16 handoff feasibility, not a performance experiment.

New mechanism: four MMA warps hand off G128 partials to four integer scale /
accumulator warps. Unlike v41's copy producer or v98's eight MMA warps, the
four-warp operand reuse is retained. First prove that narrowing the unscaled
INT32 dot to INT16 is lossless; no quantizer or full-K overflow guard changes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / 'docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78/source_provenance.jsonl'
MIN_COVERAGE = .95  # Per sample, not a selected average; before observing data.


def observe(asq, wsq, status):
    """Sufficient Cauchy bound; rejection is NOT proof of actual overflow."""
    a, w, old = map(np.asarray, (asq, wsq, status))
    for x, tile, limit in ((a, 64, 128 * 128**2), (w, 128, 128 * 8**2)):
        if (x.dtype.kind not in 'ui' or x.ndim != 2 or x.shape[1] != 32
                or not x.shape[0] or x.shape[0] % tile
                or np.any(x < 0) or np.any(x > limit)):
            raise ValueError('exact G128 sum-squares within INT8/S4 bounds required')
    if old.shape != (a.shape[0] // 64, w.shape[0] // 128) or old.dtype.kind not in 'ui' or np.any(old > 2):
        raise ValueError('existing CTA-uniform fullK guard required')
    if np.any(old == 2):
        raise ValueError('invalid source must not reach the handoff candidate')
    am = a.astype(np.int64).reshape(-1, 64, 32).max(1)
    wm = w.astype(np.int64).reshape(-1, 128, 32).max(1)
    # Independent row/column maxima are conservative for every output and group.
    bound_squared = am[:, None, :] * wm[None, :, :]
    narrow = (bound_squared <= 32767**2).all(-1)
    eligible = (old == 0) & narrow
    total = old.size
    return dict(ctas=total, integer_ctas=int((old == 0).sum()),
        narrow_integer_ctas=int(eligible.sum()), wide_integer_ctas=int(((old == 0) & ~narrow).sum()),
        fp32_fallback_ctas=int((old == 1).sum()),
        narrow_fraction_of_all_ctas=float(eligible.sum()/total),
        narrow_fraction_of_integer_ctas=float(eligible.sum()/max(1, (old == 0).sum())),
        maximum_squared_bound=int(bound_squared.max()),
        maximum_integer_abs_bound=math.isqrt(int(bound_squared.max())),
        minimum_required_fraction=MIN_COVERAGE,
        data_gate_passed=eligible.sum()/total >= MIN_COVERAGE,
        exact_integer_bound=True, actual_G128_partials_measured=False,
        rejected_bound_does_not_imply_actual_overflow=True)


def capacity_model():
    # Capacity-only investment model, NOT an additive kernel runtime estimate.
    outputs, groups, sms, mhz = 4096**2, 32, 108, 1410
    rate = sms * mhz * 1000 * 128  # one 128-byte wavefront/SM/cycle.
    return dict(sm_count=sms, clock_mhz=mhz, wavefront_bytes=128,
        assumed_wavefronts_per_sm_cycle=1,
        original_shared_reservation_bytes=34304,
        candidate_shared_reservation_bytes=34304+2*64*64*2+2*128*4,
        handoff_int32_extra_bytes=outputs*groups*4*2,
        handoff_int16_extra_bytes=outputs*groups*2*2,
        handoff_int32_service_floor_ms=outputs*groups*4*2/rate,
        handoff_int16_service_floor_ms=outputs*groups*2*2/rate,
        synchronization_packing_metadata_costs_not_included=True,
        actual_candidate_performance_not_measured=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    p.add_argument('--raw-data', type=Path, default=Path('data/raw/llama2_7b_prefill'))
    p.add_argument('--trace-config', type=Path, default=Path('configs/trace/llama2_7b_prefill.yaml'))
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        p.error('fresh project-local output required')
    import torch
    from adangel.trace.storage import sha256_file
    from adangel.trace.prepare import _load_and_validate_raw
    from adangel.quantization import mixed_formats as mf
    from benchmark_a100_mixed_trace import inspect_inputs, inspect_raw_inputs, source_identity
    from o78_fullk_integer_metadata import prepare_fullk_metadata
    torch.set_num_threads(4)
    if torch.cuda.get_device_capability() != (8, 0):
        raise ValueError('A100 required')
    manifest, ph = inspect_inputs(args.data)
    raw, rh = inspect_raw_inputs(args.raw_data, manifest, args.trace_config)
    records = list(map(json.loads, PROVENANCE.read_text().splitlines()))
    reference = {(r['sample_id'], r['variant']): r for r in records}
    if len(records) != 48 or len(reference) != 48 or len(manifest['samples']) != 24:
        raise ValueError('unique full24 source authority required')
    raw_index = {r['sample_id']: r for r in raw['samples']}
    extensions = list((ROOT/'python/adangel').glob('_sm80*.so'))
    if len(extensions) != 1:
        raise ValueError('one unchanged production extension required')
    ext_hash = sha256_file(extensions[0])
    out.mkdir(parents=True)
    environment = dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        native_extension_sha256=ext_hash, gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, torch_cuda=torch.version.cuda,
        source_hashes={s: sha256_file(ROOT/s) for s in (
            'scripts/inspect_partial_handoff.py', 'scripts/o78_fullk_integer_metadata.py',
            'scripts/benchmark_a100_mixed_trace.py', 'python/adangel/quantization/mixed_formats.py')},
        raw_manifest_sha256=rh, prepared_manifest_sha256=ph,
        provenance_sha256=sha256_file(PROVENANCE), capacity_model=capacity_model(),
        mechanism='four_MMA_warps_to_four_scale_warps_exact_INT16_shared_handoff',
        gate_per_sample_minimum_fraction=MIN_COVERAGE, source_quantizer_unchanged=True,
        no_candidate_GEMM_launched=True, no_performance_or_MSE_measured=True)
    (out/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    rows = []
    for entry in manifest['samples']:
        sid = entry['sample_id']; re = raw_index[sid]
        record = _load_and_validate_raw(args.raw_data/re['file'], re['layer'], re['projection'])
        for variant in ('o7','o8'):
            wf, af = mf.VARIANTS[variant]
            ws = mf.quantize_source(record['weight_fp16'].cuda(), wf)
            acs = mf.quantize_source(record['activation_fp16'].cuda(), af)
            identities = dict(weight=source_identity(ws), activation=source_identity(acs))
            ref = reference[sid, variant]
            if re['sha256'] != ref['raw_sha256'] or any(identities[k] != ref[k] for k in identities):
                raise ValueError('source differs from full24 measured v99')
            aq, _ = mf.to_fixed_reference(acs); wq, _ = mf.to_fixed_reference(ws)
            squares = lambda q: q.reshape(q.shape[0],32,128).long().square().sum(-1).cpu().numpy()
            asq, wsq = squares(aq), squares(wq)
            ts = np.float32((ws if variant == 'o7' else acs)['tensor_scale'].item())
            oracle = prepare_fullk_metadata(acs['scale'].cpu().numpy(), ws['scale'].cpu().numpy(), asq, wsq,
                activation_kind='ue8m0' if variant == 'o7' else 'e4m3',
                weight_kind='e4m3' if variant == 'o7' else 'e6m2',
                activation_base_multiplier=np.float32(4) if variant == 'o7' else np.float32(ts*.25),
                weight_base_multiplier=ts if variant == 'o7' else np.float32(1))
            artifact = out/f'{sid}_{variant}_squares.npz'
            np.savez_compressed(artifact, activation=asq, weight=wsq, status=oracle['status'])
            row = dict(sample_id=sid, variant=variant, raw_sha256=re['sha256'], **identities,
                v99_source_exact=True, artifact=artifact.name, artifact_sha256=sha256_file(artifact),
                statistics=observe(asq, wsq, oracle['status']))
            rows.append(row)
            with (out/'results.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False)+'\n')
            print(sid, variant, row['statistics'], flush=True)
            del aq,wq,ws,acs,oracle
        del record
    if (sha256_file(extensions[0]) != ext_hash or sha256_file(args.data/'manifest.json') != ph
            or sha256_file(args.raw_data/'trace_manifest.json') != rh):
        raise ValueError('extension or manifest changed')
    summary = dict(records=len(rows), variants=[dict(variant=v, samples=24,
        minimum_narrow_fraction=min(r['statistics']['narrow_fraction_of_all_ctas'] for r in rows if r['variant']==v),
        passed_samples=sum(r['statistics']['data_gate_passed'] for r in rows if r['variant']==v),
        maximum_partial_bound=max(r['statistics']['maximum_integer_abs_bound'] for r in rows if r['variant']==v))
        for v in ('o7','o8')],
        investment_gate_passed=all(r['statistics']['data_gate_passed'] for r in rows),
        next_step='single_handoff_codegen_only_if_data_gate_passes',
        new_performance_measured=False, new_MSE_measured=False, production_default_changed=False)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
