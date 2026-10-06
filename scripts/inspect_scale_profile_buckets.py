#!/usr/bin/env python3
"""v108: full-K scale-profile homogeneity, not v105 warp-local equality.

Read-only opportunity gate for exact profile bucketing/permutation. O3 checks
K-constant W scales; O7/O8 check proportional A-factor profiles across all32
groups. No source value/scale is changed, no GEMM or new quantizer is launched.
An optimistic removable instruction budget is NOT a predicted speedup.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OBSERVATIONS=ROOT/'docs/evidence/a100_o378_roof_v105/reports/o378_roof_v105_runtime_factor_r2'
REMOVABLE=64
LOOPS={'o3':323,'o7':383,'o8':383}
THRESHOLD_PERCENT=5.0
PROVENANCE=Path('docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78/source_provenance.jsonl')


def observation_authority(root,expected_ids,sha256_file):
    """Reuse frozen v105 snapshots only with their original-source receipts."""
    base=root/OBSERVATIONS.relative_to(ROOT)
    env=json.loads((base/'environment.json').read_text())
    for path,digest in env['source_hashes'].items():
        if sha256_file(root/path)!=digest:raise ValueError('v105 source receipt drift')
    if sha256_file(root/PROVENANCE)!=env['provenance_sha256']:
        raise ValueError('original source provenance drift')
    provenance=list(map(json.loads,(root/PROVENANCE).read_text().splitlines()))
    authority={(r['sample_id'],r['variant']):r for r in provenance}
    observed=list(map(json.loads,(base/'results.jsonl').read_text().splitlines()))
    old={(r['sample_id'],r['variant']):r for r in observed}
    keys={(s,v) for s in expected_ids for v in ('o7','o8')}
    if len(expected_ids)!=24 or len(observed)!=48 or len(provenance)!=48 or set(old)!=keys or set(authority)!=keys:
        raise ValueError('v105 full24 original-source authority required')
    for key,r in old.items():
        a=authority[key]
        if not r['v99_source_exact'] or r['source']!=a['activation'] or r['raw_sha256']!=a['raw_sha256']:
            raise ValueError('v105 snapshot is not original v99 source')
        if sha256_file(base/r['artifact'])!=r['artifact_sha256']:
            raise ValueError('v105 factor snapshot drift')
    return env,old


def activation_profiles(factors,row_status):
    f=np.asarray(factors);s=np.asarray(row_status)
    if f.dtype!=np.int32 or f.ndim!=2 or f.shape[0]!=32 or not f.shape[1] or f.shape[1]%64 or np.any(f<0):
        raise ValueError('group-major nonnegative INT32 factors [32,M64] required')
    if s.dtype!=np.uint32 or s.shape!=(f.shape[1],) or np.any(s>2):
        raise ValueError('exact operand guard status required')
    gcd=np.gcd.reduce(f,axis=0)
    valid=(s==0)&(gcd>0)
    normalized=f[:,valid]//gcd[valid][None,:]
    if not normalized.size:
        counts=np.asarray([],dtype=np.int64)
    else:
        _,counts=np.unique(normalized.T,axis=0,return_counts=True)
    tiles=int((counts//64).sum());rows=tiles*64
    # Best possible packing after permutation; incomplete tails remain generic.
    old_tiles=0
    for begin in range(0,f.shape[1],64):
        if valid[begin:begin+64].all():
            p=f[:,begin:begin+64]//gcd[None,begin:begin+64]
            old_tiles+=int(np.array_equal(p,np.broadcast_to(p[:,:1],p.shape)))
    return dict(rows=f.shape[1],valid_rows=int(valid.sum()),excluded_rows=int((~valid).sum()),
        unique_profiles=len(counts),largest_profile_rows=int(counts.max(initial=0)),
        profile_size_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(counts,return_counts=True))},
        old_homogeneous_tiles=old_tiles,optimally_bucketed_tiles=tiles,
        tile_rows=64,covered_rows=rows,covered_fraction=rows/f.shape[1],
        optimistic_loop_instruction_reduction_percent=rows/f.shape[1]*REMOVABLE/383*100,
        mechanism='fold_one_common_A_profile_into_cached_W_factor_panels; keep_row_gcd_in_base',
        exclusions='permutation/sort, folded metadata, row-index stores, range/base guards and dispatch costs')


def constant_weights(scale_codes):
    c=np.asarray(scale_codes)
    if c.dtype!=np.uint8 or c.ndim!=2 or c.shape[1]!=32 or not c.shape[0] or c.shape[0]%128:
        raise ValueError('natural uint8 W scale [N128,32] required')
    if np.any(c==255):raise ValueError('invalid UE8M0 255')
    eligible=(c==c[:,:1]).all(axis=1)&(c[:,0]>=1)
    count=int(eligible.sum());tiles=count//128;covered=tiles*128
    old=int(eligible.reshape(-1,128).all(axis=1).sum())
    return dict(columns=c.shape[0],K_constant_columns=count,
        old_homogeneous_tiles=old,optimally_bucketed_tiles=tiles,tile_columns=128,
        covered_columns=covered,covered_fraction=covered/c.shape[0],
        optimistic_loop_instruction_reduction_percent=covered/c.shape[0]*REMOVABLE/323*100,
        mechanism='move_K_constant_W_scale_to_epilogue; merge_high_shift_and_final_acc_before_low_MMA',
        exclusions='weight permutation, original-column scatter stores, guard and dispatch costs')


def summarize(rows):
    ids={r['sample_id'] for r in rows};index={(r['sample_id'],r['variant']):r for r in rows}
    if len(ids)!=24 or len(rows)!=72 or len(index)!=72 or set(index)!={(s,v) for s in ids for v in LOOPS}:
        raise ValueError('all24 unique O3/O7/O8 observations required')
    result=[]
    for variant in LOOPS:
        selected=[index[s,variant]['statistics'] for s in sorted(ids)]
        work=statistics.mean(x['optimistic_loop_instruction_reduction_percent'] for x in selected)
        result.append(dict(variant=variant,samples=24,
            mean_covered_fraction=statistics.mean(x['covered_fraction'] for x in selected),
            mean_optimistic_loop_work_reduction_percent=work,
            mean_old_homogeneous_tiles=statistics.mean(x['old_homogeneous_tiles'] for x in selected),
            mean_bucketed_tiles=statistics.mean(x['optimally_bucketed_tiles'] for x in selected),
            opportunity_gate=work>=THRESHOLD_PERCENT,threshold_percent=THRESHOLD_PERCENT,
            decision='compile_one_fixed_bucketed_candidate_if_gate_passes_else_stop'))
    return dict(scope='exact_full_K_scale_profile_data_gate_not_kernel_peak_or_performance',variants=result,
        new_kernel_implemented=False,new_GEMM_measured=False,new_MSE_measured=False,
        source_quantization_unchanged=True,production_default_changed=False,
        no_filtering=True,no_new_source_quantization=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    args=p.parse_args();out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    from benchmark_a100_mixed_trace import inspect_inputs
    from adangel.trace.storage import sha256_file
    import torch
    torch.set_num_threads(4)
    manifest,mh=inspect_inputs(args.data)
    prior_env,old=observation_authority(ROOT,{e['sample_id'] for e in manifest['samples']},sha256_file)
    if mh!=prior_env['prepared_manifest_sha256']:raise ValueError('prepared identity differs from v105')
    out.mkdir(parents=True);rows=[]
    extension=list((ROOT/'python/adangel').glob('_sm80*.so'))
    if len(extension)!=1:raise ValueError('one unchanged existing extension required')
    ext=sha256_file(extension[0])
    for entry in manifest['samples']:
        sid=entry['sample_id'];path=args.data/entry['file']
        record=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        if record['sample_id']!=sid or record['shape']!=[4096,4096,4096]:raise ValueError('embedded shape/identity drift')
        c=record['W_scale_g128']
        if c.dtype!=torch.uint8 or not c.is_contiguous() or tuple(c.shape)!=(4096,32):
            raise ValueError('formal W scale layout/dtype mismatch')
        codes=c.numpy().copy();artifact=out/f'{sid}_o3_scale_codes.npz'
        np.savez_compressed(artifact,scale_codes=codes)
        rows.append(dict(sample_id=sid,variant='o3',prepared_file=entry['file'],
            prepared_sha256=entry['sha256'],artifact=artifact.name,artifact_sha256=sha256_file(artifact),
            scale_codes_sha256=hashlib.sha256(codes.tobytes()).hexdigest(),statistics=constant_weights(codes)))
        del c,record
        for variant in ('o7','o8'):
            previous=old[sid,variant];source=OBSERVATIONS/previous['artifact']
            if sha256_file(source)!=previous['artifact_sha256']:raise ValueError('v105 factor snapshot drift')
            with np.load(source,allow_pickle=False) as arrays:
                if hashlib.sha256(arrays['factors'].tobytes()).hexdigest()!=previous['factors_sha256']:
                    raise ValueError('v105 factor array digest drift')
                result=activation_profiles(arrays['factors'],arrays['row_status'])
            rows.append(dict(sample_id=sid,variant=variant,artifact=str(source.relative_to(ROOT)),
                artifact_sha256=previous['artifact_sha256'],source=previous['source'],
                raw_sha256=previous['raw_sha256'],statistics=result))
        print(sid,'full-K scale-profile observation passed',flush=True)
    if sha256_file(extension[0])!=ext or sha256_file(args.data/'manifest.json')!=mh:
        raise ValueError('extension or prepared manifest changed')
    env=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=sha256_file(Path(__file__)),prepared_manifest_sha256=mh,
        v105_results_sha256=sha256_file(OBSERVATIONS/'results.jsonl'),
        v105_environment_sha256=sha256_file(OBSERVATIONS/'environment.json'),
        source_provenance_sha256=sha256_file(ROOT/PROVENANCE),
        native_extension_sha256=ext,no_GPU_kernel_launched=True,no_new_quantization=True)
    (out/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    (out/'results.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    result=summarize(rows);(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
