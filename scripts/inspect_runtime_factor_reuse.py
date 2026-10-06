#!/usr/bin/env python3
"""v105 read-only full24 opportunity gate, not repeat static factor caching.

Check actual SIMD-wide equality of four row-factor vectors in best v78.
Do not mistake per-lane coincidences for eliminated warp instructions.
Reuse the exact source quantizer/metadata; no GEMM, performance/MSE claim,
candidate kernel, native rebuild or production default change.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROVENANCE=Path('docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78/source_provenance.jsonl')
LOOP_INSTRUCTIONS=383


def factor_observation(factors,row_status):
    f=np.asarray(factors);st=np.asarray(row_status)
    if f.dtype!=np.int32 or f.ndim!=2 or not f.size or f.shape[1]%32 or np.any(f<0):
        raise ValueError('nonnegative group-major INT32 factors and M32 alignment required')
    if st.shape!=(f.shape[1],) or st.dtype!=np.uint32 or np.any(st>2):
        raise ValueError('exact row-status vector required')
    # Host CuTe verifier checks [base+r, base+r+8, base+r+16, base+r+24].
    panels=f.reshape(f.shape[0],-1,4,8).reshape(-1,4,8)
    valid_rows=(st==0).reshape(-1,32).all(-1)
    valid=np.tile(valid_rows,f.shape[0]);p=panels[valid]
    if not len(p):raise ValueError('no representable warp panels')
    is_new=np.ones((len(p),4),dtype=bool)
    for i in range(1,4):
        is_new[:,i]=~(p[:,i,None,:]==p[:,:i,:]).all(-1).any(-1)
    trivial=((p==1).all(-1)|(p==0).all(-1))
    nontrivial=(is_new&~trivial).sum(-1)
    same4=(p==p[:,:1,:]).all(-1).all(-1)
    same4_saved=np.where(same4,64-16*nontrivial,0)
    ideal_saved=64-16*nontrivial
    # Per-lane statistics are a deliberately optimistic contrast only.
    q=p.transpose(0,2,1).reshape(-1,4)
    q_new=np.ones(q.shape,dtype=bool)
    for i in range(1,4):q_new[:,i]=~(q[:,i,None]==q[:,:i]).any(-1)
    q_nontrivial=(q_new&(q!=0)&(q!=1)).sum(-1)
    hist=lambda x,size:np.bincount(x,minlength=size).tolist()
    return dict(shape=list(f.shape),warp_panels=int(len(panels)),representable_warp_panels=int(len(p)),
        excluded_warp_panels=int((~valid).sum()),quad_offsets=[0,8,16,24],
        per_lane_nontrivial_unique_histogram=hist(q_nontrivial,5),
        warp_nontrivial_unique_vector_histogram=hist(nontrivial,5),
        warp_same4_panels=int(same4.sum()),warp_same4_fraction=float(same4.mean()),
        per_lane_same4_fraction=float((q==q[:,:1]).all(-1).mean()),
        same4_saved_coefficient_histogram=hist(same4_saved,65),
        ideal_any_pattern_saved_coefficient_histogram=hist(ideal_saved,65),
        same4_mean_saved_coefficient_imad=float(same4_saved.mean()),
        ideal_any_pattern_mean_saved_coefficient_imad=float(ideal_saved.mean()),
        same4_optimistic_loop_work_reduction_percent=float(same4_saved.mean()/LOOP_INSTRUCTIONS*100),
        ideal_any_pattern_optimistic_loop_work_reduction_percent=float(ideal_saved.mean()/LOOP_INSTRUCTIONS*100),
        existing_coefficient_imad_per_loop=64,existing_accumulate_imad_per_loop=64,
        existing_integer_loop_instructions=LOOP_INSTRUCTIONS,
        exclusions='classification/branches/loads/cache/register cost and W/CTA fallback ignored',
        interpretation='optimistic_instruction_work_opportunity_not_speedup_or_kernel_peak')


def summarize(rows):
    keys={(r['sample_id'],r['variant']) for r in rows};ids={r['sample_id'] for r in rows}
    if len(ids)!=24 or len(rows)!=48 or keys!={(s,v) for s in ids for v in ('o7','o8')}:
        raise ValueError('complete unique full24 O7/O8 observations required')
    variants=[]
    for v in ('o7','o8'):
        selected=[r['statistics'] for r in rows if r['variant']==v]
        avg=statistics.mean(s['same4_optimistic_loop_work_reduction_percent'] for s in selected)
        variants.append(dict(variant=v,samples=24,
            mean_warp_same4_fraction=statistics.mean(s['warp_same4_fraction'] for s in selected),
            mean_per_lane_same4_fraction=statistics.mean(s['per_lane_same4_fraction'] for s in selected),
            mean_same4_optimistic_loop_work_reduction_percent=avg,
            mean_ideal_any_pattern_optimistic_loop_work_reduction_percent=statistics.mean(
                s['ideal_any_pattern_optimistic_loop_work_reduction_percent'] for s in selected),
            raw_opportunity_gate=avg>=5.0,raw_opportunity_threshold_percent=5.0,
            decision='compile_one_same4_candidate_if_gate_passes_else_stop_no_GPU_timing'))
    return dict(scope='runtime_factor_equality_full24_read_only_not_kernel_performance',variants=variants,
        new_MSE_measured=False,new_GEMM_measured=False,new_kernel_implemented=False,
        production_default_changed=False,O3_unchanged=True,
        O3_reason='activation_row_scale_already_outside_G128_loop',
        no_filtering=True,no_quantization_or_scale_change=True)


def coordinate_check(output):
    from benchmark_o78_eight_chain_probe import checked
    receipt=checked(ROOT/'reports/o378_roof_v78_codegen')
    cuda=Path('/usr/local/cuda-12.8');cutlass=ROOT/'third_party/cutlass-src'
    if subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()!=\
            'db1c288993354c88e551c40c19a8fb93a774a241':raise ValueError('pinned CUTLASS required')
    if 'release 12.8' not in subprocess.check_output([str(cuda/'bin/nvcc'),'--version'],text=True):
        raise ValueError('pinned CUDA12.8 required')
    source=ROOT/'tests/cuda/validate_factor_observation_coordinates.cu'
    binary=output/'validate_coordinates'
    cmd=[str(cuda/'bin/nvcc'),'-O2','-std=c++17','-arch=sm_80','--expt-relaxed-constexpr',
         '-I'+str(cutlass/'include'),str(source),'-o',str(binary)]
    temporary=ROOT/'tmp';temporary.mkdir(exist_ok=True)
    result=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          env={**os.environ,'TMPDIR':str(temporary)})
    (output/'coordinate_build.log').write_text(result.stdout)
    if result.returncode:raise RuntimeError('CuTe host coordinate verification build failed')
    result=json.loads(subprocess.check_output([str(binary)],text=True))
    if not result['passed'] or result['gpu_execution']:raise ValueError('host-only coordinate gate failed')
    (output/'coordinates.json').write_text(json.dumps(result,indent=2)+'\n')
    return dict(command=cmd,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        result=result,existing_best_receipt=receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    p.add_argument('--raw-data',type=Path,default=Path('data/raw/llama2_7b_prefill'))
    p.add_argument('--trace-config',type=Path,default=Path('configs/trace/llama2_7b_prefill.yaml'))
    args=p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT):p.error('fresh project output required')
    import torch
    from adangel.trace.storage import sha256_file
    from adangel.trace.prepare import _load_and_validate_raw
    from adangel.quantization import mixed_formats as mf
    from benchmark_a100_mixed_trace import inspect_inputs,inspect_raw_inputs,source_identity
    from o78_fullk_integer_metadata import operand_metadata
    from benchmark_o78_fullk_gpu_prepare import expected_operand_gpu
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if torch.cuda.get_device_capability()!=(8,0):raise ValueError('A100 required')
    manifest,ph=inspect_inputs(args.data);raw,rh=inspect_raw_inputs(args.raw_data,manifest,args.trace_config)
    authority=list(map(json.loads,(ROOT/PROVENANCE).read_text().splitlines()))
    old={(r['sample_id'],r['variant']):r for r in authority}
    if len(authority)!=48 or len(old)!=48:raise ValueError('source authority incomplete')
    raw_index={r['sample_id']:r for r in raw['samples']}
    extensions=list((ROOT/'python/adangel').glob('_sm80*.so'))
    if len(extensions)!=1:raise ValueError('one existing extension required')
    ext_sha=sha256_file(extensions[0]);args.output.mkdir(parents=True)
    coordinates=coordinate_check(args.output)
    sources=('scripts/inspect_runtime_factor_reuse.py','tests/cuda/validate_factor_observation_coordinates.cu',
             'scripts/o78_fullk_integer_metadata.py','python/adangel/quantization/mixed_formats.py')
    environment=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_hashes={s:sha256_file(ROOT/s) for s in sources},coordinates=coordinates,
        raw_manifest_sha256=rh,prepared_manifest_sha256=ph,provenance_sha256=sha256_file(ROOT/PROVENANCE),
        native_extension_sha256=ext_sha,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        existing_source_quantizer_on_GPU=True,no_native_GEMM_or_candidate_launched=True,
        no_new_CUDA_kernel=True,no_performance_timing=True,no_new_MSE=True,
        factor_guard='existing_operand_metadata_and_actual_GPU_epilogue_row_guard',
        source_policy='original_FP16_activation_direct_quantization_full_v99_identity')
    (args.output/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    rows=[]
    for entry in manifest['samples']:
        sid=entry['sample_id'];re=raw_index[sid]
        record=_load_and_validate_raw(args.raw_data/re['file'],re['layer'],re['projection'])
        for variant in ('o7','o8'):
            src=mf.quantize_source(record['activation_fp16'].cuda(),mf.VARIANTS[variant][1])
            identity=source_identity(src)
            if identity!=old[sid,variant]['activation'] or re['sha256']!=old[sid,variant]['raw_sha256']:
                raise ValueError('source differs from measured v99')
            q,_=mf.to_fixed_reference(src)
            sq=q.reshape(q.shape[0],32,128).long().square().sum(-1).cpu().numpy()
            codes=src['scale'].cpu().numpy()
            multiplier=np.float32(4 if variant=='o7' else np.float32(src['tensor_scale'].item())*np.float32(.25))
            kind='ue8m0' if variant=='o7' else 'e4m3'
            cpu=operand_metadata(codes,sq,kind,multiplier)
            gpu=expected_operand_gpu(cpu,activation=True)
            artifact=args.output/f'{sid}_{variant}_factor_observation.npz'
            np.savez_compressed(artifact,factors=gpu['factors'],row_status=gpu['row_status'],scale_codes=codes,
                                group_sum_squares=sq,base_multiplier=multiplier)
            observation=factor_observation(gpu['factors'],gpu['row_status'])
            row=dict(sample_id=sid,variant=variant,source=identity,v99_source_exact=True,raw_sha256=re['sha256'],
                scale_kind=kind,artifact=artifact.name,artifact_sha256=sha256_file(artifact),
                factors_sha256=hashlib.sha256(gpu['factors'].tobytes()).hexdigest(),statistics=observation)
            rows.append(row)
            with (args.output/'results.jsonl').open('a') as out:out.write(json.dumps(row,allow_nan=False)+'\n')
            print(sid,variant,'runtime factor observation passed',flush=True)
            del src,q,sq,codes,cpu,gpu
        del record
    if sha256_file(extensions[0])!=ext_sha or sha256_file(args.data/'manifest.json')!=ph or\
            sha256_file(args.raw_data/'trace_manifest.json')!=rh:raise ValueError('extension/manifest changed')
    result=summarize(rows);(args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
