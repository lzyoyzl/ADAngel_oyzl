#!/usr/bin/env python3
"""One targeted noise recheck. Never replace/filter the three-round main run."""
import argparse
import json
from pathlib import Path
import statistics as st
from benchmark_a100_best_sizes import ROOT, prepare, call, check_path, mse, REFERENCES
from benchmark_a100_mixed import aligned_timings
from benchmark_a100_o1 import stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT):p.error('fresh repository output required')
    original=[json.loads(x) for x in (args.input/'results.jsonl').read_text().splitlines()]
    targets=set()
    for r in original:
        stage='gemm' if r['mode']=='compute_only' else 'total'
        if r['stages'][stage]['cv_percent']>=3 and max(r['raw_ms'][stage])>1.2*r['stages'][stage]['median_ms']:
            targets.add((r['sample_id'],r['size'],r['variant'],r['mode']))
    import torch
    from adangel import _sm80 as native
    from adangel.trace.raw import validate_raw_trace
    from adangel.trace.prepare import _load_and_validate_raw
    from adangel.trace.storage import sha256_file
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    source_env=json.loads((args.input/'environment.json').read_text())
    assert sha256_file(Path(native.__file__))==source_env['extension_sha256']
    rawdir=ROOT/'data/raw/llama2_7b_prefill'
    manifest=validate_raw_trace(rawdir,ROOT/'configs/trace/llama2_7b_prefill.yaml',deep=False)
    assert sha256_file(rawdir/'trace_manifest.json')==json.loads((args.input/'data_provenance.json').read_text())['raw_manifest_sha256']
    index={r['sample_id']:r for r in manifest['samples']}
    args.output.mkdir(parents=True);rows=[]
    for sid,size in sorted({(x[0],x[1]) for x in targets}):
        entry=index[sid];raw=_load_and_validate_raw(rawdir/entry['file'],entry['layer'],entry['projection'])
        inputs=prepare(raw['activation_fp16'][:size,:size].contiguous().cuda(),raw['weight_fp16'][:size,:size].contiguous().cuda())
        refs={'o0':native.benchmark_o0(*inputs['o1'],'compute_only',0,1,2)['output'].clone()}
        for v in ('o5','o6'):refs[v]=call(native,v,'compute_only',inputs[v])['output'].clone()
        for _,_,v,mode in sorted(t for t in targets if t[:2]==(sid,size)):
            expected=call(native,v,'compute_only',inputs[v])['output'].clone()
            settings=source_env['settings']
            result=call(native,v,mode,inputs[v],settings['warmup'],settings['repeats'],settings['inner'])
            check_path(result,v,size)
            assert torch.equal(expected.view(torch.int32),result['output'].view(torch.int32))
            old=[r for r in original if (r['sample_id'],r['size'],r['variant'],r['mode'])==(sid,size,v,mode)]
            error=mse(result['output'],refs[REFERENCES[v]])
            assert error==old[0]['mse']
            rawtimes,native_total,method=aligned_timings(result,mode)
            stages={s:stats(x) for s,x in rawtimes.items()}
            primary='gemm' if mode=='compute_only' else 'total'
            before=st.median(r['stages'][primary]['median_ms'] for r in old)
            row=dict(sample_id=sid,size=size,variant=v,mode=mode,raw_ms=rawtimes,stages=stages,
                old_three_round_median_ms=before,latency_ratio_retry_to_original=stages[primary]['median_ms']/before,
                mse=error,reference=REFERENCES[v],kernel=dict(result['kernel']),total_timing=method,
                new_primary_cv=stages[primary]['cv_percent'])
            rows.append(row)
            with (args.output/'results.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(sid,size,'noise recheck complete',flush=True)
    summary=dict(passed=True,targets=len(targets),records=len(rows),original_tables_unchanged=True,
        selection='primary CV >=3% AND at least one observation >1.2*median; one repeat, no best-of selection',
        median_latency_ratio=st.median(r['latency_ratio_retry_to_original'] for r in rows) if rows else 1,
        max_latency_ratio=max((r['latency_ratio_retry_to_original'] for r in rows),default=1),
        min_latency_ratio=min((r['latency_ratio_retry_to_original'] for r in rows),default=1),
        still_primary_cv_failed=sum(r['new_primary_cv']>=3 for r in rows))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
