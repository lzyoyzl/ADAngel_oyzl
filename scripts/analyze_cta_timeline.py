#!/usr/bin/env python3
"""Approximate CTA intervals from all four warps; not exact scheduler residency."""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(records,sm_count=108,capacity=3):
    a=np.asarray(records,dtype=np.uint64)
    if a.ndim!=3 or a.shape[1:]!=(4,4) or len(a)==0 or sm_count<1 or capacity<1:
        raise ValueError('CTA x four warps x [start,end,start_smid,end_smid] required')
    if np.any(a[:,:,0]==0) or np.any(a[:,:,1]<=a[:,:,0]):raise ValueError('missing/invalid warp interval')
    if np.any(a[:,:,2]!=a[:,:,3]) or np.any(a[:,:,2]!=a[:,0:1,2]):
        raise ValueError('CTA SM changed during capture')
    # Difference BEFORE float conversion, retaining 64-bit timer precision.
    origin=int(a[:,:,0].min());start=(a[:,:,0].min(1)-origin).astype(np.int64)
    end=(a[:,:,1].max(1)-origin).astype(np.int64);sm=a[:,0,2].astype(np.int64)
    unique=np.unique(sm)
    if len(unique)>sm_count:raise ValueError('too many SM identifiers')
    span=int(end.max());last_start=int(start.max());tail=span-last_start
    hist=np.zeros(capacity+1,dtype=np.int64);tail_hist=hist.copy();durations=[]
    max_overlap=0;per_sm=[]
    for sid in unique:
        mask=sm==sid
        events={0:0,span:0,last_start:0}
        for s,e in zip(start[mask],end[mask]):
            events[int(s)]=events.get(int(s),0)+1;events[int(e)]=events.get(int(e),0)-1
        active=0;before=0
        for t,delta in sorted(events.items()):
            if active<0 or active>capacity:raise ValueError('observed overlap exceeds compiled capacity')
            hist[active]+=t-before
            tail_hist[active]+=max(0,t-max(before,last_start))
            active+=delta;max_overlap=max(max_overlap,active);before=t
        if active!=0:raise ValueError('unbalanced intervals')
        d=end[mask]-start[mask];durations.extend(d.tolist())
        per_sm.append(dict(smid=int(sid),ctas=int(mask.sum()),last_end_ms=int(end[mask].max())/1e6,
                           median_CTA_warp_envelope_ms=float(np.median(d))/1e6))
    hist[0]+=(sm_count-len(unique))*span;tail_hist[0]+=(sm_count-len(unique))*tail
    idle=lambda h:sum((capacity-i)*int(v) for i,v in enumerate(h))/(capacity*sm_count*1e6)
    return dict(ctas=len(a),observed_SMs=len(unique),reported_SM_count=sm_count,
        globaltimer_span_ms=span/1e6,last_observed_CTA_start_ms=last_start/1e6,
        after_last_start_ms=tail/1e6,observed_max_CTA_overlap_per_SM=max_overlap,
        capacity_time_fraction={str(i):int(v)/(span*sm_count) for i,v in enumerate(hist)},
        tail_capacity_time_fraction={str(i):int(v)/(tail*sm_count) for i,v in enumerate(tail_hist)},
        mean_CTA_warp_envelope_ms=float(np.mean(durations))/1e6,
        median_CTA_warp_envelope_ms=float(np.median(durations))/1e6,
        full_capture_missing_capacity_equivalent_ms=idle(hist),
        tail_missing_capacity_equivalent_ms=idle(tail_hist),per_SM=per_sm,
        scope='observed_warp_envelopes_underestimate_true_residency; instrumentation_perturbs_execution',
        not_an_achievable_latency_bound=True,not_a_new_performance_result=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('fresh output required')
    result=analyze(np.load(a.input,allow_pickle=False))
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_SM'},indent=2))


if __name__=='__main__':main()
