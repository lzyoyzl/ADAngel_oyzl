#!/usr/bin/env python3
"""CPU comparison of scale-supply candidates. NCU is not formal Event timing."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re


def analyze(raw_payload, sass_payload, tune, variant='o7'):
    allowed={'o3':(6,13,16,17,18,19),'o7':(6,11,12,14,15,16,17,18,19),'o8':(6,14,15,16,17,18,19)}
    if variant not in allowed or tune not in allowed[variant]:
        raise ValueError('unsupported variant/tune profiling pair')
    raw_rows=list(csv.DictReader(io.StringIO(raw_payload)))
    if len(raw_rows)!=2:
        raise ValueError('expected units row and one kernel')
    units,raw=raw_rows
    src=io.StringIO(sass_payload)
    identity=next(csv.reader(src))
    # The archived O3 random-input profile uses the guarded exponent path.
    # Power2 substitutions apply ONLY to O7 tunes11/12, never async tune14.
    fast=variant=='o3' or tune in (11,12)
    expected=f'adangel_sm80_roof_candidate<{int(variant!="o3")},{int(fast)},{tune}>'
    def normalized(symbol):
        return re.sub(r'\((?:bool|int)\)|\s+','',symbol)
    if identity[0]!='Kernel Name' or expected not in normalized(identity[1]) or expected not in normalized(raw['Kernel Name']):
        raise ValueError('unexpected candidate identity')
    counts={}
    work={key:0 for key in ('L1 Wavefronts Shared Excessive','L1 Wavefronts Shared',
                           'L2 Theoretical Sectors Global Excessive','L2 Theoretical Sectors Local')}
    for row in csv.DictReader(src):
        match=re.match(r'\s*(?:@!?P(?:T|\d+)\s+)?([A-Z][A-Z0-9_]*)',row['Source'])
        if not match: raise ValueError('unknown opcode')
        count=int(row['Instructions Executed'].replace(',',''))
        counts[match[1]]=counts.get(match[1],0)+count
        for key in work:
            work[key]+=int(row[key].replace(',',''))
    def metric(key,unit=None):
        if unit is not None and units[key]!=unit: raise ValueError(f'unexpected unit for {key}')
        return float(raw[key].replace(',',''))
    def shared_bytes(key):
        factors={'byte/block':1,'Kbyte/block':1000,'Mbyte/block':1000000}
        if units[key] not in factors: raise ValueError(f'unexpected memory unit for {key}')
        return metric(key)*factors[units[key]]
    if sum(counts.values())!=metric('smsp__inst_executed.sum','inst'):
        raise ValueError('raw/source dynamic instruction mismatch')
    if any(counts.get(k)!=16777216 for k in ('IMMA','I2F','FFMA')):
        raise ValueError('expected 4096^3, G128, two-route native INT4 work')
    if counts.get('FMUL',0)!=(0 if fast else 16777216):
        raise ValueError('unexpected scale math')
    cycles=metric('l1tex__cycles_elapsed.avg','cycle')
    fraction=metric('l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed','%')/100
    occupancy_limits={name:metric(f'launch__occupancy_limit_{name}','block')
                      for name in ('blocks','registers','shared_mem','warps')}
    return dict(variant=variant,tune=tune,kernel=identity[1],dynamic_instructions=sum(counts.values()),
        opcodes=counts,source_memory_work=work,
        source_memory_work_note='theoretical sectors/wavefront work; not actual HBM bytes',
        ncu_duration_ms=metric('gpu__time_duration.sum','us')/1000,
        eligible_warps=metric('smsp__warps_eligible.avg.per_cycle_active'),
        issue_active_percent=metric('smsp__issue_active.avg.pct_of_peak_sustained_active','%'),
        registers_per_thread=metric('launch__registers_per_thread','register/thread'),
        registers_per_thread_allocated=metric('launch__registers_per_thread_allocated','register/thread'),
        dynamic_shared_bytes=shared_bytes('launch__shared_mem_per_block_dynamic'),
        allocated_shared_bytes_including_driver=shared_bytes('launch__shared_mem_per_block_allocated'),
        occupancy_cta_limits=occupancy_limits,
        max_ctas_per_sm_from_launch_limits=min(occupancy_limits.values()),
        achieved_occupancy_percent=metric('sm__warps_active.avg.pct_of_peak_sustained_active','%'),
        l1_data_pipe_peak_percent=fraction*100,
        l1_service_ms_at_1410=cycles*fraction/1410000,
        l1_bound_note='profile-conditioned capacity requirement only, not attainable kernel time',
        long_scoreboard_stall_per_issue=metric('smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio'),
        math_pipe_stall_per_issue=metric('smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio'),
        mio_stall_per_issue=metric('smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--tunes',type=int,nargs='+',default=[6,11])
    p.add_argument('--variant',choices=['o3','o7','o8'],default='o7')
    args=p.parse_args()
    rows=[];sources=[]
    for tune in args.tunes:
        payloads=[]
        for suffix in ('raw','source_sass'):
            path=args.directory/f'ncu_{args.variant}_t{tune}_{suffix}.csv'
            b=path.read_bytes();payloads.append(b.decode('utf-8-sig'))
            sources.append(dict(file=str(path),sha256=hashlib.sha256(b).hexdigest()))
        rows.append(analyze(*payloads,tune,args.variant))
    print(json.dumps(dict(scope='same_binary_NCU_diagnostic_not_Event_acceptance',
        sources=sources,rows=rows),indent=2,allow_nan=False))


if __name__=='__main__': main()
