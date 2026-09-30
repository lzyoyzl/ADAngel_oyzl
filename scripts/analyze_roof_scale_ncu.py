#!/usr/bin/env python3
"""CPU comparison of guarded scale candidates. NCU is not formal Event timing."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re


def analyze(raw_payload, sass_payload, tune):
    raw_rows=list(csv.DictReader(io.StringIO(raw_payload)))
    if len(raw_rows)!=2:
        raise ValueError('expected units row and one kernel')
    units,raw=raw_rows
    src=io.StringIO(sass_payload)
    identity=next(csv.reader(src))
    expected=f'adangel_sm80_roof_candidate<1,{int(tune>=11)},{tune}>'
    def normalized(symbol):
        return re.sub(r'\((?:bool|int)\)|\s+','',symbol)
    if identity[0]!='Kernel Name' or expected not in normalized(identity[1]) or expected not in normalized(raw['Kernel Name']):
        raise ValueError('unexpected candidate identity')
    counts={}
    for row in csv.DictReader(src):
        match=re.match(r'\s*(?:@!?P(?:T|\d+)\s+)?([A-Z][A-Z0-9_]*)',row['Source'])
        if not match: raise ValueError('unknown opcode')
        count=int(row['Instructions Executed'].replace(',',''))
        counts[match[1]]=counts.get(match[1],0)+count
    def metric(key,unit=None):
        if unit is not None and units[key]!=unit: raise ValueError(f'unexpected unit for {key}')
        return float(raw[key].replace(',',''))
    if sum(counts.values())!=metric('smsp__inst_executed.sum','inst'):
        raise ValueError('raw/source dynamic instruction mismatch')
    if any(counts.get(k)!=16777216 for k in ('IMMA','I2F','FFMA')):
        raise ValueError('expected 4096^3, G128, two-route native INT4 work')
    if counts.get('FMUL',0)!=(16777216 if tune==6 else 0):
        raise ValueError('unexpected scale math')
    cycles=metric('l1tex__cycles_elapsed.avg','cycle')
    fraction=metric('l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed','%')/100
    return dict(tune=tune,kernel=identity[1],dynamic_instructions=sum(counts.values()),
        opcodes=counts,ncu_duration_ms=metric('gpu__time_duration.sum','us')/1000,
        eligible_warps=metric('smsp__warps_eligible.avg.per_cycle_active'),
        issue_active_percent=metric('smsp__issue_active.avg.pct_of_peak_sustained_active','%'),
        l1_service_ms_at_1410=cycles*fraction/1410000,
        l1_bound_note='profile-conditioned capacity requirement only, not attainable kernel time',
        math_pipe_stall_per_issue=metric('smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio'),
        mio_stall_per_issue=metric('smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--tunes',type=int,nargs='+',default=[6,11])
    args=p.parse_args()
    rows=[];sources=[]
    for tune in args.tunes:
        payloads=[]
        for suffix in ('raw','source_sass'):
            path=args.directory/f'ncu_o7_t{tune}_{suffix}.csv'
            b=path.read_bytes();payloads.append(b.decode('utf-8-sig'))
            sources.append(dict(file=str(path),sha256=hashlib.sha256(b).hexdigest()))
        rows.append(analyze(*payloads,tune))
    print(json.dumps(dict(scope='same_binary_NCU_diagnostic_not_Event_acceptance',
        sources=sources,rows=rows),indent=2,allow_nan=False))


if __name__=='__main__': main()
