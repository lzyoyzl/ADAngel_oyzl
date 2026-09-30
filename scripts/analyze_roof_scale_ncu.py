#!/usr/bin/env python3
"""CPU comparison of scale-supply candidates. NCU is not formal Event timing."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re


def pc_stall_summary(source_rows):
    """Not-issued sampling evidence, not a cycle budget or producer attribution.

    NCU attributes samples to the consumer PC. Fixed-latency wait can involve
    an earlier instruction, so retain context without inferring its producer.
    Require every reason and reconcile each PC instead of zero-filling gaps.
    """
    reasons=('barrier','branch_resolving','dispatch','drain','imc','lg','long_sb',
             'math','membar','mio','misc','no_inst','not_selected','selected',
             'short_sb','sleeping','tex','wait')
    columns={r:f'stall_{r} (Not Issued)' for r in reasons}
    total_column='Warp Stall Sampling (Not-issued Samples)'
    required=set(columns.values())|{total_column,'Address','Source'}
    totals={r:0 for r in reasons}; by_opcode={}; pcs=[]; total=0
    def count(value):
        result=int(value.replace(',',''))
        if result<0: raise ValueError('negative PC sample count')
        return result
    for index,row in enumerate(source_rows):
        missing=required-set(row)
        if missing: raise ValueError(f'missing PC sampling columns: {sorted(missing)}')
        values={r:count(row[c]) for r,c in columns.items()}
        samples=count(row[total_column])
        if sum(values.values())!=samples:
            raise ValueError('PC sampling reason/total mismatch')
        match=re.match(r'\s*(?:@!?P(?:T|\d+)\s+)?([A-Z][A-Z0-9_]*)',row['Source'])
        if not match: raise ValueError('unknown PC opcode')
        opcode=match[1]
        work=by_opcode.setdefault(opcode,{r:0 for r in reasons})
        for r,value in values.items():
            totals[r]+=value;work[r]+=value
        total+=samples
        if values['wait']:
            pcs.append(dict(address=row['Address'],instruction=row['Source'].strip(),
                wait_samples=values['wait'],not_issued_samples=samples,
                preceding_instructions=[p['Source'].strip() for p in source_rows[max(0,index-3):index]]))
    if not source_rows: raise ValueError('empty PC sampling export')
    return dict(not_issued_samples=total,reason_samples=totals,
        reason_share_percent={r:100*v/total if total else None for r,v in totals.items()},
        samples_by_consumer_opcode={op:values for op,values in by_opcode.items() if any(values.values())},
        top_wait_consumer_pcs=sorted(pcs,key=lambda p:-p['wait_samples'])[:12],
        interpretation='PC samples attributed to waiting consumers, not causal producers; shares are not runtime fractions; separate profile sample counts are not normalized cycles')


def validate_arithmetic_work(counts, tune, fast):
    """Count executed warp instructions, not idealized tree arithmetic alone."""
    groups = 16777216  # 4096^2 outputs * 32 groups / 32 lanes.
    if any(counts.get(k) != groups for k in ('IMMA', 'I2F')):
        raise ValueError('expected 4096^3, G128, two-route native INT4 work')
    if tune in (34, 35, 36, 37, 38, 39, 40):
        # Tree variants separately round P*S. Predicated tree instructions may
        # execute even when their lanes do not contribute, so FADD is a lower
        # bound here; retain the observed count in the resource model.
        if (counts.get('FFMA', 0) != 0
                or counts.get('FMUL', 0) != groups * (1 if fast else 2)
                or counts.get('FADD', 0) < groups):
            raise ValueError('unexpected separate-product tree arithmetic')
    else:
        if counts.get('FFMA') != groups:
            raise ValueError('expected 4096^3, G128, two-route native INT4 work')
        if counts.get('FMUL', 0) != (0 if fast else groups):
            raise ValueError('unexpected scale math')
        if tune in (24,25,26,27) and counts.get('FADD') != 4096**2//32*(1 if tune in (24,26) else 3):
            raise ValueError('expected one/three final FP32 additions per output for two/four chains')


def analyze(raw_payload, sass_payload, tune, variant='o7', resource_model=False, pc_sampling=False):
    allowed={'o3':(6,13,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31),'o7':(6,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31),'o8':(6,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31)}
    if variant not in allowed or tune not in allowed[variant] + (34,35,36,37,38,39,40,41,42,45,46):
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
    work_by_opcode={}
    source_rows=list(csv.DictReader(src))
    if not source_rows:
        raise ValueError('empty SASS source export')
    missing_columns=set(work)-set(source_rows[0])
    # NCU omits the local-sector column if this function has no local
    # instructions. Never interpret arbitrary missing metrics as measured0.
    local_key='L2 Theoretical Sectors Local'
    no_local_opcodes=not any(re.match(r'\s*(?:@!?P(?:T|\d+)\s+)?(?:LDL|STL)(?:\.|\s)',row['Source']) for row in source_rows)
    if missing_columns and (missing_columns!={local_key} or not no_local_opcodes):
        raise ValueError(f'missing required memory counters: {sorted(missing_columns)}')
    for row in source_rows:
        match=re.match(r'\s*(?:@!?P(?:T|\d+)\s+)?([A-Z][A-Z0-9_]*)',row['Source'])
        if not match: raise ValueError('unknown opcode')
        count=int(row['Instructions Executed'].replace(',',''))
        counts[match[1]]=counts.get(match[1],0)+count
        opcode_work=work_by_opcode.setdefault(match[1],{key:0 for key in work})
        for key in work:
            value=0 if key in missing_columns else int(row[key].replace(',',''))
            work[key]+=value
            opcode_work[key]+=value
    def metric(key,unit=None):
        if unit is not None and units[key]!=unit: raise ValueError(f'unexpected unit for {key}')
        value=float(raw[key].replace(',',''))
        if not math.isfinite(value) or value<0: raise ValueError(f'invalid counter for {key}')
        return value
    def shared_bytes(key):
        factors={'byte/block':1,'Kbyte/block':1000,'Mbyte/block':1000000}
        if units[key] not in factors: raise ValueError(f'unexpected memory unit for {key}')
        return metric(key)*factors[units[key]]
    def dram_bytes_value(key):
        factors={'byte':1,'Kbyte':1000,'Mbyte':1000000,'Gbyte':1000000000}
        if units[key] not in factors: raise ValueError(f'unexpected DRAM unit for {key}')
        return metric(key)*factors[units[key]]
    def duration_ms():
        key='gpu__time_duration.sum'
        factors={'ns':1e-6,'us':1e-3,'ms':1,'s':1000}
        if units[key] not in factors: raise ValueError('unexpected duration unit')
        return metric(key)*factors[units[key]]
    if sum(counts.values())!=metric('smsp__inst_executed.sum','inst'):
        raise ValueError('raw/source dynamic instruction mismatch')
    validate_arithmetic_work(counts, tune, fast)
    cycles=metric('l1tex__cycles_elapsed.avg','cycle')
    fraction=metric('l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed','%')/100
    occupancy_limits={name:metric(f'launch__occupancy_limit_{name}','block')
                      for name in ('blocks','registers','shared_mem','warps')}
    model={}
    if resource_model:
        # Recompute the model when a candidate changes instruction/memory work.
        # These are overlapping necessary capacity constraints, NOT additive
        # timings, a latency-DAG solution or an achieved speedup prediction.
        sm=metric('device__attribute_multiprocessor_count')
        ipc=metric('device__attribute_max_ipc_per_multiprocessor')
        if sm!=108 or ipc!=4:
            raise ValueError('resource model requires the 108-SM A100')
        cycles_per_ms=1410000
        shared_count=metric('l1tex__data_pipe_lsu_wavefronts_mem_shared.sum')
        shared_fraction=metric('l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed','%')/100
        if cycles<=0 or shared_fraction<=0:
            raise ValueError('cannot cross-check shared data-pipe capacity')
        capacity=shared_count/(sm*cycles*shared_fraction)
        if not math.isclose(capacity,1,rel_tol=1e-5):
            raise ValueError('unexpected shared wavefront/SM/cycle capacity')
        active=metric('sm__cycles_active.avg','cycle')
        dram_bytes=dram_bytes_value('dram__bytes_read.sum')+dram_bytes_value('dram__bytes_write.sum')
        useful_ops=2*4096**3
        bounds=dict(
            mma=2*useful_ops/(sm*8192*cycles_per_ms),
            all_instruction_issue=sum(counts.values())/(sm*ipc*cycles_per_ms),
            i2f=counts['I2F']*32/(sm*16*cycles_per_ms),
            fp32_scale_and_accumulate_subset=(counts.get('FFMA',0)+counts.get('FMUL',0)+counts.get('FADD',0))*32/(sm*64*cycles_per_ms),
            l1tex_data_wavefront_capacity=cycles*fraction/cycles_per_ms,
            shared_wavefront_subset=shared_count/(sm*cycles_per_ms),
            dram_observed_bytes_at_spec_bw=dram_bytes/1.555e12*1000)
        for pipe in ('fma','alu','lsu'):
            bounds[pipe+'_pipe_counter_capacity']=active*metric(
                f'sm__inst_executed_pipe_{pipe}.avg.pct_of_peak_sustained_active','%')/100/cycles_per_ms
        floor=max(bounds.values())
        model=dict(
            resource_service_lower_bounds_ms_at_1410=bounds,
            binding_modeled_resources=[k for k,v in bounds.items() if math.isclose(v,floor,rel_tol=1e-7)],
            optimistic_fixed_work_lower_bound_ms=floor,
            optimistic_effective_ceiling_tops=useful_ops/floor/1e9,
            shared_capacity_crosscheck_wavefronts_per_sm_cycle=capacity,
            resource_model_note='fixed observed work, ideal overlap, 1410MHz; not achievable peak or Event timing',
        )
    return dict(variant=variant,tune=tune,kernel=identity[1],dynamic_instructions=sum(counts.values()),
        opcodes=counts,source_memory_work=work,
        source_memory_work_by_opcode={op:values for op,values in work_by_opcode.items() if any(values.values())},
        source_memory_work_omitted_zero_columns=sorted(missing_columns),
        source_memory_work_note='theoretical sectors/wavefront work; not actual HBM bytes',
        ncu_duration_ms=duration_ms(),
        # Older minimal captures did not request DRAM bytes. Missing is not0;
        # resource_model above still requires these counters explicitly.
        dram_read_bytes=dram_bytes_value('dram__bytes_read.sum') if 'dram__bytes_read.sum' in units else None,
        dram_write_bytes=dram_bytes_value('dram__bytes_write.sum') if 'dram__bytes_write.sum' in units else None,
        eligible_warps=metric('smsp__warps_eligible.avg.per_cycle_active'),
        issue_active_percent=metric('smsp__issue_active.avg.pct_of_peak_sustained_active','%'),
        registers_per_thread=metric('launch__registers_per_thread','register/thread'),
        registers_per_thread_allocated=metric('launch__registers_per_thread_allocated','register/thread'),
        dynamic_shared_bytes=shared_bytes('launch__shared_mem_per_block_dynamic'),
        allocated_shared_bytes_including_driver=shared_bytes('launch__shared_mem_per_block_allocated'),
        occupancy_cta_limits=occupancy_limits,
        max_ctas_per_sm_from_launch_limits=min(occupancy_limits.values()),
        **model,
        **({'pc_sampling':pc_stall_summary(source_rows)} if pc_sampling else {}),
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
    p.add_argument('--resource-model',action='store_true',
                   help='require additional raw metrics and recompute all fixed-work capacity constraints')
    p.add_argument('--pc-sampling',action='store_true',
                   help='strictly reconcile not-issued PC samples and expose wait consumers with SASS context')
    args=p.parse_args()
    rows=[];sources=[]
    for tune in args.tunes:
        payloads=[]
        for suffix in ('raw','source_sass'):
            path=args.directory/f'ncu_{args.variant}_t{tune}_{suffix}.csv'
            b=path.read_bytes();payloads.append(b.decode('utf-8-sig'))
            sources.append(dict(file=str(path),sha256=hashlib.sha256(b).hexdigest()))
        rows.append(analyze(*payloads,tune,args.variant,args.resource_model,args.pc_sampling))
    print(json.dumps(dict(scope='same_binary_NCU_diagnostic_not_Event_acceptance',
        sources=sources,rows=rows),indent=2,allow_nan=False))


if __name__=='__main__': main()
