#!/usr/bin/env python3
"""Join an existing NCU capture to exact MMA-chain/scale consumers.

This is a new analysis of v114, NOT a new GPU capture, performance run, or
causal latency decomposition. Register provenance labels consumers; it does
not assert that their preceding producer is the cause of a sampled stall.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re

from inspect_eight_chain_schedule import trace
from inspect_o78_register_liveness import analyze

ROOT=Path(__file__).resolve().parents[1]
SYMBOL='adangel_roof_o78_eight_chain_candidate'
CODEGEN=ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen'
CAPTURE=ROOT/'docs/evidence/a100_o378_roof_v114/reports/o378_roof_v114_o8_warm_ncu'


def parse_entry(sass):
    block=next((b for b in re.split(r'(?=\s*Function\s*:\s*)',sass)
                if re.match(r'\s*Function\s*:\s*'+re.escape(SYMBOL)+r'\s',b)),None)
    if block is None:raise ValueError('missing exact entry')
    result={}
    for line in block.splitlines():
        m=re.search(r'/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*;',line)
        if m:result[int(m[1],16)]=m[2].strip()
    return result


def bare(ins):
    return re.sub(r'^@!?U?P\w+\s+','',ins).strip()


def normalized(ins):
    predicate=re.match(r'^\s*(@!?U?P\w+)\s+',ins)
    pred=predicate[1] if predicate else ''
    ins=bare(ins.strip()).rstrip(';').replace('.reuse','')
    # Disassembly uses relative destinations; NCU may print relocated ones.
    if ins.split()[0] in ('BRA','CALL.REL.NOINC'):
        ins=re.sub(r'0x[0-9a-fA-F]+','TARGET',ins)
    # NCU omits commas between consecutive memory operands.
    return pred+re.sub(r'\s+','',ins).replace('],[','][')


def classify(sass,live):
    schedule=trace(sass,SYMBOL,live)
    stages={int(s['pc'],16):s['stage'] for s in schedule['schedule']}
    loop=next(x for x in live['loops'] if x['kind']=='integer')
    begin,end=int(loop['begin_pc'],16),int(loop['end_pc'],16)
    tags={};categories={};details={}
    for pc,raw in parse_entry(sass).items():
        if not begin<=pc<=end:continue
        ins=bare(raw);op=ins.split()[0]
        cat='other_loop_address_or_control'
        output_tag=None;width=1
        dest=re.match(r'\S+\s+(R\d+)\s*,(.*)',ins)
        source_tags=[tags.get(r) for r in re.findall(r'\bR\d+\b',dest[2])] if dest else []
        if pc in stages:
            stage=stages[pc]
            cat=('mma_high_k0','mma_high_k1','mma_low_k0','mma_low_k1')[stage-1]
            output_tag='partial_high' if stage<=2 else 'partial_low'
            width=4
        elif op.startswith('LDSM'):
            cat='matrix_fragment_load';width=int(op.rsplit('.',1)[1]);output_tag='matrix'
        elif op.startswith('LDGSTS'):
            cat='next_stage_async_copy'
        elif op.startswith('LDS'):
            if dest:
                # Frozen Storage: Af starts at0, Wf starts at0x8200.
                immediate=re.search(r'\+0x([0-9a-f]+)\]',ins)
                offset=int(immediate[1],16) if immediate else 0
                if 0x8200<=offset<0x8400:output_tag='weight_factor'
                elif offset<0x100:output_tag='activation_factor'
                else:raise ValueError('unrecognized factor shared address: '+ins)
                cat='factor_shared_load'
                width=2 if '.64' in op else 4 if '.128' in op else 1
            else:cat='compiler_dummy_shared_read'
        elif 'partial_high' in source_tags and op.startswith(('IMAD','SHF')):
            # The merged chain contains exactly high*16 between S4 and U4.
            if (op.startswith('IMAD') and ', 0x10, RZ' in ins) or (
                op.startswith('SHF.L') and ', 0x4, RZ' in ins):
                cat='high_times_16';output_tag='partial_high'
            else:raise ValueError('unhandled high partial transformation: '+ins)
        elif op=='IMAD' and set(source_tags)=={'activation_factor','weight_factor'} and ins.endswith(', RZ'):
            cat='factor_product';output_tag='coefficient'
        elif op=='IMAD' and set(source_tags)=={'activation_factor','partial_low'} and ins.endswith(', RZ'):
            cat='partial_times_activation_factor';output_tag='scaled_partial'
        elif op=='IMAD' and 'partial_low' in source_tags and 'coefficient' in source_tags:
            if not dest or not ins.endswith(', '+dest[1]):raise ValueError('not in-place weighted accumulation')
            cat='weighted_integer_accumulate';output_tag='accumulator'
        elif op=='IMAD' and 'scaled_partial' in source_tags and 'weight_factor' in source_tags:
            if not dest or not ins.endswith(', '+dest[1]):raise ValueError('not in-place second-factor accumulation')
            cat='weighted_integer_accumulate';output_tag='accumulator'
        elif op.startswith(('BAR','DEPBAR','LDGDEPBAR')):
            cat='stage_synchronization'
        elif op.startswith(('BRA','CALL')):
            cat='loop_exit_control'
        elif op=='MOV' and len(source_tags)==1:
            output_tag=source_tags[0]
        if dest:
            first=int(dest[1][1:])
            if op.startswith('IMAD.WIDE') or op=='CS2R':width=2
            for r in range(first,first+width):tags['R'+str(r)]=output_tag
        categories[pc]=cat
        details[pc]=dict(instruction=raw,category=cat)
    counts=Counter(categories.values())
    required=dict(mma_high_k0=16,mma_high_k1=16,mma_low_k0=16,mma_low_k1=16,
        high_times_16=64,weighted_integer_accumulate=64,
        matrix_fragment_load=16,factor_shared_load=12,next_stage_async_copy=10)
    for name,count in required.items():
        if counts[name]!=count:raise ValueError(f'provenance incomplete: {name}={counts[name]} expected {count}')
    if counts['factor_product']+counts['partial_times_activation_factor']!=64:
        raise ValueError('coefficient/partial-factor multiplication accounting failed: '+str(counts))
    if len(categories)!=383:raise ValueError('unexpected loop footprint')
    return categories,details


def analyze_capture(sass,liveness,source,prior):
    live=analyze(liveness,SYMBOL)
    categories,details=classify(sass,live)
    if next(csv.reader(io.StringIO(source)))[1]!=SYMBOL:raise ValueError('capture symbol mismatch')
    rows=list(csv.DictReader(io.StringIO(source.split('\n',1)[1])))
    decoded=parse_entry(sass);base=min(int(r['Address'],16) for r in rows)
    if len(rows)!=len(decoded):raise ValueError('entry size mismatch')
    result=defaultdict(lambda:dict(static_instructions=0,warp_instructions=0,
        predicated_on_thread_instructions=0,not_issued_samples=0,reason_samples=Counter()))
    pcs=[];seen=set()
    number=lambda v:int((v or '0').replace(',',''))
    for r in rows:
        pc=int(r['Address'],16)-base
        if pc not in decoded or pc in seen or normalized(decoded[pc])!=normalized(r['Source']):
            raise ValueError('NCU instruction mismatch at '+hex(pc)+': '+r['Source'])
        seen.add(pc)
        cat=categories.get(pc,'outside_integer_loop')
        row=result[cat];row['static_instructions']+=1
        row['warp_instructions']+=number(r['Instructions Executed'])
        row['predicated_on_thread_instructions']+=number(r['Predicated-On Thread Instructions Executed'])
        count=number(r['Warp Stall Sampling (Not-issued Samples)']);row['not_issued_samples']+=count
        reasons={k[6:-13]:number(v) for k,v in r.items() if k.startswith('stall_') and k.endswith(' (Not Issued)')}
        if sum(reasons.values())!=count:raise ValueError('per-PC sampling closure failed')
        row['reason_samples'].update(reasons)
        if count:pcs.append(dict(pc=hex(pc),category=cat,instruction=r['Source'].strip(),
            not_issued_samples=count,reason_samples=reasons))
    totals=Counter()
    for row in result.values():totals.update(row['reason_samples'])
    if sum(r['warp_instructions'] for r in result.values())!=prior['dynamic_instructions']:
        raise ValueError('dynamic instruction total differs from original analysis')
    expected=Counter(prior['pc_sampling']['reason_samples'])
    if totals!=expected:raise ValueError('sampling totals differ from original analysis')
    calls=[]
    for r in rows:
        if bare(r['Source']).startswith('CALL'):
            calls.append(dict(pc=hex(int(r['Address'],16)-base),instruction=r['Source'].strip(),
                warp_instructions=number(r['Instructions Executed']),
                predicated_on_thread_instructions=number(r['Predicated-On Thread Instructions Executed'])))
    return dict(scope='existing_v114_capture_new_consumer_phase_attribution_not_causal_time_breakdown',
        variant='o8',sample_id='layer_12_o_proj',shape=[4096,4096,4096],kernel=SYMBOL,
        static_entry_operands_and_predicates_verified=True,integer_loop_static=Counter(categories.values()),
        consumer_categories=dict(result),not_issued_samples=sum(totals.values()),reason_samples=totals,
        top_pcs=sorted(pcs,key=lambda r:r['not_issued_samples'],reverse=True)[:30],calls=calls,
        instruction_roles={hex(pc):value for pc,value in details.items()},
        new_GPU_capture=False,new_performance_or_MSE_result=False,production_default_changed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output directory required')
    sources=dict(sass=CODEGEN/'o78_eight_chain.sass',liveness=CODEGEN/'liveness.txt',
        source=CAPTURE/'o8_warm_source_sass.csv',prior=CAPTURE/'analysis.json')
    r=analyze_capture(sources['sass'].read_text(),sources['liveness'].read_text(),
        sources['source'].read_text(),json.loads(sources['prior'].read_text()))
    r['inputs']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources.values()}
    r['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out.mkdir(parents=True)
    (out/'analysis.json').write_text(json.dumps(r,indent=2)+'\n')
    summary={name:dict(static=v['static_instructions'],dynamic=v['warp_instructions'],
        samples=v['not_issued_samples'],wait=v['reason_samples']['wait'],
        math=v['reason_samples']['math'],short_sb=v['reason_samples']['short_sb'])
        for name,v in r['consumer_categories'].items()}
    print(json.dumps(dict(scope=r['scope'],categories=summary,calls=r['calls']),indent=2))


if __name__=='__main__':main()
