#!/usr/bin/env python3
"""Attribute current O3 best NCU samples to verified SASS consumer roles.

Samples identify stalled consumers, not producer latency or speedup potential.
The original v133 O7/O8 parser/evidence are left immutable.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re

from analyze_o78_mma_phase_stalls import bare, normalized
from inspect_eight_chain_schedule import trace
from inspect_o78_register_liveness import analyze
from profile_o3_best_warm import ROOT, SYMBOL, SAMPLE

CODEGEN = ROOT / 'docs/evidence/a100_o378_roof_v89/reports/o378_roof_v89_o3_codegen'


def normalized_o3(ins):
    # NCU relocates BSSY's convergence destination as well as BRA/CALL.
    # Keep the barrier number, opcode and predicate; normalize only its target.
    if bare(ins).split()[0] == 'BSSY':
        ins = re.sub(r'0x[0-9a-fA-F]+', 'TARGET', ins)
    return normalized(ins)


def parse_entry(sass):
    block = next((b for b in re.split(r'(?=\s*Function\s*:\s*)', sass)
        if re.match(r'\s*Function\s*:\s*'+re.escape(SYMBOL)+r'\s', b)), None)
    if block is None:
        raise ValueError('missing exact current-best O3 entry')
    result = {}
    for line in block.splitlines():
        m = re.search(r'/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*;', line)
        if m: result[int(m[1], 16)] = m[2].strip()
    return result


def classify(sass, live):
    schedule = trace(sass, SYMBOL, live)
    stages = {int(s['pc'], 16): s['stage'] for s in schedule['schedule']}
    loop = next(x for x in live['loops'] if x['kind'] == 'integer')
    begin, end = int(loop['begin_pc'], 16), int(loop['end_pc'], 16)
    tags, categories, details = {}, {}, {}
    for pc, raw in parse_entry(sass).items():
        if not begin <= pc <= end: continue
        ins = bare(raw); op = ins.split()[0]
        cat = 'other_loop_address_or_control'; tag = None; width = 1
        dest = re.match(r'\S+\s+(R\d+)\s*,(.*)', ins)
        src = [tags.get(r) for r in re.findall(r'\bR\d+\b', dest[2])] if dest else []
        if pc in stages:
            stage = stages[pc]
            cat = ('mma_high_k0', 'mma_high_k1', 'mma_low_k0', 'mma_low_k1')[stage-1]
            tag = 'partial_high' if stage <= 2 else 'partial_low'; width = 4
        elif op.startswith('LDSM'):
            cat = 'matrix_fragment_load'; tag = 'matrix'; width = int(op.rsplit('.', 1)[1])
        elif op.startswith('LDGSTS'):
            cat = 'next_stage_async_copy'
        elif op.startswith('LDS'):
            if dest:
                offset = re.search(r'\+0x([0-9a-f]+)\]', ins)
                # Frozen O3 Storage: three A-low/high/W stages total0xc000 bytes;
                # the stage index is in the base register, the column in the immediate.
                if offset is None or not 0xc000 <= int(offset[1], 16) < 0xc200:
                    raise ValueError('unknown factor address: '+ins)
                tag = 'weight_factor'; cat = 'factor_shared_load'
                width = 4 if '.128' in op else 2 if '.64' in op else 1
            else: cat = 'compiler_dummy_shared_read'
        elif 'partial_high' in src and op.startswith(('IMAD', 'SHF')):
            if (op.startswith('IMAD') and ', 0x10, RZ' in ins) or (
                    op.startswith('SHF.L') and ', 0x4, RZ' in ins):
                cat = 'high_times_16'; tag = 'partial_high'
            else: raise ValueError('unhandled high transformation: '+ins)
        elif op == 'IMAD' and 'partial_low' in src and 'weight_factor' in src:
            if not dest or not ins.endswith(', '+dest[1]):
                raise ValueError('not in-place weighted accumulation: '+ins)
            cat = 'weighted_integer_accumulate'; tag = 'accumulator'
        elif op.startswith(('BAR', 'DEPBAR', 'LDGDEPBAR')):
            cat = 'stage_synchronization'
        elif op.startswith(('BRA', 'CALL')):
            cat = 'loop_exit_control'
        elif op == 'MOV' and len(src) == 1:
            tag = src[0]
        if dest:
            first = int(dest[1][1:])
            if op.startswith('IMAD.WIDE') or op == 'CS2R': width = 2
            for r in range(first, first+width): tags['R'+str(r)] = tag
        categories[pc] = cat; details[pc] = dict(instruction=raw, category=cat)
    counts = Counter(categories.values())
    required = dict(mma_high_k0=16, mma_high_k1=16, mma_low_k0=16, mma_low_k1=16,
        high_times_16=64, weighted_integer_accumulate=64, matrix_fragment_load=16,
        factor_shared_load=8, next_stage_async_copy=9)
    for key, count in required.items():
        if counts[key] != count: raise ValueError(f'incomplete {key}: {counts[key]} != {count}')
    if len(categories) != 323: raise ValueError('unexpected O3 loop footprint')
    return categories, details


def analyze_capture(sass, liveness, source, prior):
    categories, details = classify(sass, analyze(liveness, SYMBOL))
    if next(csv.reader(io.StringIO(source)))[1] != SYMBOL:
        raise ValueError('capture symbol mismatch')
    rows = list(csv.DictReader(io.StringIO(source.split('\n', 1)[1])))
    decoded = parse_entry(sass); base = min(int(r['Address'], 16) for r in rows)
    if len(rows) != len(decoded): raise ValueError('entry size mismatch')
    result = defaultdict(lambda: dict(static_instructions=0, warp_instructions=0,
        predicated_on_thread_instructions=0, not_issued_samples=0, reason_samples=Counter()))
    number = lambda v: int((v or '0').replace(',', ''))
    pcs, seen = [], set()
    for r in rows:
        pc = int(r['Address'], 16)-base
        if pc in seen or pc not in decoded or normalized_o3(decoded[pc]) != normalized_o3(r['Source']):
            raise ValueError('NCU instruction mismatch at '+hex(pc))
        seen.add(pc)
        cat = categories.get(pc, 'outside_integer_loop'); summary = result[cat]
        summary['static_instructions'] += 1
        summary['warp_instructions'] += number(r['Instructions Executed'])
        summary['predicated_on_thread_instructions'] += number(r['Predicated-On Thread Instructions Executed'])
        count = number(r['Warp Stall Sampling (Not-issued Samples)'])
        reasons = {k[6:-13]: number(v) for k, v in r.items() if k.startswith('stall_') and k.endswith(' (Not Issued)')}
        if sum(reasons.values()) != count: raise ValueError('per-PC sampling closure failed')
        summary['not_issued_samples'] += count; summary['reason_samples'].update(reasons)
        if count: pcs.append(dict(pc=hex(pc), category=cat, instruction=r['Source'].strip(),
            not_issued_samples=count, reason_samples=reasons))
    totals = Counter()
    for row in result.values(): totals.update(row['reason_samples'])
    if sum(r['warp_instructions'] for r in result.values()) != prior['dynamic_instructions']:
        raise ValueError('dynamic instruction closure failed')
    if totals != Counter(prior['pc_sampling']['reason_samples']):
        raise ValueError('sampling total differs from capture')
    return dict(scope='current_O3_verified_consumer_roles_not_causal_latency', variant='o3',
        sample_id=prior.get('sample_id', 'unspecified'), kernel=SYMBOL,
        static_entry_operands_and_predicates_verified=True, integer_loop_static=Counter(categories.values()),
        consumer_categories=dict(result), not_issued_samples=sum(totals.values()), reason_samples=totals,
        top_pcs=sorted(pcs, key=lambda row: row['not_issued_samples'], reverse=True)[:30],
        instruction_roles={hex(pc): value for pc, value in details.items()},
        new_performance_result=False, production_default_changed=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT):
        p.error('fresh repository output directory required')
    files = dict(sass=CODEGEN/'o3_grouped_cta.sass', liveness=CODEGEN/'liveness.txt',
        source=a.capture/'o3_warm_source_sass.csv', prior=a.capture/'analysis.json')
    prior = json.loads(files['prior'].read_text())
    if (prior.get('sample_id') != SAMPLE or prior.get('cache_control') != 'none' or
            prior.get('replay_mode') != 'application'):
        raise ValueError('expected warm representative capture')
    result = analyze_capture(files['sass'].read_text(), files['liveness'].read_text(),
                             files['source'].read_text(), prior)
    result['inputs'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files.values()}
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.mkdir(parents=True)
    (a.output/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(static=result['integer_loop_static'], categories=result['consumer_categories']), indent=2))


if __name__ == '__main__':
    main()
