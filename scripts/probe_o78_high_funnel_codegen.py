#!/usr/bin/env python3
"""v117: one fixed high-x16 opcode-routing gate, not a scale/tile sweep.

Two-source SHF uses threadIdx.x as its low word. Its top four bits are zero
for every legal SM80 CTA, so (high<<4)|(tid>>28) exactly equals high*16.
No runtime claim is made before same-entry, resource and routing gates pass.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_eight_chain_schedule import instructions, trace
from inspect_o78_register_liveness import analyze
from probe_o78_eight_chain_codegen import generated_header as eight_header
from probe_roof_fullk_integer_codegen import static_entries

ROOT = Path(__file__).resolve().parents[1]
CONTROL = 'adangel_roof_o78_eight_chain_candidate'
SYMBOL = 'adangel_roof_o78_high_funnel_candidate'
OLD = '      o1_static_for<0,32>([&](auto i) { partial(i)*=16; });'
NEW = '''      // For tid<128, tid>>28=0. PTX SHF high-word semantics give
      // (partial_bits<<4)|(tid>>28), exactly the old bounded x16.
      // Nonzero lower word prevents trivial shl -> IMAD.SHL canonicalization.
      const unsigned funnel_low=threadIdx.x;
      o1_static_for<0,32>([&](auto i) {
        int shifted;
        asm("shf.l.wrap.b32 %0, %1, %2, 4;" : "=r"(shifted)
            : "r"(funnel_low), "r"(static_cast<unsigned>(partial(i))));
        partial(i)=shifted;
      });'''
LIMITS = dict(max_allocated_gpr=168, max_hot_local_instructions=0,
              max_static_work_ratio=1.01, min_fma_family_reduction=.10,
              required_high_shf=64, required_mma_each=32,
              required_ldsm=16, required_async_copy=10)


def generated_header(source):
    text = eight_header(source)
    if text.count(OLD) != 1:
        raise ValueError('fixed high reconstruction boundary drift')
    return text.replace(OLD, NEW).replace(
        'o78_eight_chain_experiment', 'o78_high_funnel_experiment')


def high_reconstruction(sass, symbol, live):
    """Attribute scalar shifts to second signed MMA -> first unsigned MMA.

    Counts only the 64 mathematical high components, never address shifts.
    This is a static def-use audit, not a timing or occupancy prediction.
    """
    schedule = trace(sass, symbol, live)
    events = {int(e['pc'], 16): e for e in schedule['schedule']}
    tags, shifts, consumed = {}, [], []
    for pc, text in instructions(sass, symbol, schedule['loop']):
        op = text.split()[0]
        dest = re.match(r'\S+\s+R(\d+)\s*,(.*)', text)
        if not dest:
            continue
        d, rest = int(dest[1]), dest[2]
        operands = [x.strip() for x in rest.split(',')]
        if pc in events:
            e = events[pc]
            if e['stage'] == 3:
                c = int(re.search(r'R(\d+)$', operands[-1])[1])
                expected = [(e['chain'], i, 2, True) for i in range(4)]
                if [tags.get(c+i) for i in range(4)] != expected:
                    raise ValueError('low MMA did not consume four correctly shifted high components')
                consumed.extend(expected)
            for i in range(4):
                tags[d+i] = (e['chain'], i, e['stage'], False)
            continue
        origins = [tags.get(int(r)) for r in re.findall(r'\bR(\d+)\b', rest)]
        high = [t for t in origins if t is not None and t[2] == 2]
        if high:
            if len(high) != 1 or high[0][3]:
                raise ValueError('unexpected high-component arithmetic: '+text)
            if op == 'IMAD.SHL.U32':
                valid = len(operands) == 3 and operands[1] == '0x10' and operands[2] == 'RZ'
            elif op == 'SHF.L.U32':
                valid = len(operands) == 3 and operands[1] == '0x4' and operands[2] == 'RZ'
            elif op == 'SHF.L.W.U32.HI':
                valid = len(operands) == 3 and operands[1] == '0x4'
            else:
                valid = False
            if not valid:
                raise ValueError('not the exact fixed high x16 route: '+text)
            tags[d] = (*high[0][:3], True)
            shifts.append(dict(pc=hex(pc), opcode=op, instruction=text,
                               chain=high[0][0], component=high[0][1]))
            continue
        width = (int(op.rsplit('.', 1)[1]) if op.startswith('LDSM.') else
                 4 if op.startswith(('LDS', 'LDG', 'LDL')) and '.128' in op else
                 2 if (op.startswith(('LDS', 'LDG', 'LDL')) and '.64' in op) or
                       op.startswith('IMAD.WIDE') or op == 'CS2R' else 1)
        for i in range(width):
            tags.pop(d+i, None)
    if len(shifts) != 64 or len(consumed) != 64:
        raise ValueError('incomplete high reconstruction audit')
    return dict(components=64, opcodes=dict(Counter(s['opcode'] for s in shifts)),
                shifts=shifts, low_mma_consumed_all=True,
                scope='static_high_def_use_not_runtime_pipeline_or_latency')


def cost_gate(old, new, routing):
    def count(loop, prefix):
        return sum(n for op, n in loop['opcode_counts'].items() if op.startswith(prefix))
    a = next(x for x in old['loops'] if x['kind'] == 'integer')
    b = next(x for x in new['loops'] if x['kind'] == 'integer')
    # In the integer loop these are integer IMAD variants only. FMA-family
    # routing is a hypothesis from NVIDIA's pipeline description, not proof
    # of SM80 cycles or predicted speedup. NCU/Event must decide any benefit.
    old_fma, new_fma = count(a, 'IMAD'), count(b, 'IMAD')
    reduction = 1 - new_fma / old_fma
    ratio = b['static_instructions'] / a['static_instructions']
    checks = dict(
        allocation=new['allocated_gpr'] <= LIMITS['max_allocated_gpr'],
        no_hot_local=count(b, 'LDL')+count(b, 'STL') == LIMITS['max_hot_local_instructions'],
        bounded_total_work=ratio <= LIMITS['max_static_work_ratio'],
        fma_routing_reduction=reduction >= LIMITS['min_fma_family_reduction'],
        all_high_on_shf=sum(n for op,n in routing['opcodes'].items() if op.startswith('SHF.')) == 64,
        native_math=count(b,'IMMA.16864.S4.S4') == 32 and count(b,'IMMA.16864.U4.S4') == 32,
        same_supply=count(b,'LDSM.') == 16 and count(b,'LDGSTS') == 10)
    return dict(passed=all(checks.values()), checks=checks, limits=LIMITS,
        old_static=a['static_instructions'], new_static=b['static_instructions'], static_work_ratio=ratio,
        old_imad_family=old_fma, new_imad_family=new_fma, imad_family_reduction=reduction,
        scope='predeclared_compile_potential_gate_not_performance_acceptance')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, default=Path('reports/o378_roof_v78_codegen'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        p.error('fresh repository output required')
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    prior = json.loads((a.baseline/'codegen.json').read_text())
    for path, value in prior['sources'].items():
        if sha(ROOT/path) != value:
            raise ValueError('v78 source drift: '+path)
    if sha(a.baseline/'o78_eight_chain.cubin') != prior['cubin_sha256']:
        raise ValueError('v78 cubin drift')
    cuda=Path('/usr/local/cuda-12.8/bin'); cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA/CUTLASS required')
    out.mkdir(parents=True)
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    (out/'o78_eight_chain_generated.cuh').write_text(eight_header(source))
    header=out/'o78_high_funnel_generated.cuh'; header.write_text(generated_header(source))
    sources=set(prior['sources'])|{
        'csrc/sm80/roof_o78_high_funnel_probe.cu', 'scripts/probe_o78_high_funnel_codegen.py',
        'scripts/inspect_eight_chain_schedule.py','scripts/compare_a100_codegen.py'}
    receipt=dict(scope='O7_O8_fixed_high_x16_compile_gate_not_GPU_acceptance',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},generated_header_sha256=sha(header),
        baseline_cubin_sha256=prior['cubin_sha256'],nvcc=version,cutlass_commit=commit,
        production_default_changed=False,cta_tile=[64,128,128],threads=128,stages=2,shared_bytes=34304,
        changed_semantics=False,supported_variants=['o7','o8'],limits=LIMITS,commands=[])
    def run(cmd,name):
        receipt['commands'].append(cmd)
        with (out/name).open('w') as f:
            subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
        '-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out)]
    src=str(ROOT/'csrc/sm80/roof_o78_high_funnel_probe.cu'); cubin=out/'o78_high_funnel.cubin'
    run(flags+[src,'-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+[src,'-ptx','-o',str(out/'o78_high_funnel.ptx')],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],'o78_high_funnel.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/'o78_high_funnel.sass').read_text()
    entries=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    live={s:analyze((out/'liveness.txt').read_text(),s) for s in (CONTROL,SYMBOL)}
    ptx=(out/'o78_high_funnel.ptx').read_text()
    for s in (CONTROL,SYMBOL):
        body=next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+s+'('))
        if not all(x in body for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32')):
            raise ValueError('same-entry PTX math/copy audit failed')
        if s==SYMBOL and 'shf.l.wrap.b32' not in body:
            raise ValueError('fixed funnel shift missing from PTX')
    routing={s:high_reconstruction(sass,s,live[s]) for s in (CONTROL,SYMBOL)}
    gate=cost_gate(live[CONTROL],live[SYMBOL],routing[SYMBOL])
    control=compare((a.baseline/'o78_eight_chain.sass').read_text(),sass,'^'+CONTROL+'$')
    gate['control_encoding_unchanged']=control['passed']
    gate['passed'] &= control['passed']
    receipt.update(entries=entries,liveness=live,high_reconstruction=routing,cost_gate=gate,
        cubin_sha256=sha(cubin),control_comparison=control,
        artifact_sha256={s:sha(out/s) for s in ('build.log','o78_high_funnel.sass',
            'o78_high_funnel.ptx','resources.txt','liveness.txt')})
    (out/'codegen.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if not control['passed'] or not all(e['native_u4_s4'] and e['native_s4_s4'] and
        not e['int8_mma'] and e['all_copies_bypass_l1'] for e in entries.values()):
        raise ValueError('same-entry INT4/control audit failed')
    print(json.dumps(gate,indent=2),flush=True)
    print('Gate '+('PASSED: GPU validation still required.' if gate['passed'] else
                  'FAILED: do not launch candidate or scan adjacent funnel variants.'),flush=True)


if __name__=='__main__': main()
