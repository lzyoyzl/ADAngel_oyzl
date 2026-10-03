#!/usr/bin/env python3
"""v87: shift the O7 coefficient, NOT the partial; preserve the final MAD.

Unlike v71, no separate shift/add follows partial multiplication. Unlike
v72, coefficient formation uses the integer shift pipe, not another IMAD.
Compile/coordinate/arithmetic gate first, never a production default change.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_o78_eight_chain_codegen import generated_header as eight_header
from probe_roof_fullk_integer_codegen import static_entries

ROOT = Path(__file__).resolve().parents[1]
CONTROL = 'adangel_roof_o78_eight_chain_candidate'
SYMBOL = 'adangel_roof_o7_shift_coefficient_candidate'
MARKER = '    // Preserve N64 operand reuse; merge eight M/N MMA chains below.\n'
OLD_COEFFICIENT = '''            const int coefficient=s.activation_factors[slot][cute::get<0>(coord)]*
                                  s.weight_factors[slot][cute::get<1>(coord)];'''
ROW_SHIFTS = '''    // CuTe coordinate verifier proves four unique rows/thread.
    auto row_shifts=cute::make_tensor<unsigned>(cute::make_shape(cute::_2{},cute::_2{}));
    o1_static_for<0,2>([&](auto mi) {
      o1_static_for<0,2>([&](auto ri) {
        const auto p=coords(ri*cute::_2{},mi,cute::_0{});
        const unsigned factor=static_cast<unsigned>(s.activation_factors[slot][cute::get<0>(p)]);
        // O7 accepted rows: factor=2^d, 0<=d<=30, never zero.
        row_shifts(ri,mi)=31-__clz(factor);
      });
    });
'''
NEW_COEFFICIENT = '''            const unsigned weight_factor=static_cast<unsigned>(
                s.weight_factors[slot][cute::get<1>(coord)]);
            unsigned coefficient_bits;
            // shf.l returns the high word of [weight_factor,0] << d.
            // Existing coefficient guard proves <=INT32_MAX, so this cast
            // is value-preserving. No negative signed shift or quantization.
            asm("shf.l.wrap.b32 %0, 0, %1, %2;" : "=r"(coefficient_bits)
                : "r"(weight_factor),"r"(row_shifts(vi/cute::_2{},mi)));
            const int coefficient=static_cast<int>(coefficient_bits);'''


def generated_header(source):
    text = eight_header(source)
    if text.count(MARKER) != 1 or text.count(OLD_COEFFICIENT) != 1:
        raise ValueError('v78 coefficient or coordinate boundary drift')
    return text.replace(MARKER, ROW_SHIFTS + MARKER).replace(
        OLD_COEFFICIENT, NEW_COEFFICIENT).replace(
        'o78_eight_chain_experiment', 'o7_shift_coefficient_experiment')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, default=Path('reports/o378_roof_v78_codegen'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        p.error('fresh repository output required')
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    prior = json.loads((a.baseline / 'codegen.json').read_text())
    for path, value in prior['sources'].items():
        if sha(ROOT / path) != value:
            raise ValueError('v78 source drift: ' + path)
    if sha(a.baseline / 'o78_eight_chain.cubin') != prior['cubin_sha256']:
        raise ValueError('v78 cubin drift')
    cuda=Path('/usr/local/cuda-12.8/bin'); cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA/CUTLASS required')
    out.mkdir(parents=True)
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    (out/'o78_eight_chain_generated.cuh').write_text(eight_header(source))
    header=out/'o7_shift_coefficient_generated.cuh'; header.write_text(generated_header(source))
    sources=set(prior['sources'])|{
        'csrc/sm80/roof_o7_shift_coefficient_probe.cu',
        'tests/cuda/validate_o7_shift_coefficient_coordinates.cu',
        'scripts/probe_o7_shift_coefficient_codegen.py'}
    receipt=dict(scope='O7_shift_coefficient_compile_gate_not_GPU_acceptance',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},generated_header_sha256=sha(header),
        baseline_cubin_sha256=prior['cubin_sha256'],nvcc=version,cutlass_commit=commit,
        production_default_changed=False,cta_tile=[64,128,128],threads=128,stages=2,shared_bytes=34304,
        changed_semantics=False,supported_variant='o7',commands=[])
    def run(cmd,name):
        receipt['commands'].append(cmd)
        with (out/name).open('w') as f:
            subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
        '-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out)]
    verifier=out/'verify_coordinates'
    run(flags+[str(ROOT/'tests/cuda/validate_o7_shift_coefficient_coordinates.cu'),'-o',str(verifier)],'coordinate_build.log')
    run([str(verifier)],'coordinate_verification.json')
    src=str(ROOT/'csrc/sm80/roof_o7_shift_coefficient_probe.cu')
    cubin=out/'o7_shift_coefficient.cubin'
    run(flags+[src,'-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+[src,'-ptx','-o',str(out/'o7_shift_coefficient.ptx')],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],'o7_shift_coefficient.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/'o7_shift_coefficient.sass').read_text()
    entries=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    live={s:analyze((out/'liveness.txt').read_text(),s) for s in (CONTROL,SYMBOL)}
    ptx=(out/'o7_shift_coefficient.ptx').read_text()
    for s in (CONTROL,SYMBOL):
        body=next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+s+'('))
        if not all(x in body for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32')):
            raise ValueError('same-entry PTX math/copy audit failed')
        if s==SYMBOL and 'shf.l.wrap.b32' not in body: raise ValueError('coefficient shift missing')
    receipt.update(entries=entries,liveness=live,cubin_sha256=sha(cubin),
        control_comparison=compare((a.baseline/'o78_eight_chain.sass').read_text(),sass,'^'+CONTROL+'$'),
        coordinate_verification=json.loads((out/'coordinate_verification.json').read_text()),
        artifact_sha256={s:sha(out/s) for s in ('build.log','o7_shift_coefficient.sass','o7_shift_coefficient.ptx',
            'resources.txt','liveness.txt','coordinate_verification.json')})
    (out/'codegen.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if not receipt['control_comparison']['passed'] or not all(
        e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] and e['all_copies_bypass_l1']
        for e in entries.values()): raise ValueError('same-entry INT4/control audit failed')
    print((out/'build.log').read_text())
    for s in (CONTROL,SYMBOL):
        loop=next(x for x in live[s]['loops'] if x['kind']=='integer')
        print(json.dumps(dict(symbol=s,registers=live[s]['allocated_gpr'],integer_loop=loop),indent=2))
    print('Compile/coordinate audit only: review actual work before authorizing a GPU screen.')


if __name__=='__main__': main()
