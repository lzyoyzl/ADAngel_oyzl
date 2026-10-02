#!/usr/bin/env python3
"""One v78 scheduling probe: eight merged MMA chains, no extra operand copies.

Unlike v63's four merged N chains, process both M atoms together. Each N64
slice holds 4 registers x 2 M atoms x 4 N atoms = 32 partial registers, the
same logical live-partial budget as v67's low/high pair. The high dot is
multiplied by16 BEFORE the low MMA; group factors and full-K guard unchanged.
No register-cap/tile/pipeline enumeration, production binding or new quantizer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from inspect_o78_register_liveness import analyze
from probe_roof_fullk_integer_codegen import static_entries

ROOT = Path(__file__).resolve().parents[1]
CONTROL = 'adangel_roof_o78_fullk_candidate'
SYMBOL = 'adangel_roof_o78_eight_chain_candidate'
START = '      o1_static_for<0,2>([&](auto mi) {\n'
END = '    });\n  }\n  // Reuse the 64 INT32'
MERGED = '''      // Eight independent chains: two M atoms x four N atoms.
      // Same32 logical partial registers; retain all original A/B reuse.
      auto partial=cute::make_tensor<int>(
          cute::make_shape(cute::_4{},cute::_2{},cute::_4{}));
      cute::clear(partial);
      o1_static_for<0,2>([&](auto mi) {
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,mi,ni);
          cute::gemm(HA{},p,h0(cute::_,mi,cute::_0{}),b0(cute::_,ni,cute::_0{}),p);
        });
      });
      o1_static_for<0,2>([&](auto mi) {
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,mi,ni);
          cute::gemm(HA{},p,h1(cute::_,mi,cute::_0{}),b1(cute::_,ni,cute::_0{}),p);
        });
      });
      // A G128 signed-high dot is at most8192 in magnitude. Multiplication,
      // not signed left-shift, is defined for every intermediate here.
      o1_static_for<0,32>([&](auto i) { partial(i)*=16; });
      o1_static_for<0,2>([&](auto mi) {
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,mi,ni);
          cute::gemm(LA{},p,a0(cute::_,mi,cute::_0{}),b0(cute::_,ni,cute::_0{}),p);
        });
      });
      o1_static_for<0,2>([&](auto mi) {
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,mi,ni);
          cute::gemm(LA{},p,a1(cute::_,mi,cute::_0{}),b1(cute::_,ni,cute::_0{}),p);
        });
      });
      o1_static_for<0,2>([&](auto mi) {
        o1_static_for<0,4>([&](auto ni) {
          auto full_ni=nb*cute::_4{}+ni;
          o1_static_for<0,4>([&](auto vi) {
            const auto coord=coords(vi,mi,full_ni);
            const int coefficient=s.activation_factors[slot][cute::get<0>(coord)]*
                                  s.weight_factors[slot][cute::get<1>(coord)];
            acc(vi,mi,full_ni)+=partial(vi,mi,ni)*coefficient;
          });
        });
      });
'''


def generated_header(source):
    if source.count(START) != 1 or source.count(END) != 1:
        raise ValueError('v67 source boundary drift')
    begin, end = source.index(START), source.index(END)
    if end <= begin:
        raise ValueError('invalid replacement range')
    return ('// Generated v78; same guard/quantization/epilogue, different MMA schedule.\n' +
        (source[:begin] + MERGED + source[end:]).replace(
            'o78_fullk_integer_experiment', 'o78_eight_chain_experiment').replace(
            '// Same N64 stream and four independent MMA atoms as tune59.',
            '// Preserve N64 operand reuse; merge eight M/N MMA chains below.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, default=Path('reports/o378_roof_v67_codegen'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        p.error('fresh repository output required')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    prior = json.loads((a.baseline / 'codegen.json').read_text())
    for path, sha in prior['sources'].items():
        if digest(ROOT / path) != sha:
            raise ValueError('v67 source drift: ' + path)
    if digest(a.baseline / 'o78_fullk.cubin') != prior['cubin_sha256']:
        raise ValueError('v67 binary drift')
    cuda = Path('/usr/local/cuda-12.8/bin'); cutlass = ROOT / 'third_party/cutlass-src'
    version = subprocess.check_output([str(cuda/'nvcc'), '--version'], text=True)
    commit = subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'], text=True).strip()
    if 'release 12.8' not in version or commit != 'db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA/CUTLASS required')
    out.mkdir(parents=True)
    header = out / 'o78_eight_chain_generated.cuh'
    header.write_text(generated_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()))
    sources = set(prior['sources']) | {'csrc/sm80/roof_o78_eight_chain_probe.cu',
                                      'scripts/probe_o78_eight_chain_codegen.py'}
    receipt = dict(scope='isolated_eight_chain_merge_compile_not_performance_acceptance',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:digest(ROOT/s) for s in sorted(sources)}, generated_header_sha256=digest(header),
        baseline_cubin_sha256=prior['cubin_sha256'], production_default_changed=False,
        cta_tile=[64,128,128], threads=128, stages=2, shared_bytes=34304,
        logical_live_partial_registers=32, independent_partial_chains=8,
        nvcc=version,cutlass_commit=commit,commands=[])
    def run(command, filename):
        receipt['commands'].append(command)
        with (out/filename).open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    base=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
        '-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out),str(ROOT/'csrc/sm80/roof_o78_eight_chain_probe.cu')]
    cubin=out/'o78_eight_chain.cubin'
    run(base+['-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(base+['-ptx','-o',str(out/'o78_eight_chain.ptx')],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],'o78_eight_chain.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    symbols={SYMBOL,CONTROL,'adangel_roof_o78_fullk_control'}
    entries=static_entries((out/'o78_eight_chain.sass').read_text(),
        r'^adangel_roof_o78_(?:fullk_(?:control|candidate)|eight_chain_candidate)$',symbols)
    assert all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] and e['all_copies_bypass_l1'] for e in entries.values())
    for symbol in symbols:
        body=next(b for b in re.split(r'(?=\.visible \.entry )',(out/'o78_eight_chain.ptx').read_text())
                  if b.startswith('.visible .entry '+symbol+'('))
        assert all(s in body for s in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32'))
    live=(out/'liveness.txt').read_text()
    receipt.update(cubin_sha256=digest(cubin),entries=entries,
        control_opcode_counts_match=entries[CONTROL]['opcode_counts']==prior['entries'][CONTROL]['opcode_counts'],
        control_instructions_match=entries[CONTROL]['instructions']==prior['entries'][CONTROL]['instructions'],
        liveness={s:analyze(live,s) for s in (CONTROL,SYMBOL)})
    (out/'codegen.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print((out/'build.log').read_text())
    print('same-entry INT4 audit passed; inspect cost before any GPU timing')


if __name__=='__main__':main()
