#!/usr/bin/env python3
"""v79: port the measured eight-chain schedule to O3's unchanged 3-stage path.

Only the partial MMA schedule changes. Same O3 G128 factors, full-K INT32
guard, input layout, async ring, native U4/S4 MMA and FP32 fallback/epilogue.
Not a tile/register-cap sweep, production default, or timing acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_o78_eight_chain_codegen import MERGED
from probe_roof_device_factor_codegen import wrapper
from probe_roof_factor_async_codegen import generated_header as async_header
from probe_roof_fullk_integer_codegen import static_entries

ROOT=Path(__file__).resolve().parents[1]
CONTROL='adangel_roof_device_factor_o3'
CANDIDATE='adangel_roof_o3_eight_chain_candidate'
SENTINEL='adangel_roof_fullk_integer_o78'
BEGIN='      o1_static_for<0,2>([&](auto mi) {\n'
END='    });\n  }\n  auto final_value='
LAMBDA_BEGIN='  auto integer_group='
LAMBDA_END='  const int groups=32;'


def generated_header(source):
    text=async_header(source)
    for marker in (BEGIN,END,LAMBDA_BEGIN,LAMBDA_END):
        if text.count(marker)!=1:raise ValueError('O3 v61 source boundary drift: '+marker)
    merged=MERGED
    names=dict(h0='h00',h1='h01',a0='a00',a1='a01',b0='b00',b1='b01')
    merged=re.sub(r'\b(?:h0|h1|a0|a1|b0|b1)\b',lambda m:names[m[0]],merged)
    coefficient=('const int coefficient=s.activation_factors[slot][cute::get<0>(coord)]*\n'
                 '                                  s.weight_factors[slot][cute::get<1>(coord)];')
    if merged.count(coefficient)!=1:raise ValueError('eight-chain source drift')
    merged=merged.replace(coefficient,'const int coefficient=s.factor[slot][cute::get<1>(coord)];')
    lo,hi=text.index(BEGIN),text.index(END)
    text=text[:lo]+merged+text[hi:]
    # No stale unused old MMA lambda: all other generated source remains v61.
    lo,hi=text.index(LAMBDA_BEGIN),text.index(LAMBDA_END)
    text=text[:lo]+text[hi:]
    return text.replace('o3_fullk_integer_experiment','o3_eight_chain_experiment')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('reports/o378_roof_v61'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    prior=json.loads((a.baseline/'codegen.json').read_text())
    for path,sha in prior['sources'].items():
        if digest(ROOT/path)!=sha:raise ValueError('O3 v61 source drift: '+path)
    if digest(a.baseline/'device_factor.cubin')!=prior['cubin_sha256']:
        raise ValueError('O3 v61 cubin drift')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA12.8/CUTLASS required')
    source=(ROOT/'csrc/sm80/o3_fullk_integer_probe.cuh').read_text()
    original=wrapper((ROOT/'csrc/sm80/roof_fullk_integer_probe.cu').read_text())
    if original!=(a.baseline/'device_factor.cu').read_text():raise ValueError('baseline wrapper drift')
    if async_header(source)!=(a.baseline/'o3_factor_async_generated.cuh').read_text():
        raise ValueError('baseline generated header drift')
    out.mkdir(parents=True)
    (out/'o3_device_control_generated.cu').write_text(original)
    (out/'o3_factor_async_generated.cuh').write_text(async_header(source))
    header=out/'o3_eight_chain_generated.cuh';header.write_text(generated_header(source))
    sources=set(prior['sources'])|{
        'scripts/probe_o3_eight_chain_codegen.py','scripts/probe_o78_eight_chain_codegen.py',
        'scripts/probe_roof_device_factor_codegen.py','scripts/probe_roof_factor_async_codegen.py',
        'csrc/sm80/roof_o3_eight_chain_probe.cu'}
    result=dict(scope='O3_eight_chain_codegen_not_performance_acceptance',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:digest(ROOT/s) for s in sorted(sources)},generated_header_sha256=digest(header),
        baseline_cubin_sha256=prior['cubin_sha256'],production_default_changed=False,
        cta_tile=[64,128,128],threads=128,stages=3,shared_bytes=50688,
        logical_live_partial_registers=32,independent_partial_chains=8,
        nvcc=version,cutlass_commit=commit,commands=[])
    def run(command,name):
        result['commands'].append(command)
        with (out/name).open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    base=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
          '-DADANGEL_FULLK_INTEGER=1','-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),
          '-I'+str(out),str(ROOT/'csrc/sm80/roof_o3_eight_chain_probe.cu')]
    cubin=out/'o3_eight_chain.cubin'
    run(base+['-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(base+['-ptx','-o',str(out/'o3_eight_chain.ptx')],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],'o3_eight_chain.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/'o3_eight_chain.sass').read_text();symbols={CONTROL,CANDIDATE,SENTINEL}
    entries=static_entries(sass,r'^adangel_roof_(?:device_factor_o3|o3_eight_chain_candidate|fullk_integer_o78)$',symbols)
    assert all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] and e['all_copies_bypass_l1'] for e in entries.values())
    ptx=(out/'o3_eight_chain.ptx').read_text()
    for symbol in symbols:
        block=next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+symbol+'('))
        assert all(s in block for s in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32'))
    result.update(cubin_sha256=digest(cubin),entries=entries,
        control_comparison=compare((a.baseline/'device_factor.sass').read_text(),sass,'^'+CONTROL+'$'),
        sentinel_comparison=compare((a.baseline/'device_factor.sass').read_text(),sass,'^'+SENTINEL+'$'),
        liveness={s:analyze((out/'liveness.txt').read_text(),s) for s in (CONTROL,CANDIDATE)})
    (out/'codegen.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['control_comparison']['passed'] or not result['sentinel_comparison']['passed']:
        raise ValueError('unchanged control/sentinel codegen drift')
    print('O3 eight-chain compile/audit completed; no performance/MSE acceptance yet')


if __name__=='__main__':main()
