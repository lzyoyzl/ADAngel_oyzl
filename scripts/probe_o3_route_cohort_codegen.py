#!/usr/bin/env python3
"""v137 fixed route-cohort compile audit, no runtime or performance claim.

Distinguish from v125/v135: separate warp cohorts keep route sums over all K,
then combine once. Shared W fragment reads increase; register capacity must
permit two256-thread CTAs before spending a full24 runtime experiment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_grouped_cta_codegen import ROOT, checked
from probe_roof_fullk_integer_codegen import static_entries

SYMBOL = 'adangel_roof_o3_route_cohort_candidate'
CONTROL = 'adangel_roof_o3_grouped_cta_candidate'
STEM = 'o3_route_cohort'
LIMITS = dict(max_allocated_gpr=128, max_hot_local_instructions=4,
    min_actual_ctas_before_runtime=2, block_threads=256, shared_bytes=50688,
    tile=[64,128,128], stages=3, producer_specialization=False)


def cost_gate(live, entries, control):
    hot = next(row for row in live['loops'] if row['kind'] == 'integer')
    count = lambda prefix: sum(v for k, v in hot['opcode_counts'].items() if k.startswith(prefix))
    checks = dict(control_unchanged=control['passed'],
        allocation_for_two_ctas=live['allocated_gpr']<=128,
        limited_hot_local=count('LDL')+count('STL')<=4,
        native_u4_and_s4=entries[SYMBOL]['native_u4_s4'] and entries[SYMBOL]['native_s4_s4'],
        no_int8=not entries[SYMBOL]['int8_mma'],
        route_static_mma=count('IMMA.16864.U4.S4')==count('IMMA.16864.S4.S4')==32,
        no_hot_float=count('I2F')==count('FFMA')==count('FMUL')==0,
        one_loop_barrier=count('BAR')==1)
    return dict(passed=all(checks.values()), checks=checks, limits=LIMITS,
        integer_loop_static=hot['static_instructions'], integer_loop_peak_live=hot['max_live_gpr'],
        hot_local_loads=count('LDL'), hot_local_stores=count('STL'),
        interpretation='static branch arms are mutually exclusive perwarp; not direct dynamic speedup',
        full24_and_sanitizer_required=True, runtime_resources_not_yet_verified=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    baseline=ROOT/'reports/o378_roof_v89_o3_codegen';prior=checked(baseline,'o3')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('fixed CUDA12.8/CUTLASS required')
    out.mkdir(parents=True)
    for path in baseline.iterdir():
        if path.suffix in ('.cu','.cuh'):(out/path.name).write_bytes(path.read_bytes())
    sources=set(prior['sources'])|{'csrc/sm80/o3_route_cohort_candidate.cuh',
        'csrc/sm80/roof_o3_route_cohort_probe.cu','scripts/probe_o3_route_cohort_codegen.py',
        'scripts/inspect_o78_register_liveness.py','scripts/probe_roof_fullk_integer_codegen.py',
        'scripts/compare_a100_codegen.py'}
    result=dict(scope='route_cohort_compile_gate_not_GPU_validation',limits=LIMITS,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        nvcc=version,cutlass_commit=commit,sources={name:sha(ROOT/name) for name in sorted(sources)},
        baseline_cubin_sha256=prior['cubin_sha256'],commands=[],production_default_changed=False,
        source_quantization_changed=False,scale_semantics_changed=False,
        conversion_changed=False,new_runtime_measurements=False,
        extra_epilogue_shared_bytes_per_cta=65536,weight_fragment_read_multiplier=2,
        total_fragment_read_multiplier=1.5)
    def run(cmd,name):
        result['commands'].append(cmd);(out/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
        with (out/name).open('w') as f:
            subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,TMPDIR=str(ROOT/'tmp')),stdout=f,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-std=c++17','--expt-relaxed-constexpr','-arch=sm_80',
        '-DADANGEL_FULLK_INTEGER=1','-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out)]
    source=str(ROOT/'csrc/sm80/roof_o3_route_cohort_probe.cu');cubin=out/(STEM+'.cubin')
    run(flags+['-O3','-lineinfo',source,'-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+['-O3','-lineinfo',source,'-ptx','-o',str(out/(STEM+'.ptx'))],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],STEM+'.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/(STEM+'.sass')).read_text();ptx=(out/(STEM+'.ptx')).read_text()
    entries=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    control=compare((baseline/'o3_grouped_cta.sass').read_text(),sass,'^'+CONTROL+'$')
    live=analyze((out/'liveness.txt').read_text(),SYMBOL)
    # The existing parser's128-thread occupancy note is not valid here.
    live['cta_threads']=256
    live.pop('register_only_four_cta_limit_per_thread')
    live['register_only_two_cta_limit_per_thread']=128
    body=next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+SYMBOL+'('))
    if not all(x in body for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32')):
        raise ValueError('same-entry native PTX/supply missing')
    gate=cost_gate(live,entries,control)
    result.update(entries=entries,liveness=live,control_comparison=control,cost_gate=gate,
        cubin_sha256=sha(cubin),artifact_sha256={path.name:sha(path) for path in out.iterdir()
            if path.is_file() and path.name not in ('progress.json','codegen.json')})
    (out/'codegen.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(cost_gate=gate,liveness=live),indent=2),flush=True)
    print('GATE PASS: actual occupancy, numerical/sanitizer and full24 required' if gate['passed'] else
        'GATE FAIL: no runtime timing or neighboring parameter scan',flush=True)


if __name__=='__main__':main()
