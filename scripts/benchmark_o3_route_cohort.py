#!/usr/bin/env python3
"""v137: isolated two-native-INT4 route cohorts versus best v89, full24 only.

No default, source quantization or conversion change. Small MN cases here
validate correctness/safety; no small performance screen is performed.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import benchmark_o3_eight_chain_probe as protocol
from benchmark_o3_grouped_cta import validate as original_validate
from benchmark_o78_grouped_cta import full_sample_args
from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_o3_route_cohort_codegen import ROOT, SYMBOL, CONTROL, STEM, cost_gate
from probe_roof_fullk_integer_codegen import static_entries
from roof_full_pipeline_probe import Pipeline as FullPipeline, build as full_build


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(codegen):
    r=json.loads((codegen/'codegen.json').read_text())
    for name,digest in r['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('audited source drift: '+name)
    for name,digest in r['artifact_sha256'].items():
        if sha(codegen/name)!=digest:raise ValueError('audited artifact drift: '+name)
    old=ROOT/'reports/o378_roof_v89_o3_codegen'
    if sha(old/'o3_grouped_cta.cubin')!=r['baseline_cubin_sha256']:raise ValueError('v89 drift')
    sass=(codegen/(STEM+'.sass')).read_text()
    control=compare((old/'o3_grouped_cta.sass').read_text(),sass,'^'+CONTROL+'$')
    entries=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    live=analyze((codegen/'liveness.txt').read_text(),SYMBOL)
    if cost_gate(live,entries,control)!=r['cost_gate'] or not r['cost_gate']['passed']:
        raise ValueError('compile gate replay failed')
    return r


def build(output,codegen):
    receipt=checked(codegen)
    built=full_build(output,ROOT/'reports/o378_roof_v59',ROOT/'reports/o378_roof_v61',
        ROOT/'runs/o378_roof_v60_screen/build')
    driver=ROOT/'csrc/sm80/roof_o3_route_cohort_driver.cpp';lib=output/'libcohort_driver.so'
    command=['g++','-O3','-std=c++17','-shared','-fPIC','-I/usr/local/cuda-12.8/include',
        str(driver),'-lcuda','-o',str(lib)]
    with (output/'cohort_driver_build.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    paths=[Path(__file__),driver]+[ROOT/x for x in (
        'scripts/benchmark_o3_eight_chain_probe.py','scripts/benchmark_o3_grouped_cta.py',
        'scripts/roof_full_pipeline_probe.py','csrc/sm80/roof_full_pipeline_driver.cpp',
        'csrc/sm80/roof_device_factor_driver.cpp','csrc/sm80/roof_gpu_factor_driver.cpp',
        'csrc/sm80/roof_producer_warp_driver.cpp')]
    (output/'cohort_build.json').write_text(json.dumps(dict(codegen=receipt,
        runtime_sources={str(p.relative_to(ROOT)):sha(p) for p in paths},command=command,
        driver_sha256=sha(lib),no_GEMM_recompile=True,preparation_identical=True,
        production_default_changed=False),indent=2)+'\n')
    return (lib,*built[1:],ROOT/'reports/o378_roof_v89_o3_codegen/o3_grouped_cta.cubin',
        codegen/(STEM+'.cubin'))


class Pipeline(FullPipeline):
    def __init__(self,*built):
        self.pair={}
        super().__init__(*built[:-2])
        self.lib.roof_cohort_open.argtypes=[ct.c_char_p,ct.c_char_p,ct.POINTER(ct.c_void_p)]
        try:
            for policy,(cubin,symbol) in enumerate(zip(built[-2:],(CONTROL,SYMBOL))):
                handle=ct.c_void_p()
                self.check(self.lib.roof_cohort_open(str(cubin.resolve()).encode(),symbol.encode(),ct.byref(handle)))
                self.pair[policy]=handle
                values=(ct.c_int*4)();self.check(self.lib.roof_probe_resources(handle,values))
                threads=256 if policy else 128
                if values[2]!=threads or values[3]<(2 if policy else 3):
                    raise ValueError('actual capacity gate failed before any launch: '+str(list(values)))
                self.resources['v137' if policy else 'v89']=dict(registers_per_thread=values[0],
                    local_size_bytes=values[1],threads=values[2],active_blocks_per_sm=values[3],
                    active_warps_per_sm=values[3]*threads//32,shared_memory_bytes=50688,kernel_symbol=symbol)
        except Exception:self.close();raise

    def close(self):
        for handle in getattr(self,'pair',{}).values():self.check(self.lib.roof_probe_close(handle))
        self.pair={};super().close()

    def run_four(self,policy,*args,**kwargs):
        if policy not in (0,1):raise ValueError('paired policies0/1 required')
        old=self.device
        try:
            self.device=self.pair[policy]
            result=super().run_four(1,*args,**kwargs)
        finally:self.device=old
        result['kernel'].update(gemm_tune='v137_route_cohort' if policy else 'v89_grouped_eight',
            kernel_symbol=SYMBOL if policy else CONTROL,cta_tile=[64,128,128],pipeline_stages=3,
            cta_order_group_m=8,threads=256 if policy else 128,two_native_int4=True,int8_tensor_core=False,
            fullk_route_cohorts=bool(policy),shared_handoff='once_after_K' if policy else None,
            preparation_identical=True)
        return result


def safety_validate(driver):
    """Small grid exercises both routes and retired-warp fallback in one launch."""
    import torch
    checks=[]
    m,n=128,384
    a=((torch.arange(m*4096,device='cuda')*13)%256-128).to(torch.int8).reshape(m,4096)
    w=(torch.arange(n*2048,device='cuda')%256).byte().reshape(n,2048)
    asc=torch.linspace(0,.01,m,device='cuda')
    ws=((torch.arange(n*32,device='cuda')%4)+116).byte().reshape(n,32)
    ws[256:,-1]=131
    old=driver.run_four(0,'compute_only',a,asc,w,ws,0,1,2)
    for mode in protocol.MODES:
        new=driver.run_four(1,mode,a,asc,w,ws,0,2,2)
        assert new['status']==1
        assert torch.equal(old['output'].view(torch.int32),new['output'].view(torch.int32))
        checks.append(dict(mode=mode,mixed_integer_fallback=True,bitwise_best=True))
    return dict(passed=True,checks=checks,scope='mixed128x384x4096 safety; no performance result')


def main():
    safety='--safety-only' in sys.argv
    if safety:
        sys.argv.remove('--safety-only')
        if '--validate-only' not in sys.argv:sys.argv.append('--validate-only')
    full_sample_args()
    for name,value in (('--warmup','1000'),('--repeats','200'),('--rounds','3'),('--inner','100'),
        ('--codegen','reports/o378_roof_v137_o3_route_cohort_fixed_codegen')):
        if name not in sys.argv:sys.argv.extend((name,value))
    protocol.build=build;protocol.Pipeline=Pipeline
    protocol.validate=safety_validate if safety else original_validate
    protocol.main()
    out=Path(sys.argv[sys.argv.index('--output')+1]);env=json.loads((out/'environment.json').read_text())
    env.update(scope='O3 v89 vs v137 full-K route cohorts; identical online conversion/guard',
        control=CONTROL,candidate=SYMBOL,no_small_performance_screen=True,
        single_int8_route=False,safety_only=safety)
    (out/'environment.json').write_text(json.dumps(env,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
