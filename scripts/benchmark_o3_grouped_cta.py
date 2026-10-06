#!/usr/bin/env python3
"""v89: O3 grouped CTA versus v79, same online conversion2 and GPU guard."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys

import benchmark_o3_eight_chain_probe as protocol
from benchmark_o78_grouped_cta import full_sample_args
from probe_grouped_cta_codegen import ROOT, CONFIG, checked
from roof_full_pipeline_probe import Pipeline as FullPipeline, build as full_build

CFG=CONFIG['o3']
ORIGINAL_VALIDATE=protocol.validate


def build(output,codegen):
    receipt=checked(codegen,'o3')
    baseline=ROOT/CFG['baseline'];control=baseline/CFG['cubin']
    if hashlib.sha256(control.read_bytes()).hexdigest()!=receipt['baseline_cubin_sha256']:
        raise ValueError('v79 best identity drift')
    built=full_build(output,ROOT/'reports/o378_roof_v59',ROOT/'reports/o378_roof_v61',
        ROOT/'runs/o378_roof_v60_screen/build')
    (output/'grouped_build.json').write_text(json.dumps(dict(
        codegen=receipt,baseline_cubin_sha256=receipt['baseline_cubin_sha256'],
        runtime_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__),ROOT/'scripts/benchmark_o3_eight_chain_probe.py')},
        comparison='v79_vs_v89_grouped_CTA',conversion_candidate=2,
        preparation_identical=True,production_default_changed=False),indent=2)+'\n')
    return (*built,control,codegen/(CFG['stem']+'.cubin'))


class Pipeline(FullPipeline):
    def __init__(self,*built):
        self.grouped_handles={}
        super().__init__(*built[:-2])
        try:
            for policy,(cubin,symbol) in enumerate(zip(built[-2:],(CFG['control'],CFG['symbol']))):
                handle=ct.c_void_p()
                self.check(self.lib.roof_probe_open(str(cubin.resolve()).encode(),symbol.encode(),
                    CFG['shared'],ct.byref(handle)))
                self.grouped_handles[policy]=handle
                values=(ct.c_int*4)();self.check(self.lib.roof_probe_resources(handle,values))
                self.resources['v79' if policy==0 else 'v89']=dict(registers_per_thread=values[0],
                    local_size_bytes=values[1],threads=values[2],active_blocks_per_sm=values[3],
                    shared_memory_bytes=CFG['shared'],kernel_symbol=symbol)
        except Exception:self.close();raise

    def close(self):
        for handle in getattr(self,'grouped_handles',{}).values():self.check(self.lib.roof_probe_close(handle))
        self.grouped_handles={};super().close()

    def run_four(self,policy,*args,**kwargs):
        if policy not in (0,1):raise ValueError('O3 paired policies0/1 required')
        old=self.device
        try:
            self.device=self.grouped_handles[policy]
            result=super().run_four(1,*args,**kwargs)
        finally:self.device=old
        result['kernel'].update(gemm_tune='v79_eight_chain' if policy==0 else 'v89_grouped_CTA',
            kernel_symbol=CFG['control'] if policy==0 else CFG['symbol'],cta_tile=[64,128,128],
            pipeline_stages=3,cta_order_group_m=1 if policy==0 else 8)
        return result


def validate(driver):
    result=ORIGINAL_VALIDATE(driver)
    import torch
    extra=[]
    for m,n in ((512,1024),(576,384)):
        torch.manual_seed(20261006)
        a=torch.randint(-128,128,(m,4096),device='cuda',dtype=torch.int8)
        w=torch.randint(0,256,(n,2048),device='cuda',dtype=torch.uint8)
        asc=torch.linspace(.001,.01,m,device='cuda')
        ws=torch.full((n,32),116,device='cuda',dtype=torch.uint8)
        ws[128:,-1]=131
        old=driver.run_four(0,'compute_only',a,asc,w,ws,0,1,2)['output']
        new=driver.run_four(1,'compute_only',a,asc,w,ws,0,1,2)['output']
        assert torch.equal(old.view(torch.int32),new.view(torch.int32)) and bool(torch.isfinite(new).all())
        extra.append(dict(shape=[m,n,4096],fast_or_tail_group_mapping=True,
            logical_guard_routing=True,bitwise_v79=True,finite_fp32=True))
    result.update(grouped_coordinates_checks=extra)
    return result


def main():
    full_sample_args()
    if '--codegen' not in sys.argv:
        sys.argv.extend(['--codegen','reports/o378_roof_v89_o3_codegen'])
    protocol.build=build;protocol.Pipeline=Pipeline;protocol.validate=validate
    protocol.main()
    # The reused protocol preserves all original record/timing fields. Correct
    # its historical v61/v79 descriptive label; numerical records are untouched.
    output=Path(sys.argv[sys.argv.index('--output')+1])
    env=json.loads((output/'environment.json').read_text())
    env.update(scope='O3 v79 vs v89 grouped_CTA, same conversion2 and online GPU guard',
        control=CFG['control'],candidate=CFG['symbol'],cta_order_group_m=8,
        no_small_performance_screen=True)
    (output/'environment.json').write_text(json.dumps(env,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
