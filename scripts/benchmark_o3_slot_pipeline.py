#!/usr/bin/env python3
"""v95: all24 O3 versus best v89; same conversion2 and online guard."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys

import benchmark_o3_eight_chain_probe as protocol
from benchmark_o3_grouped_cta import validate
from benchmark_o78_grouped_cta import full_sample_args
from probe_slot_pipeline_codegen import ROOT, CONFIG, checked
from roof_full_pipeline_probe import Pipeline as FullPipeline, build as full_build

CFG=CONFIG['o3']


def build(output,codegen):
    receipt=checked(codegen,'o3');control=ROOT/CFG['baseline']/CFG['cubin']
    if not receipt['worth_runtime_validation']:raise ValueError('slot-pipeline compile gate failed')
    if hashlib.sha256(control.read_bytes()).hexdigest()!=receipt['baseline_cubin_sha256']:
        raise ValueError('v89 control identity drift')
    built=full_build(output,ROOT/'reports/o378_roof_v59',ROOT/'reports/o378_roof_v61',
        ROOT/'runs/o378_roof_v60_screen/build')
    (output/'slot_build.json').write_text(json.dumps(dict(codegen=receipt,
        comparison='v89_vs_v95_slot_handshake',conversion_candidate=2,
        preparation_identical=True,production_default_changed=False),indent=2)+'\n')
    return (*built,control,codegen/(CFG['stem']+'.cubin'))


class Pipeline(FullPipeline):
    def __init__(self,*built):
        self.slot_handles={};super().__init__(*built[:-2])
        try:
            for policy,(cubin,symbol,shared) in enumerate(zip(built[-2:],
                    (CFG['control'],CFG['symbol']),(CFG['old_shared'],CFG['shared']))):
                handle=ct.c_void_p()
                self.check(self.lib.roof_probe_open(str(cubin.resolve()).encode(),symbol.encode(),shared,ct.byref(handle)))
                self.slot_handles[policy]=handle
                values=(ct.c_int*4)();self.check(self.lib.roof_probe_resources(handle,values))
                if values[2]!=128 or values[3]<3:raise ValueError('three resident CTAs required')
                self.resources['v89' if policy==0 else 'v95']=dict(registers_per_thread=values[0],
                    local_size_bytes=values[1],threads=values[2],active_blocks_per_sm=values[3],
                    shared_memory_bytes=shared,kernel_symbol=symbol,producer_extra_warps=0)
        except Exception:self.close();raise

    def close(self):
        for h in getattr(self,'slot_handles',{}).values():self.check(self.lib.roof_probe_close(h))
        self.slot_handles={};super().close()

    def run_four(self,policy,*args,**kwargs):
        if policy not in (0,1):raise ValueError('O3 paired policies0/1 required')
        old=self.device
        try:
            self.device=self.slot_handles[policy];r=super().run_four(1,*args,**kwargs)
        finally:self.device=old
        r['kernel'].update(gemm_tune='v89_grouped_CTA' if policy==0 else 'v95_slot_handshake',
            kernel_symbol=CFG['control'] if policy==0 else CFG['symbol'],cta_tile=[64,128,128],
            pipeline_stages=3,cta_order_group_m=8,producer_extra_warps=0)
        return r


def main():
    full_sample_args()
    if '--codegen' not in sys.argv:sys.argv.extend(['--codegen','reports/o378_roof_v95_o3_codegen'])
    protocol.build=build;protocol.Pipeline=Pipeline;protocol.validate=validate
    protocol.main()
    output=Path(sys.argv[sys.argv.index('--output')+1])
    env=json.loads((output/'environment.json').read_text())
    env.update(scope='O3 v89 vs v95 SM80 slot handshake, same conversion2 and GPU guard',
        control=CFG['control'],candidate=CFG['symbol'],cta_order_group_m=8,
        producer_extra_warps=0,no_small_performance_screen=True)
    (output/'environment.json').write_text(json.dumps(env,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
