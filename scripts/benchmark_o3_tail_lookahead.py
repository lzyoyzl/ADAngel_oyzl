#!/usr/bin/env python3
"""v97: all24 O3 versus best v89, same conversion2 and online guard."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys

import benchmark_o3_eight_chain_probe as protocol
from benchmark_o3_grouped_cta import validate
from benchmark_o78_grouped_cta import full_sample_args
from probe_tail_lookahead_codegen import ROOT, CONFIG, checked
from roof_full_pipeline_probe import Pipeline as FullPipeline, build as full_build

CFG=CONFIG['o3']


def build(output,codegen):
    receipt=checked(codegen,'o3')
    if not receipt['worth_runtime_validation']:raise ValueError('negative compile gate')
    control=ROOT/CFG['baseline']/(CFG['old_stem']+'.cubin')
    if hashlib.sha256(control.read_bytes()).hexdigest()!=receipt['baseline_cubin_sha256']:
        raise ValueError('v89 best cubin drift')
    built=full_build(output,ROOT/'reports/o378_roof_v59',ROOT/'reports/o378_roof_v61',
        ROOT/'runs/o378_roof_v60_screen/build')
    (output/'tail_lookahead_build.json').write_text(json.dumps(dict(codegen=receipt,
        comparison='v89_vs_v97_fixed_two_startup_chains',conversion_candidate=2,
        preparation_identical=True,production_default_changed=False,
        runtime_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__),ROOT/'scripts/benchmark_o3_eight_chain_probe.py',
                      ROOT/'scripts/benchmark_o3_grouped_cta.py')}),indent=2)+'\n')
    return (*built,control,codegen/(CFG['stem']+'.cubin')),receipt


class Pipeline(FullPipeline):
    def __init__(self,built,receipt):
        self.lookahead_handles={}
        self.receipt=receipt
        super().__init__(*built[:-2])
        try:
            for policy,(cubin,symbol) in enumerate(zip(built[-2:],(CFG['control'],CFG['symbol']))):
                handle=ct.c_void_p()
                self.check(self.lib.roof_probe_open(str(cubin.resolve()).encode(),symbol.encode(),
                    CFG['shared'],ct.byref(handle)))
                self.lookahead_handles[policy]=handle
                values=(ct.c_int*4)();self.check(self.lib.roof_probe_resources(handle,values))
                if values[0]>168 or values[2]!=128 or values[3]<3:
                    raise ValueError('168-GPR and three-CTA capacity gates must be preserved')
                self.resources['v89' if policy==0 else 'v97']=dict(registers_per_thread=values[0],
                    local_size_bytes=values[1],threads=values[2],active_blocks_per_sm=values[3],
                    shared_memory_bytes=CFG['shared'],kernel_symbol=symbol,
                    source_max_chains=8 if policy==0 else 10,partial_registers=32 if policy==0 else 40,
                    static_programmed_chain_peak=receipt['ordering'][symbol]['peak_started_not_finished_chains'])
        except Exception:self.close();raise

    def close(self):
        for handle in getattr(self,'lookahead_handles',{}).values():self.check(self.lib.roof_probe_close(handle))
        self.lookahead_handles={};super().close()

    def run_four(self,policy,*args,**kwargs):
        if policy not in (0,1):raise ValueError('paired policy0/1 required')
        old=self.device
        try:
            self.device=self.lookahead_handles[policy];result=super().run_four(1,*args,**kwargs)
        finally:self.device=old
        symbol=CFG['control'] if policy==0 else CFG['symbol']
        result['kernel'].update(gemm_tune='v89_grouped_eight' if policy==0 else 'v97_two_startup_chains',
            kernel_symbol=symbol,cta_tile=[64,128,128],pipeline_stages=3,cta_order_group_m=8,
            source_max_chains=8 if policy==0 else 10,partial_registers=32 if policy==0 else 40,
            static_programmed_chain_peak=self.receipt['ordering'][symbol]['peak_started_not_finished_chains'])
        return result


def main():
    full_sample_args()
    if '--codegen' not in sys.argv:sys.argv.extend(['--codegen','reports/o378_roof_v97_o3_codegen'])
    protocol.build=build;protocol.Pipeline=Pipeline;protocol.validate=validate
    protocol.main()
    output=Path(sys.argv[sys.argv.index('--output')+1]);env=json.loads((output/'environment.json').read_text())
    env.update(scope='O3 v89 vs v97 fixed two startup chains; same conversion2/GPU guard',
        control=CFG['control'],candidate=CFG['symbol'],no_small_performance_screen=True)
    (output/'environment.json').write_text(json.dumps(env,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
