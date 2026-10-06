#!/usr/bin/env python3
"""v96: all24 O7/O8 versus best v78, identical v73 online preparation."""
import ctypes as ct
from pathlib import Path

import benchmark_o78_eight_chain_probe as eight
from benchmark_o78_coefficient_probe import main as paired_main
from benchmark_o78_grouped_cta import full_sample_args, validate
from compare_a100_codegen import compare
from probe_interleaved_tail_codegen import ROOT, CONFIG, checked

CFG=CONFIG['o78']


def timing_contract(mode,inner):
    result=eight.timing_contract(mode,inner)
    result.update(comparison='v78_vs_v96_N64_tail_same_v73_preparation',
        new_preparation_or_layout=False,independent_partial_chains=8,partial_registers=32)
    return result


class Driver(eight.Driver):
    def __init__(self,library,baseline,candidate):
        receipt=checked(candidate,'o78')
        if not receipt['worth_runtime_validation']:raise ValueError('negative compile gate')
        control=ROOT/CFG['baseline'];super().__init__(library,baseline,control)
        comparison=compare((control/(CFG['old_stem']+'.sass')).read_text(),
            (candidate/(CFG['stem']+'.sass')).read_text(),'^'+CFG['control']+'$')
        if not comparison['passed']:self.close();raise ValueError('encoded v78 control drift')
        self.handles[0]=self.handles[1];self.resources[0]=dict(self.resources[1])
        try:
            handle=ct.c_void_p()
            self.check(self.lib.roof_probe_open(str((candidate/(CFG['stem']+'.cubin')).resolve()).encode(),
                CFG['symbol'].encode(),CFG['shared'],ct.byref(handle)))
            self.handles[1]=handle
            values=(ct.c_int*4)();self.check(self.lib.roof_probe_resources(handle,values))
            if values[2]!=128 or values[3]<3:raise ValueError('three-CTA capacity must be preserved')
            self.resources[1]=dict(registers_per_thread=values[0],local_size_bytes=values[1],threads=values[2],
                active_blocks_per_sm=values[3],shared_memory_bytes=CFG['shared'],cta_tile=[64,128,128],
                pipeline_stages=2,kernel_symbol=CFG['symbol'],independent_partial_chains=8,partial_registers=32)
            self.codegen=dict(v78=self.codegen,interleaved_tail=receipt,encoded_control=comparison)
        except Exception:self.close();raise


if __name__=='__main__':
    full_sample_args()
    paired_main(driver_cls=Driver,default_gpu_build=Path('reports/o378_roof_v73_codegen'),
        labels=('v78_eight_chain_same_v73_preparation','v96_interleaved_N64_tail_same_v73_preparation'),
        experiment='interleaved_N64_tail',banner='INTERLEAVED TAIL',contract=timing_contract,
        description=__doc__,validation_fn=validate)
