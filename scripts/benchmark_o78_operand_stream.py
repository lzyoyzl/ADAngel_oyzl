#!/usr/bin/env python3
"""v94: all24 O7/O8 versus v78, same v73 online preparation."""
import ctypes as ct
from pathlib import Path

import benchmark_o78_eight_chain_probe as eight
from benchmark_o78_coefficient_probe import main as paired_main
from benchmark_o78_grouped_cta import full_sample_args, validate
from probe_operand_stream_codegen import ROOT, BASELINE, CONTROL, SYMBOL, STEM, SHARED, checked


def timing_contract(mode,inner):
    result=eight.timing_contract(mode,inner)
    result.update(comparison='v78_vs_v94_operand_lifetime_same_v73_preparation',
        new_preparation_or_layout=False,independent_partial_chains=4,partial_registers=16)
    return result


class Driver(eight.Driver):
    def __init__(self,library,baseline,candidate):
        receipt=checked(candidate)
        if not receipt['worth_runtime_validation']:raise ValueError('four-CTA compile gate failed')
        super().__init__(library,baseline,ROOT/BASELINE)
        if self.codegen['eight_chain']['build']['cubin_sha256']!=receipt['baseline_cubin_sha256']:
            self.close();raise ValueError('v78 identity drift')
        self.handles[0]=self.handles[1];self.resources[0]=dict(self.resources[1])
        try:
            handle=ct.c_void_p()
            self.check(self.lib.roof_probe_open(str((candidate/(STEM+'.cubin')).resolve()).encode(),
                SYMBOL.encode(),SHARED,ct.byref(handle)))
            self.handles[1]=handle
            values=(ct.c_int*4)();self.check(self.lib.roof_probe_resources(handle,values))
            if values[2]!=128 or values[3]<4:raise ValueError('four naturally resident CTAs required')
            self.resources[1]=dict(registers_per_thread=values[0],local_size_bytes=values[1],threads=values[2],
                active_blocks_per_sm=values[3],shared_memory_bytes=SHARED,cta_tile=[64,128,128],
                pipeline_stages=2,kernel_symbol=SYMBOL,independent_partial_chains=4,partial_registers=16)
            self.codegen=dict(v78=self.codegen,operand_stream=receipt)
        except Exception:self.close();raise


if __name__=='__main__':
    full_sample_args()
    paired_main(driver_cls=Driver,default_gpu_build=Path('reports/o378_roof_v73_codegen'),
        labels=('v78_eight_chain_same_v73_preparation','v94_operand_stream_same_v73_preparation'),
        experiment='A_operand_lifetime_four_CTA',banner='OPERAND STREAM',contract=timing_contract,
        description=__doc__,validation_fn=validate)
