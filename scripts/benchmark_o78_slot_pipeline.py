#!/usr/bin/env python3
"""v95: all24 O7/O8 versus best v78; same v73 online preparation."""
import ctypes as ct
from pathlib import Path

import benchmark_o78_eight_chain_probe as eight
from benchmark_o78_coefficient_probe import main as paired_main
from benchmark_o78_grouped_cta import full_sample_args, validate
from probe_slot_pipeline_codegen import ROOT, CONFIG, checked

CFG=CONFIG['o78']


def timing_contract(mode,inner):
    r=eight.timing_contract(mode,inner)
    r.update(comparison='v78_vs_v95_slot_handshake_same_v73_preparation',
        new_preparation_or_layout=False,producer_warp_also_computes=True,
        consumer_warps=4,producer_extra_warps=0,full_arrivals=32,empty_arrivals=128)
    return r


class Driver(eight.Driver):
    def __init__(self,library,baseline,candidate):
        receipt=checked(candidate,'o78')
        if not receipt['worth_runtime_validation']:raise ValueError('slot-pipeline compile gate failed')
        super().__init__(library,baseline,ROOT/CFG['baseline'])
        if self.codegen['eight_chain']['build']['cubin_sha256']!=receipt['baseline_cubin_sha256']:
            self.close();raise ValueError('v78 control identity drift')
        self.handles[0]=self.handles[1];self.resources[0]=dict(self.resources[1])
        try:
            handle=ct.c_void_p()
            self.check(self.lib.roof_probe_open(str((candidate/(CFG['stem']+'.cubin')).resolve()).encode(),
                CFG['symbol'].encode(),CFG['shared'],ct.byref(handle)))
            self.handles[1]=handle
            values=(ct.c_int*4)();self.check(self.lib.roof_probe_resources(handle,values))
            if values[2]!=128 or values[3]<3:raise ValueError('three resident CTAs required')
            self.resources[1]=dict(registers_per_thread=values[0],local_size_bytes=values[1],
                threads=values[2],active_blocks_per_sm=values[3],shared_memory_bytes=CFG['shared'],
                cta_tile=[64,128,128],pipeline_stages=2,kernel_symbol=CFG['symbol'],
                producer_extra_warps=0,producer_warp_also_computes=True)
            self.codegen=dict(v78=self.codegen,slot_pipeline=receipt)
        except Exception:self.close();raise


if __name__=='__main__':
    full_sample_args()
    paired_main(driver_cls=Driver,default_gpu_build=Path('reports/o378_roof_v73_codegen'),
        labels=('v78_eight_chain_same_v73_preparation','v95_slot_handshake_same_v73_preparation'),
        experiment='SM80_slot_handshake',banner='SLOT PIPELINE',contract=timing_contract,
        description=__doc__,validation_fn=validate)
