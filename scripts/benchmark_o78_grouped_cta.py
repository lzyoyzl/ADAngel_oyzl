#!/usr/bin/env python3
"""v89: directly test 24 samples versus v78; no four-sample performance screen."""
import ctypes as ct
import json
from pathlib import Path
import sys

import benchmark_o78_eight_chain_probe as eight
from benchmark_o78_coefficient_probe import main as paired_main, validate as old_validate
from probe_grouped_cta_codegen import ROOT, CONFIG, checked

CFG=CONFIG['o78']


def full_sample_args():
    if '--samples' not in sys.argv:
        sys.argv.extend(['--samples','24'])
    elif sys.argv[sys.argv.index('--samples')+1]!='24':
        raise SystemExit('v89 goes directly to all24 real samples; no small performance screen')
    if any(x.startswith('--samples=') and x!='--samples=24' for x in sys.argv):
        raise SystemExit('all24 real samples required')


def timing_contract(mode,inner):
    r=eight.timing_contract(mode,inner)
    r.update(comparison='v78_vs_v89_grouped_CTA_same_v73_preparation',
        new_preparation_or_layout=False,cta_order_group_m=8)
    return r


class Driver(eight.Driver):
    def __init__(self,library,baseline,candidate):
        receipt=checked(candidate,'o78')
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
            self.resources[1]=dict(registers_per_thread=values[0],local_size_bytes=values[1],
                threads=values[2],active_blocks_per_sm=values[3],shared_memory_bytes=CFG['shared'],
                cta_tile=[64,128,128],pipeline_stages=2,kernel_symbol=CFG['symbol'],cta_order_group_m=8)
            self.codegen=dict(v78=self.codegen,grouped_cta=receipt)
        except Exception:self.close();raise


def validate(driver,variants=('o7','o8')):
    # These are correctness/guard tests, not extra performance screens.
    result=old_validate(driver,variants)
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from benchmark_o78_fused_prepare import Case
    from roof_reduction_validation import reference_fp64
    extra=[]
    for m,n in ((512,1024),(576,384)):
        torch.manual_seed(20261006)
        a=torch.randn(m,4096,device='cuda').half()*.25
        w=torch.randn(n,4096,device='cuda').half()*.1
        for variant in variants:
            wf,af=mf.VARIANTS[variant]
            ws,acs=mf.quantize_source(w,wf),mf.quantize_source(a,af)
            # Force a spatially mixed safe/fallback mask, including a tail group.
            if variant=='o7':
                acs['scale'].fill_(127);acs['scale'][:64,-1]=159
            else:
                ws['scale'].fill_(1);ws['scale'][:128,-1]=192
            old=native._benchmark_mixed(variant,'compute_only',ws,acs,0,1,2,'64x128x256','group_major',59,5)
            case=Case(variant,ws,acs,old);guard=driver.prepare(case)
            expected,_=driver.run(case,0,'compute_only',0,1,2);expected=expected.clone()
            semantic=reference_fp64(variant,(*old['converted_activation'],*old['converted_weight']))
            for policy in (0,1):
                output,_=driver.run(case,policy,'compute_only',0,1,2)
                assert torch.equal(output.view(torch.int32),expected.view(torch.int32))
                torch.testing.assert_close(output.double(),semantic,rtol=1e-3,atol=1e-3)
                extra.append(dict(variant=variant,shape=[m,n,4096],policy=policy,
                    fast_or_tail_group_mapping=True,logical_guard_routing=True,
                    bitwise_v78=True,finite_fp32=True,semantic_tolerance_passed=True,**guard))
    result.update(grouped_coordinates_checks=extra,grouped_coordinate_count=len(extra))
    return result


if __name__=='__main__':
    full_sample_args()
    paired_main(driver_cls=Driver,default_gpu_build=Path('reports/o378_roof_v73_codegen'),
        labels=('v78_eight_chain_same_v73_preparation','v89_grouped_CTA_same_v73_preparation'),
        experiment='grouped_CTA_order',banner='GROUPED CTA',contract=timing_contract,
        description=__doc__,validation_fn=validate)
