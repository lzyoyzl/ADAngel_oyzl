#!/usr/bin/env python3
"""v91: direct all24 paired GEMM versus v78, unchanged v73 preparation."""
import ctypes as ct
from pathlib import Path

import benchmark_o78_eight_chain_probe as eight
from benchmark_o78_coefficient_probe import main as paired_main
from benchmark_o78_grouped_cta import full_sample_args, validate
from compare_a100_codegen import compare
from probe_o78_address_remat_codegen import ROOT, STEM, SYMBOL, checked


def timing_contract(mode, inner):
    result = eight.timing_contract(mode, inner)
    result.update(comparison='v78_vs_v91_grouped_A_address_rematerialization_same_v73_preparation',
                  new_preparation_or_layout=False, cta_order_group_m=8,
                  a_copy_address_rematerialization=True)
    return result


class Driver(eight.Driver):
    def __init__(self, library, baseline, candidate):
        receipt = checked(candidate)
        if not receipt['worth_runtime_validation']:
            raise ValueError('negative compile gate: no runtime/performance screening')
        v78 = ROOT / 'reports/o378_roof_v78_codegen'
        super().__init__(library, baseline, v78)
        encoded_control = compare((v78 / 'o78_eight_chain.sass').read_text(),
                                  (candidate / (STEM + '.sass')).read_text(),
                                  '^' + eight.SYMBOL + '$')
        if not encoded_control['passed']:
            self.close()
            raise ValueError('v78 numerical/performance control encoding changed')
        # Reference2 still owns the v67 handle; policy0 becomes the current v78.
        self.handles[0] = self.handles[1]
        self.resources[0] = dict(self.resources[1])
        try:
            handle = ct.c_void_p()
            self.check(self.lib.roof_probe_open(str((candidate / (STEM + '.cubin')).resolve()).encode(),
                                               SYMBOL.encode(), 34304, ct.byref(handle)))
            self.handles[1] = handle
            values = (ct.c_int * 4)()
            self.check(self.lib.roof_probe_resources(handle, values))
            self.resources[1] = dict(registers_per_thread=values[0], local_size_bytes=values[1],
                threads=values[2], active_blocks_per_sm=values[3], shared_memory_bytes=34304,
                cta_tile=[64, 128, 128], pipeline_stages=2, kernel_symbol=SYMBOL,
                cta_order_group_m=8, a_copy_address_rematerialization=True)
            self.codegen = dict(v78=self.codegen, address_rematerialization=receipt,
                                encoded_v78_runtime_control=encoded_control)
        except Exception:
            self.close()
            raise


if __name__ == '__main__':
    full_sample_args()
    paired_main(driver_cls=Driver, default_gpu_build=Path('reports/o378_roof_v73_codegen'),
        labels=('v78_eight_chain_same_v73_preparation', 'v91_grouped_A_address_remat_same_v73_preparation'),
        experiment='grouped_CTA_A_address_rematerialization', banner='ADDRESS REMAT',
        contract=timing_contract, description=__doc__, validation_fn=validate)
