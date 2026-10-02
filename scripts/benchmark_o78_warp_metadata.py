#!/usr/bin/env python3
"""v81 vs v78, identical v73 online preparation; no formal default change."""
import ctypes as ct
import hashlib
import json
from pathlib import Path

import benchmark_o78_eight_chain_probe as eight
from benchmark_o78_coefficient_probe import main as paired_main
from probe_o78_warp_metadata_codegen import ROOT, SYMBOL, CONTROL, generated_header


def checked(directory):
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((directory/'codegen.json').read_text())
    if digest(directory/'o78_warp_metadata.cubin')!=r['cubin_sha256']:raise ValueError('cubin drift')
    for path,sha in r['sources'].items():
        if digest(ROOT/path)!=sha:raise ValueError('source drift: '+path)
    header=directory/'o78_warp_metadata_generated.cuh'
    if digest(header)!=r['generated_header_sha256'] or header.read_text()!=generated_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()):raise ValueError('header drift')
    if not r['control_comparison']['passed'] or r['production_default_changed']:raise ValueError('control/default drift')
    for symbol in (CONTROL,SYMBOL):
        c=eight.merged_mma_counts((directory/'o78_warp_metadata.sass').read_text(),r['liveness'][symbol],symbol)
        if c!=dict(u4_total=32,u4_zero_c=0,s4_total=32,s4_zero_c=16):raise ValueError('MMA work drift')
        loop=next(l for l in r['liveness'][symbol]['loops'] if l['kind']=='integer')
        if loop['opcode_counts'].get('LDSM.16.M88.4')!=16:raise ValueError('fragment rereads changed')
    return r


class Driver(eight.Driver):
    def __init__(self,library,baseline,candidate):
        receipt=checked(candidate)
        super().__init__(library,baseline,ROOT/'reports/o378_roof_v78_codegen')
        self.handles[0]=self.handles[1]
        self.resources[0]=dict(self.resources[1])
        # handles[2] retains exact v67 for synthetic/MSE regression.
        try:
            h=ct.c_void_p()
            self.check(self.lib.roof_probe_open(str((candidate/'o78_warp_metadata.cubin').resolve()).encode(),SYMBOL.encode(),34304,ct.byref(h)))
            self.handles[1]=h
            values=(ct.c_int*4)();self.check(self.lib.roof_probe_resources(h,values))
            self.resources[1]=dict(registers_per_thread=values[0],local_size_bytes=values[1],threads=values[2],
                active_blocks_per_sm=values[3],shared_memory_bytes=34304,cta_tile=[64,128,128],pipeline_stages=2,kernel_symbol=SYMBOL)
            self.codegen=dict(previous=self.codegen,warp_metadata=receipt)
        except Exception:self.close();raise


def timing_contract(mode,inner):
    r=eight.timing_contract(mode,inner)
    r['comparison']='v78_vs_v81_full_warp_metadata_same_v73_preparation'
    return r


if __name__=='__main__':
    paired_main(driver_cls=Driver,default_gpu_build=Path('reports/o378_roof_v73_codegen'),
        labels=('v78_eight_chain','v81_full_warp_factor_copy'),experiment='full_warp_metadata',
        banner='FULL WARP FACTOR COPY',contract=timing_contract,description=__doc__)
