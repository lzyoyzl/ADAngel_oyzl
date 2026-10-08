#!/usr/bin/env python3
"""Full24 paired v141/v78; same source formats, GPU preparation and stream.

Small shapes are used only for correctness/sanitizer, never a performance
screen. The frozen production extension and all default dispatch are retained.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys

import benchmark_o78_eight_chain_probe as eight
from benchmark_o78_coefficient_probe import main as paired_main, validate as common_validate
from probe_o78_recycled_coefficient_codegen import (
    ROOT, SYMBOL, CONTROL, STEM, LIMITS, gate, generated_header, dependency_audit)


def checked(directory):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((directory/'codegen.json').read_text())
    for path,digest in r['sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError('v141 source drift: '+path)
    for path,digest in r['artifact_sha256'].items():
        if sha(directory/path)!=digest:raise ValueError('v141 artifact drift: '+path)
    if sha(directory/(STEM+'.cubin'))!=r['cubin_sha256']:
        raise ValueError('v141 cubin drift')
    if (directory/(STEM+'_generated.cuh')).read_text()!=generated_header(
            (ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()):
        raise ValueError('v141 generated register handoff drift')
    if r['production_default_changed'] or r['conversion_changed'] or (
        r['cta_tile'],r['threads'],r['stages'],r['shared_bytes'],r['coefficient_slots'])!=(
            [64,128,128],128,2,34304,8):
        raise ValueError('default/geometry/lifetime contract drift')
    if set(r['entries'])!={CONTROL,SYMBOL} or not all(e['native_u4_s4'] and e['native_s4_s4']
        and not e['int8_mma'] and e['all_copies_bypass_l1'] for e in r['entries'].values()):
        raise ValueError('same-entry native INT4/copy audit failed')
    if r['limits']!=LIMITS or not r['control_comparison']['passed']:
        raise ValueError('predeclared limits/control drift')
    sass=(directory/(STEM+'.sass')).read_text()
    dep={s:dependency_audit(sass,s,r['liveness'][s]) for s in (CONTROL,SYMBOL)}
    if dep!=r['dependency_audit']:raise ValueError('static dependency evidence drift')
    actual=gate(r['liveness'][CONTROL],r['liveness'][SYMBOL],dep[SYMBOL])
    if actual!=r['compile_gate'] or not actual['passed']:
        raise ValueError('compile gate failed; do not execute candidate')
    return dict(build=r,runtime_source_sha256={name:sha(ROOT/name) for name in (
        'scripts/benchmark_o78_recycled_coefficient.py',
        'scripts/benchmark_o78_coefficient_probe.py','scripts/benchmark_o78_eight_chain_probe.py',
        'scripts/benchmark_o78_fullk_gpu_prepare.py','scripts/benchmark_o78_row_fused.py')})


class Driver(eight.Driver):
    def __init__(self,library,baseline,candidate):
        receipt=checked(candidate)
        control=candidate.parent/'o378_roof_v78_codegen'
        super().__init__(library,baseline,control)
        if self.codegen['eight_chain']['build']['cubin_sha256']!=receipt['build']['baseline_cubin_sha256']:
            self.close();raise ValueError('loaded v78 identity drift')
        # Policy2 remains original v67 for bitwise regression, policy0 is v78.
        self.handles[0]=self.handles[1]
        self.resources[0]=dict(self.resources[1])
        try:
            handle=ct.c_void_p()
            self.check(self.lib.roof_probe_open(str((candidate/(STEM+'.cubin')).resolve()).encode(),
                SYMBOL.encode(),34304,ct.byref(handle)))
            self.handles[1]=handle
            values=(ct.c_int*4)();self.check(self.lib.roof_probe_resources(handle,values))
            self.resources[1]=dict(registers_per_thread=values[0],local_size_bytes=values[1],
                threads=values[2],active_blocks_per_sm=values[3],shared_memory_bytes=34304,
                cta_tile=[64,128,128],pipeline_stages=2,kernel_symbol=SYMBOL)
            if values[0]>168 or values[1]!=0 or values[2]!=128 or values[3]<3:
                raise ValueError('loaded candidate resource gate failed')
            self.codegen=dict(v78=self.codegen,recycled_coefficient=receipt)
        except Exception:
            self.close();raise

    def prepare(self,case):
        require_variant(case)
        return super().prepare(case)

    def run(self,case,policy,mode,warmup,repeats,inner):
        require_variant(case)
        return super().run(case,policy,mode,warmup,repeats,inner)


def require_variant(case):
    if case.variant not in ('o7','o8'):
        raise ValueError('bounded two-sided coefficients are O7/O8 only')


def validate(driver):
    r=common_validate(driver,variants=('o7','o8'))
    r.update(bounded_coefficient_handoff=True,bitwise_reference='original_v67_fullK_and_v78',
             quantization_and_scale_unchanged=True,no_small_performance_screen=True)
    return r


def timing_contract(mode,inner):
    r=eight.timing_contract(mode,inner)
    r.update(comparison='v78_vs_v141_same_v73_preparation',
             supported_variants=['o7','o8'],conversion_changed=False)
    return r


def full_sample_args(argv):
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--samples',type=int,choices=(24,),default=24)
    p.parse_known_args(argv)
    return argv if any(x=='--samples' or x.startswith('--samples=') for x in argv) else argv+['--samples','24']


if __name__=='__main__':
    sys.argv[1:]=full_sample_args(sys.argv[1:])
    paired_main(driver_cls=Driver,default_gpu_build=Path('reports/o378_roof_v73_codegen'),
        labels=('v78_eight_chain_same_v73_preparation','v141_bounded_coefficient_same_v73_preparation'),
        experiment='bounded_coefficient_register_handoff',banner='RECYCLED COEFFICIENT',
        contract=timing_contract,variants=('o7','o8'),validation_fn=validate,description=__doc__)
