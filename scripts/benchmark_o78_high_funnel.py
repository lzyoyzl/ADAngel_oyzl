#!/usr/bin/env python3
"""v117 fixed high-x16 SHF versus exact v78, same v73 online preparation.

Independent O7/O8 test only. No small performance screen, production change,
new factor metadata, changed scale multiplication or floating summation.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys

import benchmark_o78_eight_chain_probe as eight
from benchmark_o78_coefficient_probe import main as paired_main, validate as common_validate
from inspect_eight_chain_schedule import trace
from probe_o78_high_funnel_codegen import (ROOT,SYMBOL,CONTROL,LIMITS,cost_gate,
                                         generated_header,high_reconstruction)


def checked(directory):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((directory/'codegen.json').read_text())
    for path,digest in r['sources'].items():
        if sha(ROOT/path)!=digest:
            raise ValueError('v117 source drift: '+path)
    for path,digest in r['artifact_sha256'].items():
        if sha(directory/path)!=digest:
            raise ValueError('v117 artifact drift: '+path)
    if sha(directory/'o78_high_funnel.cubin')!=r['cubin_sha256']:
        raise ValueError('v117 cubin drift')
    h=directory/'o78_high_funnel_generated.cuh'
    if sha(h)!=r['generated_header_sha256'] or h.read_text()!=generated_header(
            (ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()):
        raise ValueError('fixed high-only body drift')
    if r['production_default_changed'] or r['supported_variants']!=['o7','o8'] or (
        r['cta_tile'],r['threads'],r['stages'],r['shared_bytes'])!=([64,128,128],128,2,34304):
        raise ValueError('variant/default/resources contract drift')
    if set(r['entries'])!={CONTROL,SYMBOL} or not all(e['native_u4_s4'] and e['native_s4_s4']
        and not e['int8_mma'] and e['all_copies_bypass_l1'] for e in r['entries'].values()):
        raise ValueError('same-entry native INT4/copy audit failed')
    if r['limits']!=LIMITS or not r['control_comparison']['passed']:
        raise ValueError('predeclared gate/control drift')
    sass=(directory/'o78_high_funnel.sass').read_text()
    routes={s:high_reconstruction(sass,s,r['liveness'][s]) for s in (CONTROL,SYMBOL)}
    if routes!=r['high_reconstruction']:
        raise ValueError('high def-use receipt drift')
    gate=cost_gate(r['liveness'][CONTROL],r['liveness'][SYMBOL],routes[SYMBOL])
    gate['control_encoding_unchanged']=True
    if gate!=r['cost_gate'] or not gate['passed']:
        raise ValueError('fixed high opcode gate failed: candidate must not launch')
    return dict(build=r,chains={s:trace(sass,s,r['liveness'][s]) for s in (CONTROL,SYMBOL)},
        runtime_source_sha256={name:sha(ROOT/name) for name in (
            'scripts/benchmark_o78_high_funnel.py','scripts/benchmark_o78_coefficient_probe.py',
            'scripts/benchmark_o78_fullk_gpu_prepare.py','scripts/inspect_eight_chain_schedule.py')})


class Driver(eight.Driver):
    def __init__(self,library,baseline,candidate):
        receipt=checked(candidate)
        control=candidate.parent/'o378_roof_v78_codegen'
        super().__init__(library,baseline,control)
        if self.codegen['eight_chain']['build']['cubin_sha256']!=receipt['build']['baseline_cubin_sha256']:
            self.close();raise ValueError('actual v78 control identity drift')
        # Original v67 remains policy2; v78 becomes the actual paired policy0.
        self.handles[0]=self.handles[1]
        self.resources[0]=dict(self.resources[1])
        try:
            handle=ct.c_void_p()
            self.check(self.lib.roof_probe_open(str((candidate/'o78_high_funnel.cubin').resolve()).encode(),
                SYMBOL.encode(),34304,ct.byref(handle)))
            self.handles[1]=handle
            values=(ct.c_int*4)();self.check(self.lib.roof_probe_resources(handle,values))
            self.resources[1]=dict(registers_per_thread=values[0],local_size_bytes=values[1],threads=values[2],
                active_blocks_per_sm=values[3],shared_memory_bytes=34304,cta_tile=[64,128,128],
                pipeline_stages=2,kernel_symbol=SYMBOL)
            if values[0]>168 or values[1]!=0 or values[2]!=128 or values[3]<3:
                raise ValueError('actual loaded candidate resource gate failed')
            self.codegen=dict(v78=self.codegen,high_funnel=receipt)
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
        raise ValueError('v117 is O7/O8 only; no O3 or other dispatch')


def validate(driver):
    # Same exact source/packing/scales, nondefault stream, positive/negative,
    # zero, random, invalid encoding, saturation and fallback edge suite.
    result=common_validate(driver,variants=('o7','o8'))
    result.update(fixed_high_only=True,bitwise_reference='original_v67_fullK_and_v78',
        quantization_and_scale_unchanged=True,no_small_performance_screen=True)
    return result


def timing_contract(mode,inner):
    r=eight.timing_contract(mode,inner)
    r.update(comparison='v78_vs_v117_fixed_high_funnel_same_v73_preparation',
        supported_variants=['o7','o8'],new_preparation_or_layout=False)
    return r


def require_full_samples(argv):
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--samples',type=int,choices=(24,),default=24)
    p.parse_known_args(argv)
    return argv if any(x=='--samples' or x.startswith('--samples=') for x in argv) else argv+['--samples','24']


if __name__=='__main__':
    sys.argv[1:]=require_full_samples(sys.argv[1:])
    paired_main(driver_cls=Driver,default_gpu_build=Path('reports/o378_roof_v73_codegen'),
        labels=('v78_eight_chain_same_v73_preparation','v117_fixed_high_funnel_same_v73_preparation'),
        experiment='fixed_high_funnel',banner='FIXED HIGH FUNNEL',contract=timing_contract,
        variants=('o7','o8'),validation_fn=validate,description=__doc__)
