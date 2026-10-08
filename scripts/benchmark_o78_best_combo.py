#!/usr/bin/env python3
"""v142: v78 versus v99 GEMM with identical best O7/O8 conversion.

Reuse audited binaries: O7=v139 preparation, O8=v138 preparation.
No CUDA compilation, default switch, timing change, or source quantization change.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import benchmark_o7_conversion_combo as o7
import benchmark_o8_hif4_swar as o8
from benchmark_o78_coefficient_probe import main as paired_main
from benchmark_o378_output_streaming import resources
from probe_output_streaming_codegen import ROOT, CONFIG, checked as checked_streaming

BUILDS = {'o7': o7.BUILD, 'o8': Path('reports/o378_roof_v138_codegen_r2')}
PREPARATIONS = {'o7': 'v139_NVFP4_packed_W_MXFP8_lookup_A',
                'o8': 'v138_HiF4_packed_W_v123_FP6_packed_A'}
SYMBOLS = ('adangel_roof_o78_eight_chain_candidate',
           'adangel_roof_o78_output_streaming_candidate')


def route(variant, policy):
    if variant not in BUILDS or policy not in (0, 1):
        raise ValueError('O7/O8 paired policy 0/1 required')
    # Selectors choose CONVERSION, not the GEMM. Both sides use the same one.
    return (('roof_o78_mx8_swar_benchmark', 0) if variant == 'o7'
            else ('roof_o78_hif4_swar_benchmark', 1))


def timing_contract(variant, mode, inner):
    result = o7.eight.row_fused.timing_contract(mode, inner, 1)
    result.update(comparison='v78_vs_v99_with_identical_best_conversion',
        gemm_cufunction_identical_between_policies=False,
        weight_preparation_identical=True, activation_preparation_identical=True,
        preparation_implementation=PREPARATIONS[variant],
        modified_stage='GEMM_selection_only_existing_v99_binary',
        integer_guard_unchanged=True, payload_norm_checked_after_every_call=True,
        failed_v126_packed_MX8_candidate_executed=False,
        no_small_performance_screen=True, new_CUDA_compilation=False,
        production_default_changed=False)
    return result


def make_driver(variant):
    parent = o7.Driver if variant == 'o7' else o8.Driver

    class Driver(parent):
        def __init__(self, library, baseline, candidate):
            receipt = checked_streaming(candidate, 'o78')
            if not receipt['worth_runtime_validation']:
                raise ValueError('v99 compile gate failed')
            super().__init__(library, baseline, ROOT/CONFIG['o78']['baseline'])
            try:
                control = self.codegen['best_GEMM']['eight_chain']['build']
                if control['cubin_sha256'] != receipt['baseline_cubin_sha256']:
                    raise ValueError('actual v78 control differs from v99 audited control')
                # Parent has aliased 0/1 to v78. Keep 0, replace only 1 by v99.
                preparation_resource = dict(self.resources[1])
                self.resources[0] = dict(preparation_resource)
                handle = ct.c_void_p()
                cfg = CONFIG['o78']
                self.check(self.lib.roof_probe_open(
                    str((candidate/(cfg['stem']+'.cubin')).resolve()).encode(),
                    cfg['symbol'].encode(), cfg['shared'], ct.byref(handle)))
                self.handles[1] = handle
                self.resources[1] = dict(preparation_resource)
                self.resources[1].update(resources(self.lib, self.check, handle, 'o78', cfg['symbol']))
                for policy in (0, 1):
                    fn, selector = route(variant, policy)
                    self.resources[policy].update(conversion_host_function=fn,
                        conversion_selector=selector, preparation=PREPARATIONS[variant],
                        GEMM_modified=False, GEMM_selection_changed=bool(policy),
                        integer_output_policy='streaming_cs' if policy else 'default_wb')
                    if self.resources[policy]['kernel_symbol'] != SYMBOLS[policy]:
                        raise ValueError('wrong actual GEMM symbol')
                self.codegen = dict(retained_preparation_and_v78=self.codegen,
                    output_streaming=receipt, preparation=PREPARATIONS[variant],
                    new_CUDA_compilation=False, production_default_changed=False,
                    runtime_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            except Exception:
                self.close()
                raise

        def run(self, case, policy, mode, warmup, repeats, inner):
            import torch
            if policy == 2:
                # Independently retained v67 reference, not either tested kernel.
                return super().run(case, policy, mode, warmup, repeats, inner)
            if case.variant != variant or mode not in o7.eight.base.MODES:
                raise ValueError('invalid combined case/mode')
            name, selector = route(variant, policy)
            if np.any(case.oracle['status_flat'] == 2):
                raise ValueError('invalid source cannot expose an unwritten output')
            fn = getattr(self.lib, name)
            fn.argtypes = self.lib.roof_o78_gpu_benchmark.argtypes
            fn.restype = ct.c_int
            values = (ct.c_float*(4*repeats))()
            self.check(fn(self.handles[policy], int(variant[1:]), selector,
                o7.eight.base.MODES.index(mode), case.a_source, case.w_source,
                case.state_pointers, case.m, case.n, case.a_multiplier,
                case.w_multiplier, warmup, repeats, inner,
                torch.cuda.current_stream().cuda_stream, values))
            # Outside timing: verify every output payload, scale, and exact norm.
            for key in case.expected_payload:
                if not torch.equal(case.state[key].view(torch.uint8),
                                   case.expected_payload[key].view(torch.uint8)):
                    raise ValueError('conversion payload/scale mismatch: '+key)
            for key in ('asq', 'wsq'):
                if not np.array_equal(case.state[key].cpu().numpy(), case.group_squares_reference[key]):
                    raise ValueError('conversion exact norm mismatch: '+key)
            return case.state['y'], o7.eight.base.normalize_timings(mode,
                np.ctypeslib.as_array(values).reshape(4, repeats), repeats)

    return Driver


def main():
    if '--variant' not in sys.argv:
        raise SystemExit('--variant o7/o8 required')
    index = sys.argv.index('--variant')
    variant = sys.argv[index+1]
    del sys.argv[index:index+2]
    if variant not in BUILDS:
        raise SystemExit('only O7/O8 integration is in scope')
    for flag, value in (('--samples','24'), ('--rounds','3'), ('--warmup','1000'),
                        ('--repeats','200'), ('--inner','100')):
        if flag in sys.argv and sys.argv[sys.argv.index(flag)+1] != value:
            raise SystemExit('fixed full24x3/1000/200/100; no performance screen')
        if flag not in sys.argv:
            sys.argv.extend([flag, value])
    if '--full-modes' not in sys.argv:
        sys.argv.append('--full-modes')
    if '--cubins' not in sys.argv:
        sys.argv.extend(['--cubins', 'reports/o378_roof_v99_o78_codegen'])
    paired_main(driver_cls=make_driver(variant), default_gpu_build=BUILDS[variant],
        labels=('v78_same_'+PREPARATIONS[variant], 'v99_same_'+PREPARATIONS[variant]),
        experiment='v142_best_conversion_GEMM_integration', banner='BEST COMBINATION',
        contract=lambda mode, inner: timing_contract(variant, mode, inner),
        description=__doc__, variants=(variant,),
        validation_fn=o7.validate_weight if variant == 'o7' else o8.validate)
    if '--validate-only' in sys.argv:
        return
    output = Path(sys.argv[sys.argv.index('--output')+1])
    prior = ROOT/'docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78/source_provenance.jsonl'
    old = {r['sample_id']: r for r in map(json.loads, prior.read_text().splitlines())
           if r['variant'] == variant}
    rows = list(map(json.loads, (output/'source_provenance.jsonl').read_text().splitlines()))
    if len(rows) != 24 or {r['sample_id'] for r in rows} != set(old) or any(r != old[r['sample_id']] for r in rows):
        raise ValueError('full24 source identity differs from v99')
    (output/'source_identity_checked.json').write_text(json.dumps(dict(passed=True,
        samples=24, variant=variant, full_v99_source_identity_equal=True,
        old_provenance_sha256=hashlib.sha256(prior.read_bytes()).hexdigest()), indent=2)+'\n')


if __name__ == '__main__':
    main()
