#!/usr/bin/env python3
"""v139: integrate previously measured O7 conversions, not a new GEMM.

Control = v118 packed NV4 W + original v73 MX8 A.
Combined = the SAME W + v106 MX8 warp lookup A.
Reuse the frozen v126 library's CONTROL branch only. The failed v126
packed-MX8 candidate is never launched and its failed gate is not changed.
Both policies use the identical original v78 native-INT4 CUfunction.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

import benchmark_o78_eight_chain_probe as eight
from benchmark_o78_coefficient_probe import main as paired_main
from benchmark_o7_nv4_swar import validate as validate_weight
from compare_a100_codegen import compare
from probe_mx8_swar_codegen import ROOT, checked as checked_library, generated_host
from analyze_o78_row_fused_codegen import entries

BUILD = Path('reports/o378_roof_v126_codegen_r2')


def route(policy):
    if policy == 0:
        return 'roof_o78_nv4_swar_benchmark', 1
    if policy == 1:
        # 0 selects lookup_activation(), NOT the rejected packed_activation().
        return 'roof_o78_mx8_swar_benchmark', 0
    if policy == 2:
        return 'roof_o78_row_fused_benchmark', 1
    raise ValueError('invalid combined-conversion policy')


def checked(directory):
    receipt = checked_library(directory)
    after = (directory/'prepare.sass').read_text()
    proofs = {}
    for label, prior, name, suffix in (
        ('packed_weight', 'reports/o378_roof_v118_codegen', 'adangel_sm80_row_swar_metadata', 'GroupedSourceKindE0E'),
        ('lookup_activation', 'reports/o378_roof_v106_codegen', 'adangel_sm80_row_warp_lut_metadata', 'GroupedSourceKindE1E'),
        ('scalar_activation', 'reports/o378_roof_v73_codegen', 'adangel_sm80_row_conversion_metadata', 'GroupedSourceKindE1E'),
        ('guard', 'reports/o378_roof_v73_codegen', 'adangel_o78_prepare_cta_guard', ''),
    ):
        before = (ROOT/prior/'prepare.sass').read_text()
        symbols = [s for s in entries(before) if name in s and suffix in s]
        if len(symbols) != 1:
            raise ValueError('one retained conversion entry required: '+label)
        proofs[label] = compare(before, after, '^'+re.escape(symbols[0])+'$')
        if not proofs[label]['passed']:
            raise ValueError('retained conversion SASS changed: '+label)
    host = generated_host()
    if ('if(candidate)packed_activation();else lookup_activation();' not in host or
            'void weight(bool) {Nv4SwarOnline::weight(true);}' not in host):
        raise ValueError('frozen host route no longer selects the measured kernels')
    return dict(frozen_build=receipt, retained_entry_comparisons=proofs,
        frozen_v126_candidate_gate_passed=receipt['audit']['worth_runtime_validation'],
        failed_v126_packed_MX8_candidate_executed=False, new_CUDA_compilation=False,
        control_route=route(0), combined_route=route(1), GEMM_modified=False,
        production_default_changed=False)


def timing_contract(mode, inner):
    result = eight.row_fused.timing_contract(mode, inner, 1)
    result.update(comparison='v118_W_v73_A_vs_same_W_v106_A_identical_v78_GEMM',
        gemm_cufunction_identical_between_policies=True, weight_preparation_identical=True,
        weight_preparation_implementation='v118_packed_NVFP4',
        preparation_implementation='integration_of_existing_measured_conversions',
        modified_stage='activation_conversion_selection_only', integer_guard_unchanged=True,
        failed_v126_packed_MX8_candidate_executed=False, no_small_performance_screen=True,
        payload_norm_checked_after_every_call=True)
    return result


class Driver(eight.Driver):
    def __init__(self, library, baseline, candidate):
        receipt = checked(library.parent)
        super().__init__(library, baseline, candidate)
        self.handles[0] = self.handles[1]
        self.resources[0] = dict(self.resources[1])
        self.functions = {}
        for policy in (0, 1, 2):
            name, selector = route(policy)
            fn = getattr(self.lib, name)
            fn.argtypes = self.lib.roof_o78_gpu_benchmark.argtypes
            fn.restype = ct.c_int
            self.functions[policy] = (fn, selector)
            self.resources[policy] = dict(self.resources[policy], GEMM_modified=False,
                conversion_host_function=name, conversion_selector=selector,
                weight_preparation='v73_reference' if policy == 2 else 'v118_packed_NVFP4',
                activation_preparation='v106_MXFP8_lookup' if policy == 1 else 'v73_scalar_MXFP8')
        decoder = self.lib.roof_nv4_swar_exhaustive
        decoder.argtypes = [ct.c_void_p, ct.c_void_p]
        decoder.restype = ct.c_int

        def prepare_combined(variant, sa, sw, state, m, n, am, wm, stream):
            if variant != 7:
                raise ValueError('O7 integration only')
            times = (ct.c_float*4)()
            fn, selector = self.functions[1]
            return fn(self.handles[1], variant, selector, 0, sa, sw, state, m, n,
                      am, wm, 0, 1, 2, stream, times)
        self.lib.roof_o78_gpu_prepare = prepare_combined
        self.codegen = dict(best_GEMM=self.codegen, conversion_combo=receipt,
            runtime_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def run(self, case, policy, mode, warmup, repeats, inner):
        import torch
        if case.variant != 'o7' or policy not in self.functions or mode not in eight.base.MODES:
            raise ValueError('invalid O7 combination policy/mode')
        if np.any(case.oracle['status_flat'] == 2):
            raise ValueError('invalid source cannot expose an unwritten output')
        fn, selector = self.functions[policy]
        values = (ct.c_float*(4*repeats))()
        self.check(fn(self.handles[policy], 7, selector, eight.base.MODES.index(mode),
            case.a_source, case.w_source, case.state_pointers, case.m, case.n,
            case.a_multiplier, case.w_multiplier, warmup, repeats, inner,
            torch.cuda.current_stream().cuda_stream, values))
        for key in case.expected_payload:
            if not torch.equal(case.state[key].view(torch.uint8), case.expected_payload[key].view(torch.uint8)):
                raise ValueError('conversion payload/scale mismatch: '+key)
        for key in ('asq', 'wsq'):
            if not np.array_equal(case.state[key].cpu().numpy(), case.group_squares_reference[key]):
                raise ValueError('conversion exact norm mismatch: '+key)
        return case.state['y'], eight.base.normalize_timings(mode,
            np.ctypeslib.as_array(values).reshape(4, repeats), repeats)


def main():
    for flag, value in (('--samples','24'),('--rounds','3'),('--warmup','1000'),('--repeats','200'),('--inner','100')):
        if flag in sys.argv and sys.argv[sys.argv.index(flag)+1] != value:
            raise SystemExit('fixed full24x3/1000/200/100; no small performance screen')
        if flag not in sys.argv:
            sys.argv.extend([flag, value])
    if '--full-modes' not in sys.argv:
        sys.argv.append('--full-modes')
    if '--cubins' not in sys.argv:
        sys.argv.extend(['--cubins','reports/o378_roof_v78_codegen'])
    paired_main(driver_cls=Driver, default_gpu_build=BUILD,
        labels=('v118_W_v73_A_same_v78_GEMM','v118_W_v106_A_same_v78_GEMM'),
        experiment='O7_existing_conversion_combination_not_new_GEMM', banner='O7 CONVERSION COMBINATION',
        contract=timing_contract, description=__doc__, variants=('o7',), validation_fn=validate_weight)
    if '--validate-only' in sys.argv:
        return
    output = Path(sys.argv[sys.argv.index('--output')+1])
    prior = ROOT/'docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78/source_provenance.jsonl'
    old = {r['sample_id']:r for r in map(json.loads, prior.read_text().splitlines()) if r['variant'] == 'o7'}
    rows = list(map(json.loads, (output/'source_provenance.jsonl').read_text().splitlines()))
    if len(rows) != 24 or {r['sample_id'] for r in rows} != set(old) or any(r != old[r['sample_id']] for r in rows):
        raise ValueError('source identity differs from v99')
    (output/'source_identity_checked.json').write_text(json.dumps(dict(passed=True, samples=24,
        variant='o7', full_v99_source_identity_equal=True,
        old_provenance_sha256=hashlib.sha256(prior.read_bytes()).hexdigest()), indent=2)+'\n')


if __name__ == '__main__':
    main()
