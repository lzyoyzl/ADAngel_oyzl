#!/usr/bin/env python3
"""v127 full24 GEMM pair; original zero-spill gate remains failed.

One address-word reload is allowed by a separate, pre-execution resource
review under the user's small-spill authorization. No default dispatch change.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess

import benchmark_o78_eight_chain_probe as eight
from benchmark_o78_coefficient_probe import main as paired_main, validate as old_validate
from benchmark_o78_grouped_cta import full_sample_args
from compare_a100_codegen import compare
from inspect_eight_chain_schedule import trace
from inspect_o78_register_liveness import analyze
from probe_o78_activation_panel_codegen import (
    ROOT, BASELINE, CONTROL, SYMBOL, STEM, SHARED, generated_header, cost_gate,
)

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def same_json_value(live, saved):
    # Chain histograms have integer keys in memory, string keys in JSON.
    return json.loads(json.dumps(live, allow_nan=False)) == saved


def resource_review(gate, candidate_live, resources):
    """Investment decision only, not an acceptance rule or performance claim."""
    counts = next(x for x in candidate_live['loops'] if x['kind'] == 'integer')['opcode_counts']
    loads = sum(v for k, v in counts.items() if k.startswith('LDL'))
    stores = sum(v for k, v in counts.items() if k.startswith('STL'))
    checks = dict(
        initial_failure_preserved=not gate['passed'] and not gate['checks']['no_hot_local'],
        all_other_original_checks=all(v for k, v in gate['checks'].items() if k != 'no_hot_local'),
        one_audited_hot_reload=loads == 1 and stores == 0,
        actual_resources=resources['candidate']['registers_per_thread'] == 168
        and resources['candidate']['local_bytes'] == 8
        and resources['candidate']['threads'] == 128
        and resources['candidate']['active_blocks_per_sm'] == 3,
    )
    return dict(passed=all(checks.values()), checks=checks,
        authority='user explicitly permits small spill if correct and faster',
        original_zero_spill_gate=gate,
        caveat='static work -7.05% is not speed; static incomplete-chain peak 8->6; startup copy and local reload cost remain',
        no_latency_observed_before_resource_review=True)


def checked(directory):
    r = json.loads((directory / 'codegen.json').read_text())
    for name, digest in r['sources'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('source drift: ' + name)
    for name, digest in r['artifact_sha256'].items():
        if sha(directory / name) != digest:
            raise ValueError('artifact drift: ' + name)
    if (directory / (STEM + '_generated.cuh')).read_text() != generated_header(
            (ROOT / 'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()):
        raise ValueError('generated activation panel drift')
    if sha(BASELINE / 'o78_eight_chain.cubin') != r['baseline_cubin_sha256']:
        raise ValueError('old best cubin drift')
    sass = (directory / (STEM + '.sass')).read_text()
    live = {s: analyze((directory / 'liveness.txt').read_text(), s) for s in (CONTROL, SYMBOL)}
    if live != r['liveness']:
        raise ValueError('resource audit replay drift')
    if not same_json_value({s: trace(sass, s, live[s]) for s in (CONTROL, SYMBOL)}, r['schedules']):
        raise ValueError('schedule replay drift')
    control = compare((BASELINE / 'o78_eight_chain.sass').read_text(), sass, '^' + CONTROL + '$')
    if not control['passed'] or control != r['control_comparison']:
        raise ValueError('old complete machine code drift')
    gate = cost_gate(live[CONTROL], live[SYMBOL], r['runtime_resources'])
    gate['checks']['control_encoding_unchanged'] = control['passed']
    gate['passed'] = all(gate['checks'].values())
    if gate != r['cost_gate'] or r['production_default_changed'] or not r['math_and_quantization_unchanged']:
        raise ValueError('original gate or numerical contract drift')
    if not all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma']
               and e['all_copies_bypass_l1'] for e in r['entries'].values()):
        raise ValueError('native INT4/copy audit failed')
    review = resource_review(gate, live[SYMBOL], r['runtime_resources'])
    if not review['passed']:
        raise ValueError('pre-execution small-spill resource review failed')
    return dict(codegen=r, resource_review=review)


def timing_contract(mode, inner):
    r = eight.timing_contract(mode, inner)
    r.update(comparison='v78_vs_v127_fullK_activation_factor_panel_same_v73_preparation',
        conversion_unchanged=True, activation_panel_startup_in_GEMM=True,
        original_compile_gate_passed=False, independent_small_spill_review=True)
    return r


class Driver(eight.Driver):
    def __init__(self, library, baseline, candidate):
        receipt = checked(candidate)
        super().__init__(library, baseline, BASELINE)
        # v67 remains policy2 (reference/owner); v78 becomes the measured control.
        self.handles[0] = self.handles[1]
        self.resources[0] = dict(self.resources[1])
        try:
            handle = ct.c_void_p()
            self.check(self.lib.roof_probe_open(str((candidate / (STEM + '.cubin')).resolve()).encode(),
                SYMBOL.encode(), SHARED, ct.byref(handle)))
            self.handles[1] = handle
            values = (ct.c_int * 4)()
            self.check(self.lib.roof_probe_resources(handle, values))
            current = dict(zip(('registers_per_thread', 'local_bytes', 'threads', 'active_blocks_per_sm'), values))
            current['shared_bytes'] = SHARED
            if current != receipt['codegen']['runtime_resources']['candidate']:
                raise ValueError('actual candidate resources drift')
            self.resources[1] = dict(registers_per_thread=values[0], local_size_bytes=values[1],
                threads=values[2], active_blocks_per_sm=values[3], shared_memory_bytes=SHARED,
                cta_tile=[64,128,128], pipeline_stages=2, kernel_symbol=SYMBOL)
            self.codegen = dict(v78=self.codegen, activation_factor_panel=receipt)
        except Exception:
            self.close()
            raise


def validate(driver):
    # Correctness only: no performance selection from these small-MN cases.
    result = old_validate(driver)
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from benchmark_o78_fused_prepare import Case
    from roof_reduction_validation import reference_fp64
    extra = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.manual_seed(20261007)
        m, n, k = 128, 256, 4096
        rows = torch.arange(m, device='cuda')[:, None]
        groups = torch.arange(32, device='cuda')[None, :]
        exponents = ((rows * 7 + groups * 3) % 5 - 2).float()
        a = (torch.randn(m, 32, 128, device='cuda') * .4 * torch.exp2(exponents)[:, :, None]).reshape(m,k).half()
        w = (torch.randn(n, k, device='cuda') * .1).half()
        for variant in ('o7', 'o8'):
            wf, af = mf.VARIANTS[variant]
            ws, acs = mf.quantize_source(w, wf), mf.quantize_source(a, af)
            old = native._benchmark_mixed(variant, 'compute_only', ws, acs, 0, 1, 2,
                                         '64x128x256', 'group_major', 59, 5)
            case = Case(variant, ws, acs, old)
            guard = driver.prepare(case)
            factors = case.state['af']
            assert guard['integer_ctas'] > 0
            assert bool((factors[1:] != factors[:-1]).any())
            assert bool((factors[:,1:] != factors[:,:-1]).any())
            semantic = reference_fp64(variant, (*old['converted_activation'], *old['converted_weight']))
            expected, _ = driver.run(case, 0, 'compute_only', 0, 1, 2)
            expected = expected.clone()
            output, _ = driver.run(case, 1, 'compute_only', 0, 1, 2)
            assert torch.equal(output.view(torch.int32), expected.view(torch.int32))
            torch.testing.assert_close(output.double(), semantic, rtol=1e-3, atol=1e-3)
            extra.append(dict(variant=variant, shape=[m,n,k], nondefault_stream=True,
                activation_factors_vary_across_rows_and_groups=True,
                unique_activation_factors=int(factors.unique().numel()), bitwise_v78=True,
                finite_fp32=bool(torch.isfinite(output).all()), semantic_tolerance_passed=True, **guard))
        stream.synchronize()
    result.update(activation_panel_mapping_checks=extra)
    return result


def main():
    full_sample_args()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--output', type=Path, required=True)
    pre.add_argument('--cubins', type=Path, required=True)
    args, _ = pre.parse_known_args()
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        raise ValueError('fresh project output required')
    review = out.with_name(out.name + '_preexecution_review.json')
    if review.exists():
        raise ValueError('refuse to overwrite pre-execution review')
    receipt = checked(args.cubins)
    runtime_sources = (Path(__file__), ROOT/'scripts/benchmark_o78_coefficient_probe.py',
                       ROOT/'scripts/benchmark_o78_eight_chain_probe.py')
    out.parent.mkdir(parents=True, exist_ok=True)
    review.write_text(json.dumps(dict(**receipt,
        review_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        runtime_sources={str(p.relative_to(ROOT)):sha(p) for p in runtime_sources},
        review_precedes_this_process_candidate_GPU_execution=True,
        production_default_changed=False), indent=2) + '\n')
    paired_main(driver_cls=Driver, default_gpu_build=Path('reports/o378_roof_v73_codegen'),
        labels=('v78_eight_chain_same_v73_preparation', 'v127_activation_factor_panel_same_v73_preparation'),
        experiment='activation_factor_panel', banner='ACTIVATION PANEL', contract=timing_contract,
        description=__doc__, validation_fn=validate)


if __name__ == '__main__':
    main()
