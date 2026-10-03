#!/usr/bin/env python3
"""v87 O7-only coefficient shift versus exact v78; identical v73 preparation.

No O8 dispatch, production change, new quantization, or uncharged metadata.
Synthetic validation includes all legal shifts, fallback and invalid sources.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import re

import numpy as np

import benchmark_o78_eight_chain_probe as eight
from benchmark_o78_coefficient_probe import main as paired_main, validate as common_validate
from inspect_eight_chain_schedule import instructions, trace
from probe_o7_shift_coefficient_codegen import ROOT, SYMBOL, CONTROL, generated_header

base = eight.base


def coefficient_dependencies(sass, symbol, live):
    """Conservative static def/use: shared factors -> SHF -> final IMAD.

    This is not an execution-time model or a claim about pipeline saturation.
    """
    loop = next(x for x in live['loops'] if x['kind'] == 'integer')
    tags, shifts, updates = {}, [], []
    for pc, text in instructions(sass, symbol, loop):
        op = text.split()[0]
        dest = re.match(r'\S+\s+R(\d+)\s*,(.*)', text)
        if not dest:
            continue
        d, rest = int(dest[1]), dest[2]
        operands = [x.strip() for x in rest.split(',')]
        def origin(value):
            return frozenset().union(*(tags.get(int(r), frozenset())
                for r in re.findall(r'\bR(\d+)\b', value)))
        tag, width = origin(rest), 1
        if op.startswith('IMMA.'):
            tag, width = frozenset({'partial'}), 4
        elif op.startswith('LDSM.'):
            tag, width = frozenset({'payload'}), int(op.rsplit('.', 1)[1])
        elif op == 'LDS' or op.startswith('LDS.'):
            tag = frozenset({'scale'})
            width = 4 if '.128' in op else 2 if '.64' in op else 1
        elif op.startswith(('LDG', 'LDL', 'S2R', 'CS2R')):
            tag = frozenset({'other'})
            width = 4 if '.128' in op else 2 if '.64' in op or op == 'CS2R' else 1
        elif op == 'FLO.U32':
            if tag != {'scale'}:
                raise ValueError('shift exponent not derived solely from shared scale')
            tag = frozenset({'scale'})
        elif op == 'SHF.L.W.U32.HI':
            if tag != {'scale'} or len(operands) != 3 or operands[0] != 'RZ':
                raise ValueError('coefficient shift has unexpected inputs: ' + text)
            tag = frozenset({'coefficient'})
            shifts.append(dict(pc=hex(pc), instruction=text))
        elif op == 'IMAD' and len(operands) == 3:
            a, b = origin(operands[0]), origin(operands[1])
            if (a == {'partial'} and b == {'coefficient'}) or (b == {'partial'} and a == {'coefficient'}):
                if operands[2] != 'R' + str(d):
                    raise ValueError('partial multiply did not retain accumulator add')
                tag = frozenset({'weighted_sum'})
                updates.append(dict(pc=hex(pc), instruction=text))
        elif op.startswith('IMAD.WIDE'):
            width = 2
        for i in range(width):
            tags[d + i] = tag
    if len(shifts) != 64 or len(updates) != 64:
        raise ValueError(f'coefficient/fused-update mismatch: {len(shifts)}/{len(updates)}')
    return dict(coefficient_shifts=shifts, coefficient_imad_updates=updates,
                scope='static_def_use_not_dynamic_pipeline_or_latency')


def checked(directory):
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    r = json.loads((directory / 'codegen.json').read_text())
    for path, digest in r['sources'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('v87 source drift: ' + path)
    for path, digest in r['artifact_sha256'].items():
        if sha(directory / path) != digest:
            raise ValueError('v87 artifact drift: ' + path)
    if sha(directory / 'o7_shift_coefficient.cubin') != r['cubin_sha256']:
        raise ValueError('v87 cubin drift')
    h = directory / 'o7_shift_coefficient_generated.cuh'
    if sha(h) != r['generated_header_sha256'] or h.read_text() != generated_header(
            (ROOT / 'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()):
        raise ValueError('generated O7 body drift')
    if r['production_default_changed'] or r['supported_variant'] != 'o7' or not r['control_comparison']['passed']:
        raise ValueError('control/default/variant drift')
    if set(r['entries']) != {CONTROL, SYMBOL} or not all(e['native_u4_s4'] and e['native_s4_s4']
            and not e['int8_mma'] and e['all_copies_bypass_l1'] for e in r['entries'].values()):
        raise ValueError('same-entry native INT4/copy audit failed')
    if r['coordinate_verification'] != dict(passed=True, threads=128, outputs=8192,
            unique_rows_per_thread=4, gpu_execution=False):
        raise ValueError('CuTe row-coordinate verification failed')
    sass = (directory / 'o7_shift_coefficient.sass').read_text()
    chain_info = {}
    for symbol in (CONTROL, SYMBOL):
        live = r['liveness'][symbol]
        loop = next(x for x in live['loops'] if x['kind'] == 'integer')
        if live['allocated_gpr'] != 168 or loop['opcode_counts'].get('LDSM.16.M88.4') != 16 or any(
                op.startswith(('LDL', 'STL')) for op in loop['opcode_counts']):
            raise ValueError('resource/hot-loop work gate changed')
        chain_info[symbol] = trace(sass, symbol, live)
    return dict(build=r, chains=chain_info, dependencies=coefficient_dependencies(sass, SYMBOL, r['liveness'][SYMBOL]),
        runtime_source_sha256={name: sha(ROOT / name) for name in (
            'scripts/benchmark_o7_shift_coefficient.py', 'scripts/benchmark_o78_coefficient_probe.py',
            'scripts/benchmark_o78_fullk_gpu_prepare.py', 'scripts/inspect_eight_chain_schedule.py')})


def require_o7(case):
    if case.variant != 'o7':
        raise ValueError('v87 coefficient-shift specialization is O7-only; O8 is forbidden')


def check_factor_domain(oracle):
    operand = oracle['activation']
    accepted_rows = np.asarray(operand['row_status']) == 0
    factors = np.asarray(operand['factors'])[:, accepted_rows].astype(np.int64)
    if np.any(factors <= 0) or np.any(factors > 2**30) or np.any(factors & (factors - 1)):
        raise ValueError('accepted O7 activation factors must be exact positive powers of two <=2^30')


class Driver(eight.Driver):
    def __init__(self, library, baseline, candidate):
        receipt = checked(candidate)
        control = candidate.parent / 'o378_roof_v78_codegen'
        super().__init__(library, baseline, control)
        if self.codegen['eight_chain']['build']['cubin_sha256'] != receipt['build']['baseline_cubin_sha256']:
            self.close(); raise ValueError('actual v78 control identity drift')
        # Preserve original v67 as policy2, use exact existing v78 as policy0.
        self.handles[0] = self.handles[1]
        self.resources[0] = dict(self.resources[1])
        try:
            handle = ct.c_void_p()
            self.check(self.lib.roof_probe_open(str((candidate / 'o7_shift_coefficient.cubin').resolve()).encode(),
                SYMBOL.encode(), 34304, ct.byref(handle)))
            self.handles[1] = handle
            values = (ct.c_int * 4)(); self.check(self.lib.roof_probe_resources(handle, values))
            self.resources[1] = dict(registers_per_thread=values[0], local_size_bytes=values[1], threads=values[2],
                active_blocks_per_sm=values[3], shared_memory_bytes=34304, cta_tile=[64,128,128],
                pipeline_stages=2, kernel_symbol=SYMBOL)
            self.codegen = dict(v78=self.codegen, shift_coefficient=receipt)
        except Exception:
            self.close(); raise

    def prepare(self, case):
        require_o7(case)
        check_factor_domain(case.oracle)
        return super().prepare(case)

    def run(self, case, policy, mode, warmup, repeats, inner):
        require_o7(case)
        return super().run(case, policy, mode, warmup, repeats, inner)


def validate(driver):
    import torch
    from types import SimpleNamespace
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    result = common_validate(driver, variants=('o7',))
    # One nonzero product in the last G128: exercise accepted integer GEMM
    # for all d=0..30, including signs, rather than merely testing fallback.
    checks = []
    for d in range(31):
        ws = mf.quantize_source(torch.zeros((128,4096),device='cuda',dtype=torch.float16), mf.VARIANTS['o7'][0])
        acs = mf.quantize_source(torch.zeros((64,4096),device='cuda',dtype=torch.float16), mf.VARIANTS['o7'][1])
        ws['payload'].zero_(); ws['payload'][:, -1] = 0x20; ws['payload'][1::2, -1] = 0xa0
        ws['scale'].fill_(56); ws['tensor_scale'].fill_(1)
        acs['payload'].zero_(); acs['payload'][:, -1] = 0x48; acs['payload'][1::2, -1] = 0xc8
        acs['scale'].fill_(127); acs['scale'][:, -1] = 127 + d
        old = native._benchmark_mixed('o7', 'compute_only', ws, acs, 0, 1, 2, '64x128x256', 'group_major', 59, 5)
        case = eight.row_fused.fused.Case('o7', ws, acs, old)
        guard = driver.prepare(case)
        if guard['integer_ctas'] != guard['ctas']:
            raise ValueError('legal-shift test unintentionally fell back')
        expected = torch.full((64,128), float(4 * (1 << d)), dtype=torch.float32, device='cuda')
        expected[1::2, :] *= -1; expected[:, 1::2] *= -1
        for policy in (0,1,2):
            out, _ = driver.run(case, policy, 'compute_only', 0, 1, 2)
            assert torch.equal(out.view(torch.int32), expected.view(torch.int32)), (d, policy)
        checks.append(dict(shift=d, policies=[0,1,2], exact_analytic_fp32=True, shape=[64,128,4096], **guard))
    try:
        driver.run(SimpleNamespace(variant='o8'), 1, 'compute_only', 0, 1, 2)
    except ValueError as exc:
        assert 'O7-only' in str(exc)
    else:
        raise AssertionError('O8 was not rejected before launch')
    result.update(legal_shift_checks=checks, legal_shift_count=len(checks), o8_rejected_before_launch=True)
    return result


def timing_contract(mode, inner):
    r = eight.timing_contract(mode, inner)
    r.update(comparison='v78_vs_v87_O7_coefficient_shift_same_v73_preparation',
             supported_variant='o7', new_preparation_or_layout=False)
    return r


if __name__ == '__main__':
    paired_main(driver_cls=Driver, default_gpu_build=Path('reports/o378_roof_v73_codegen'),
        labels=('v78_eight_chain_same_v73_preparation', 'v87_O7_coefficient_shift_same_v73_preparation'),
        experiment='O7_shift_coefficient', banner='O7 SHIFT COEFFICIENT', contract=timing_contract,
        variants=('o7',), validation_fn=validate, description=__doc__)
