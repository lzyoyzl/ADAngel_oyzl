#!/usr/bin/env python3
"""v84 vs v78: three-slot integer pipeline, identical v73 online preparation."""
import ctypes as ct
import hashlib
import json
from pathlib import Path

import benchmark_o78_eight_chain_probe as eight
from benchmark_o78_coefficient_probe import main as paired_main
from probe_o78_three_stage_codegen import ROOT, SYMBOL, CONTROL, generated_header


def checked(directory):
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    r = json.loads((directory / 'codegen.json').read_text())
    if digest(directory / 'o78_three_stage.cubin') != r['cubin_sha256']:
        raise ValueError('cubin drift')
    for path, sha in r['sources'].items():
        if digest(ROOT / path) != sha:
            raise ValueError('source drift: ' + path)
    for path, sha in r['artifact_sha256'].items():
        if digest(directory / path) != sha:
            raise ValueError('artifact drift: ' + path)
    if (directory / 'o78_three_stage_generated.cuh').read_text() != generated_header(
            (ROOT / 'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()):
        raise ValueError('generated header drift')
    if not r['control_comparison']['passed'] or r['production_default_changed']:
        raise ValueError('control/default changed')
    sass = (directory / 'o78_three_stage.sass').read_text()
    for symbol in (CONTROL, SYMBOL):
        counts = eight.merged_mma_counts(sass, r['liveness'][symbol], symbol)
        if counts != dict(u4_total=32, u4_zero_c=0, s4_total=32, s4_zero_c=16):
            raise ValueError('eight-chain MMA work changed')
        loop = next(l for l in r['liveness'][symbol]['loops'] if l['kind'] == 'integer')
        if loop['opcode_counts'].get('LDSM.16.M88.4') != 16:
            raise ValueError('operand load work changed')
    return r


class Driver(eight.Driver):
    def __init__(self, library, baseline, candidate):
        receipt = checked(candidate)
        super().__init__(library, baseline, ROOT / 'reports/o378_roof_v78_codegen')
        self.handles[0] = self.handles[1]
        self.resources[0] = dict(self.resources[1])
        # Retain the exact v67 handle2; close() deduplicates retained ownership.
        try:
            handle = ct.c_void_p()
            self.check(self.lib.roof_probe_open(str((candidate / 'o78_three_stage.cubin').resolve()).encode(),
                SYMBOL.encode(), 51456, ct.byref(handle)))
            self.handles[1] = handle
            values = (ct.c_int * 4)()
            self.check(self.lib.roof_probe_resources(handle, values))
            if values[2] != 128 or values[3] < 3:
                raise ValueError('three-stage candidate did not retain three-CTA capacity')
            self.resources[1] = dict(registers_per_thread=values[0], local_size_bytes=values[1],
                threads=values[2], active_blocks_per_sm=values[3], shared_memory_bytes=51456,
                cta_tile=[64, 128, 128], pipeline_stages=3, fallback_pipeline_stages=2, kernel_symbol=SYMBOL)
            self.codegen = dict(previous=self.codegen, three_stage=receipt)
        except Exception:
            self.close()
            raise


def timing_contract(mode, inner):
    result = eight.timing_contract(mode, inner)
    result['comparison'] = 'v78_two_stage_vs_v84_three_stage_same_v73_preparation'
    return result


if __name__ == '__main__':
    paired_main(driver_cls=Driver, default_gpu_build=Path('reports/o378_roof_v73_codegen'),
        labels=('v78_two_stage_integer', 'v84_three_stage_integer'), experiment='three_stage_integer',
        banner='THREE STAGE INTEGER', contract=timing_contract, description=__doc__)
