#!/usr/bin/env python3
"""v88 CuTe MMA traversal versus exact v78; identical v73 online preparation.

All native INT4, guards, source formats, and FP32 output are unchanged.
Only independent M/N atom ordering differs. Require bitwise v67 equality.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path

import benchmark_o78_eight_chain_probe as eight
from benchmark_o78_coefficient_probe import main as paired_main
from inspect_eight_chain_schedule import trace
from probe_o78_cute_traversal_codegen import ROOT, SYMBOL, CONTROL, STEM, generated_header, loop_summary


def checked(directory):
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    r = json.loads((directory / 'codegen.json').read_text())
    for path, digest in r['sources'].items():
        if sha(ROOT / path) != digest: raise ValueError('v88 source drift: ' + path)
    for path, digest in r['artifact_sha256'].items():
        if sha(directory / path) != digest: raise ValueError('v88 artifact drift: ' + path)
    if sha(directory / (STEM + '.cubin')) != r['cubin_sha256']:
        raise ValueError('v88 cubin drift')
    header = directory / (STEM + '_generated.cuh')
    if sha(header) != r['generated_header_sha256'] or header.read_text() != generated_header(
            (ROOT / 'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()):
        raise ValueError('generated traversal body drift')
    if r['production_default_changed'] or r['changed_semantics'] or not r['control_comparison']['passed']:
        raise ValueError('control/default/semantics drift')
    if set(r['entries']) != {CONTROL, SYMBOL} or not all(e['native_u4_s4'] and e['native_s4_s4']
            and not e['int8_mma'] and e['all_copies_bypass_l1'] for e in r['entries'].values()):
        raise ValueError('same-entry native INT4/copy audit failed')
    sass = (directory / (STEM + '.sass')).read_text()
    for symbol in (CONTROL, SYMBOL):
        live = r['liveness'][symbol]
        info = loop_summary(sass, symbol, live)
        if info != r['loop_summary'][symbol] or trace(sass, symbol, live) != r['schedule'][symbol]:
            raise ValueError('encoded loop or MMA dependency drift')
        if info['mma_count'] != 64 or info['opcode_counts'].get('LDSM.16.M88.4') != 16:
            raise ValueError('necessary MMA/operand work changed')
    return dict(build=r, runtime_source_sha256={p: sha(ROOT / p) for p in (
        'scripts/benchmark_o78_cute_traversal.py', 'scripts/benchmark_o78_coefficient_probe.py',
        'scripts/benchmark_o78_eight_chain_probe.py', 'scripts/benchmark_o78_fullk_gpu_prepare.py')})


class Driver(eight.Driver):
    def __init__(self, library, baseline, candidate):
        receipt = checked(candidate)
        super().__init__(library, baseline, candidate.parent / 'o378_roof_v78_codegen')
        if self.codegen['eight_chain']['build']['cubin_sha256'] != receipt['build']['baseline_cubin_sha256']:
            self.close(); raise ValueError('actual v78 control identity drift')
        # Policy2 remains the original v67; policy0 uses the old v78 cubin.
        self.handles[0] = self.handles[1]
        self.resources[0] = dict(self.resources[1])
        try:
            handle = ct.c_void_p()
            self.check(self.lib.roof_probe_open(str((candidate / (STEM + '.cubin')).resolve()).encode(),
                                                SYMBOL.encode(), 34304, ct.byref(handle)))
            self.handles[1] = handle
            values = (ct.c_int * 4)(); self.check(self.lib.roof_probe_resources(handle, values))
            self.resources[1] = dict(registers_per_thread=values[0], local_size_bytes=values[1], threads=values[2],
                active_blocks_per_sm=values[3], shared_memory_bytes=34304, cta_tile=[64,128,128],
                pipeline_stages=2, kernel_symbol=SYMBOL)
            self.codegen = dict(v78=self.codegen, cute_traversal=receipt)
        except Exception:
            self.close(); raise


def timing_contract(mode, inner):
    r = eight.timing_contract(mode, inner)
    r.update(comparison='v78_vs_v88_CuTe_traversal_same_v73_preparation', new_preparation_or_layout=False)
    return r


if __name__ == '__main__':
    paired_main(driver_cls=Driver, default_gpu_build=Path('reports/o378_roof_v73_codegen'),
        labels=('v78_eight_chain_same_v73_preparation', 'v88_CuTe_traversal_same_v73_preparation'),
        experiment='CuTe_MMA_traversal', banner='CUTE TRAVERSAL', contract=timing_contract, description=__doc__)
