#!/usr/bin/env python3
"""v99: direct24 paired streaming-output test, same online preparation.

Use --kind o3 with --codegen; --kind o78 with --cubins. All remaining arguments
are the established O3 or O7/O8 paired protocol, including four timing modes.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys

from probe_output_streaming_codegen import ROOT, CONFIG, checked


def resources(lib, check, handle, kind, symbol):
    values = (ct.c_int*4)()
    check(lib.roof_probe_resources(handle, values))
    if values[0] > 168 or values[2] != 128 or values[3] < 3:
        raise ValueError('preserve 168-GPR / 128-thread / three-CTA capacity gates')
    cfg = CONFIG[kind]
    return dict(registers_per_thread=values[0], local_size_bytes=values[1], threads=values[2],
        active_blocks_per_sm=values[3], shared_memory_bytes=cfg['shared'], kernel_symbol=symbol,
        cta_tile=[64,128,128], pipeline_stages=cfg['stages'], cta_order_group_m=cfg['group_m'],
        partial_registers=32, source_max_chains=8,
        integer_output_policy='streaming_cs' if symbol == cfg['symbol'] else 'default_wb',
        fallback_output_policy='unchanged_default_wb')


def o3_main():
    import benchmark_o3_eight_chain_probe as protocol
    from benchmark_o3_grouped_cta import validate
    from benchmark_o78_grouped_cta import full_sample_args
    from roof_full_pipeline_probe import Pipeline as FullPipeline, build as full_build
    cfg = CONFIG['o3']

    def build(output, codegen):
        receipt = checked(codegen, 'o3')
        if not receipt['worth_runtime_validation']:
            raise ValueError('negative compile gate')
        control = ROOT/cfg['baseline']/(cfg['old_stem']+'.cubin')
        if hashlib.sha256(control.read_bytes()).hexdigest() != receipt['baseline_cubin_sha256']:
            raise ValueError('v89 control cubin drift')
        built = full_build(output, ROOT/'reports/o378_roof_v59', ROOT/'reports/o378_roof_v61',
                           ROOT/'runs/o378_roof_v60_screen/build')
        (output/'output_streaming_build.json').write_text(json.dumps(dict(codegen=receipt,
            comparison='v89_vs_v99_output_streaming', conversion_candidate=2,
            preparation_identical=True, production_default_changed=False,
            runtime_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')
        return (*built, control, codegen/(cfg['stem']+'.cubin'))

    class Pipeline(FullPipeline):
        def __init__(self, *built):
            self.streaming_handles = {}
            super().__init__(*built[:-2])
            try:
                for policy, (cubin, symbol) in enumerate(zip(built[-2:], (cfg['control'], cfg['symbol']))):
                    handle = ct.c_void_p()
                    self.check(self.lib.roof_probe_open(str(cubin.resolve()).encode(), symbol.encode(),
                        cfg['shared'], ct.byref(handle)))
                    self.streaming_handles[policy] = handle
                    self.resources['v89' if policy == 0 else 'v99'] = resources(
                        self.lib, self.check, handle, 'o3', symbol)
            except Exception:
                self.close()
                raise

        def close(self):
            for handle in getattr(self, 'streaming_handles', {}).values():
                self.check(self.lib.roof_probe_close(handle))
            self.streaming_handles = {}
            super().close()

        def run_four(self, policy, *args, **kwargs):
            if policy not in (0,1):
                raise ValueError('paired policy0/1 required')
            old = self.device
            try:
                self.device = self.streaming_handles[policy]
                result = super().run_four(1, *args, **kwargs)
            finally:
                self.device = old
            result['kernel'].update(self.resources['v89' if policy == 0 else 'v99'],
                gemm_tune='v89_grouped_eight' if policy == 0 else 'v99_output_streaming')
            return result

    full_sample_args()
    if '--codegen' not in sys.argv:
        sys.argv.extend(['--codegen', 'reports/o378_roof_v99_o3_codegen'])
    protocol.build, protocol.Pipeline, protocol.validate = build, Pipeline, validate
    protocol.main()
    output = Path(sys.argv[sys.argv.index('--output')+1])
    environment = json.loads((output/'environment.json').read_text())
    environment.update(scope='O3 v89 vs v99 output store policy; identical conversion2/GPU guard',
        control=cfg['control'], candidate=cfg['symbol'], no_small_performance_screen=True,
        runtime_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'environment.json').write_text(json.dumps(environment, indent=2, allow_nan=False)+'\n')


def o78_main():
    import benchmark_o78_eight_chain_probe as eight
    from benchmark_o78_coefficient_probe import main as paired_main
    from benchmark_o78_grouped_cta import full_sample_args, validate
    cfg = CONFIG['o78']

    def contract(mode, inner):
        result = eight.timing_contract(mode, inner)
        result.update(comparison='v78_vs_v99_output_streaming_same_v73_preparation',
                      new_preparation_or_layout=False)
        return result

    class Driver(eight.Driver):
        def __init__(self, library, baseline, candidate):
            receipt = checked(candidate, 'o78')
            if not receipt['worth_runtime_validation']:
                raise ValueError('negative compile gate')
            control = ROOT/cfg['baseline']
            super().__init__(library, baseline, control)
            if self.codegen['eight_chain']['build']['cubin_sha256'] != receipt['baseline_cubin_sha256']:
                self.close()
                raise ValueError('v78 control cubin drift')
            self.handles[0], self.resources[0] = self.handles[1], dict(self.resources[1])
            try:
                handle = ct.c_void_p()
                self.check(self.lib.roof_probe_open(str((candidate/(cfg['stem']+'.cubin')).resolve()).encode(),
                    cfg['symbol'].encode(), cfg['shared'], ct.byref(handle)))
                self.handles[1] = handle
                self.resources[1] = resources(self.lib, self.check, handle, 'o78', cfg['symbol'])
                self.resources[0].update(integer_output_policy='default_wb', partial_registers=32, source_max_chains=8)
                self.codegen = dict(v78=self.codegen, output_streaming=receipt,
                    runtime_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            except Exception:
                self.close()
                raise

    full_sample_args()
    paired_main(driver_cls=Driver, default_gpu_build=Path('reports/o378_roof_v73_codegen'),
        labels=('v78_eight_chain_same_v73_preparation', 'v99_output_streaming_same_v73_preparation'),
        experiment='output_streaming', banner='OUTPUT STREAMING', contract=contract,
        description=__doc__, validation_fn=validate)


if __name__ == '__main__':
    if '--kind' not in sys.argv:
        raise SystemExit('--kind o3/o78 required')
    index = sys.argv.index('--kind')
    kind = sys.argv[index+1]
    del sys.argv[index:index+2]
    if kind == 'o3':
        o3_main()
    elif kind == 'o78':
        o78_main()
    else:
        raise SystemExit('O3 or O7/O8 required')
