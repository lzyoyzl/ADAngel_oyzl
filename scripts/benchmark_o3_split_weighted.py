#!/usr/bin/env python3
"""v125 full24 pair after explicit small-spill resource review.

Keep the original zero-hot-local compile gate FALSE; do not rewrite it. The
user permits small spill if faster/correct. This independent investment review
requires actual3-CTA capacity and the observed homogeneous two-MMA chains. No
latency was collected before the review; no adjacent scheduling variants.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys

import benchmark_o3_eight_chain_probe as protocol
from benchmark_o3_grouped_cta import validate
from benchmark_o78_grouped_cta import full_sample_args
from probe_split_weighted_codegen import ROOT,CONTROL,SYMBOL,STEM,generated_header,cost_gate,split_chain_trace
from inspect_o78_register_liveness import analyze
from roof_full_pipeline_probe import Pipeline as FullPipeline, build as full_build

sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()


def checked(codegen):
    r=json.loads((codegen/'codegen.json').read_text())
    for name,digest in r['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('source drift: '+name)
    for name,digest in r['artifact_sha256'].items():
        if sha(codegen/name)!=digest:raise ValueError('artifact drift: '+name)
    if (codegen/(STEM+'_generated.cuh')).read_text()!=generated_header():
        raise ValueError('generated header drift')
    live={s:analyze((codegen/'liveness.txt').read_text(),s) for s in (CONTROL,SYMBOL)}
    chain=split_chain_trace((codegen/(STEM+'.sass')).read_text(),SYMBOL,live[SYMBOL])
    if live!=r['liveness'] or chain!=r['chains']:raise ValueError('audit replay drift')
    gate=cost_gate(live[CONTROL],live[SYMBOL],chain);gate['control_encoding_unchanged']=True
    if gate!=r['cost_gate'] or not r['control_comparison']['passed']:
        raise ValueError('original gate/control drift')
    if r['changed_semantics'] or r['production_default_changed'] or r['conversion_changed']:
        raise ValueError('experiment contract drift')
    if not all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma']
               and e['all_copies_bypass_l1'] for e in r['entries'].values()):
        raise ValueError('native math/supply drift')
    counts=next(x for x in live[SYMBOL]['loops'] if x['kind']=='integer')['opcode_counts']
    # Not a correctness assertion or a post-hoc performance acceptance rule.
    # Before GPU execution, explicitly allow the4 audited loads, not stores.
    if not all(v for k,v in gate['checks'].items() if k!='hot_local') or not (
        sum(v for op,v in counts.items() if op.startswith('LDL'))<=4 and
        sum(v for op,v in counts.items() if op.startswith('STL'))==0):
        raise ValueError('small-spill investment review failed')
    return r


def build(output,codegen):
    r=checked(codegen)
    control=ROOT/'reports/o378_roof_v89_o3_codegen/o3_grouped_cta.cubin'
    if sha(control)!=r['baseline_cubin_sha256']:raise ValueError('best O3 cubin drift')
    built=full_build(output,ROOT/'reports/o378_roof_v59',ROOT/'reports/o378_roof_v61',
        ROOT/'runs/o378_roof_v60_screen/build')
    sources=(Path(__file__),ROOT/'scripts/benchmark_o3_eight_chain_probe.py',
             ROOT/'scripts/benchmark_o3_grouped_cta.py')
    (output/'split_weighted_build.json').write_text(json.dumps(dict(codegen=r,
        comparison='v89_vs_v125_independent_weighted_high_low',conversion_candidate=2,
        preparation_identical=True,production_default_changed=False,
        initial_zero_spill_gate=r['cost_gate'],
        investment_review='user allows small spill; fixed4 local reads/no local stores, require actual3CTA before launch',
        review_precedes_candidate_GPU_execution=True,
        runtime_sources={str(p.relative_to(ROOT)):sha(p) for p in sources}),indent=2)+'\n')
    return (*built,control,codegen/(STEM+'.cubin'))


class Pipeline(FullPipeline):
    def __init__(self,*built):
        self.split_handles={}
        super().__init__(*built[:-2])
        try:
            for policy,(cubin,symbol) in enumerate(zip(built[-2:],(CONTROL,SYMBOL))):
                handle=ct.c_void_p()
                self.check(self.lib.roof_probe_open(str(cubin.resolve()).encode(),symbol.encode(),50688,ct.byref(handle)))
                self.split_handles[policy]=handle
                values=(ct.c_int*4)();self.check(self.lib.roof_probe_resources(handle,values))
                if values[2]!=128 or values[3]<3:raise ValueError('actual3CTA capacity required before launch')
                self.resources['v89' if policy==0 else 'v125']=dict(registers_per_thread=values[0],
                    local_size_bytes=values[1],threads=values[2],active_blocks_per_sm=values[3],
                    shared_memory_bytes=50688,kernel_symbol=symbol,partial_registers=32)
        except Exception:self.close();raise

    def close(self):
        for handle in getattr(self,'split_handles',{}).values():self.check(self.lib.roof_probe_close(handle))
        self.split_handles={};super().close()

    def run_four(self,policy,*args,**kwargs):
        if policy not in (0,1):raise ValueError('pair policy0/1 required')
        old=self.device
        try:
            self.device=self.split_handles[policy];r=super().run_four(1,*args,**kwargs)
        finally:self.device=old
        r['kernel'].update(gemm_tune='v89_grouped_eight' if policy==0 else 'v125_split_weighted',
            kernel_symbol=CONTROL if policy==0 else SYMBOL,cta_tile=[64,128,128],pipeline_stages=3,
            cta_order_group_m=8,partial_registers=32,split_weighted_modular_mad=bool(policy))
        return r


def main():
    full_sample_args()
    if '--codegen' not in sys.argv:sys.argv.extend(['--codegen','reports/o378_roof_v125_codegen'])
    protocol.build=build;protocol.Pipeline=Pipeline;protocol.validate=validate
    protocol.main()
    out=Path(sys.argv[sys.argv.index('--output')+1]);env=json.loads((out/'environment.json').read_text())
    env.update(scope='O3 v89 vs v125 split factor-weighted updates; same conversion2 and GPU guard',
        control=CONTROL,candidate=SYMBOL,no_small_performance_screen=True,
        original_compile_gate_passed=False,small_spill_review_before_GPU_execution=True)
    (out/'environment.json').write_text(json.dumps(env,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
