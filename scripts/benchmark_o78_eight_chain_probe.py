#!/usr/bin/env python3
"""v78 merged-eight-chain GEMM versus v67; identical v73 online preparation.

Same actual source formats/scales/guard, not the unit-scale diagnostic.
Default four-sample cached screen. --full-modes measures online costs too.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import re

import numpy as np

import benchmark_o78_row_fused as row_fused
from benchmark_o78_coefficient_probe import main as paired_main
from probe_o78_eight_chain_codegen import ROOT, SYMBOL, CONTROL, generated_header

base = row_fused.base


def merged_mma_counts(sass, live, symbol):
    """Exact audited entry/loop only; count nonzero MMA accumulator inputs."""
    block = next(b for b in re.split(r'(?=\s*Function\s*:\s*)', sass)
                 if re.match(r'\s*Function\s*:\s*'+re.escape(symbol)+r'\s', b))
    loop = next(l for l in live['loops'] if l['kind'] == 'integer')
    first, last = int(loop['begin_pc'],16), int(loop['end_pc'],16)
    result = {'u4_total':0, 'u4_zero_c':0, 's4_total':0, 's4_zero_c':0}
    for line in block.splitlines():
        ins = re.search(r'/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*;',line)
        if not ins or not first <= int(ins[1],16) <= last:
            continue
        mma = re.search(r'IMMA\.16864\.(U4|S4)\.S4\s+.*?,\s*(RZ|R\d+)\s*$',ins[2])
        if mma:
            tag=mma[1].lower()
            result[tag+'_total']+=1
            result[tag+'_zero_c']+=int(mma[2]=='RZ')
    if result['u4_total']!=32 or result['s4_total']!=32:
        raise ValueError('unexpected integer-loop native MMA work')
    return result


def checked(directory):
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((directory/'codegen.json').read_text())
    for path, sha in r['sources'].items():
        if digest(ROOT/path)!=sha: raise ValueError('source drift: '+path)
    if digest(directory/'o78_eight_chain.cubin')!=r['cubin_sha256']:
        raise ValueError('eight-chain cubin drift')
    header=directory/'o78_eight_chain_generated.cuh'
    if digest(header)!=r['generated_header_sha256'] or header.read_text()!=generated_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()):
        raise ValueError('generated body drift')
    if not r['control_opcode_counts_match'] or not r['control_instructions_match'] or r['production_default_changed']:
        raise ValueError('control/default drift')
    assert all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] and e['all_copies_bypass_l1'] for e in r['entries'].values())
    counts={s:merged_mma_counts((directory/'o78_eight_chain.sass').read_text(),r['liveness'][s],s)
            for s in (CONTROL,SYMBOL)}
    if counts[CONTROL] != dict(u4_total=32,u4_zero_c=16,s4_total=32,s4_zero_c=16):
        raise ValueError('old independent-chain work drift')
    if counts[SYMBOL] != dict(u4_total=32,u4_zero_c=0,s4_total=32,s4_zero_c=16):
        raise ValueError('low MMA did not consume the shifted high accumulator')
    loop=next(l for l in r['liveness'][SYMBOL]['loops'] if l['kind']=='integer')
    if loop['opcode_counts']['LDSM.16.M88.4']!=16 or any(k.startswith(('LDL','STL')) for k in loop['opcode_counts']):
        raise ValueError('extra operand loads or hot spills require reassessment')
    return dict(build=r,mma_accumulator_inputs=counts)


def timing_contract(mode, inner):
    result=row_fused.timing_contract(mode,inner,1)
    result.update(gemm_cufunction_identical_between_policies=False,
                  comparison='v67_fullK_vs_v78_eight_chains_same_v73_preparation')
    return result


class Driver(row_fused.Driver):
    def __init__(self, library, baseline, candidate):
        receipt=checked(candidate)
        super().__init__(library,baseline)
        # 0 and2 retain the exact v67 function. close() deduplicates ownership.
        self.resources={p:dict(r) for p,r in self.resources.items()}
        try:
            handle=ct.c_void_p()
            self.check(self.lib.roof_probe_open(str((candidate/'o78_eight_chain.cubin').resolve()).encode(),
                                               SYMBOL.encode(),34304,ct.byref(handle)))
            self.handles[1]=handle
            values=(ct.c_int*4)();self.check(self.lib.roof_probe_resources(handle,values))
            self.resources[1]=dict(registers_per_thread=values[0],local_size_bytes=values[1],threads=values[2],
                active_blocks_per_sm=values[3],shared_memory_bytes=34304,cta_tile=[64,128,128],pipeline_stages=2,kernel_symbol=SYMBOL)
            self.codegen=dict(v67=self.codegen,eight_chain=receipt)
        except Exception:
            self.close();raise

    def run(self, case, policy, mode, warmup, repeats, inner):
        import torch
        if policy not in self.handles or mode not in base.MODES:
            raise ValueError('invalid eight-chain test policy/mode')
        if np.any(case.oracle['status_flat']==2):
            raise ValueError('invalid source must not expose an unwritten output')
        values=(ct.c_float*(4*repeats))()
        # Row-fused preparation policy1 for ALL kernels, including reference2.
        self.check(self.lib.roof_o78_gpu_benchmark(self.handles[policy],int(case.variant[1:]),1,base.MODES.index(mode),
            case.a_source,case.w_source,case.state_pointers,case.m,case.n,case.a_multiplier,case.w_multiplier,
            warmup,repeats,inner,torch.cuda.current_stream().cuda_stream,values))
        return case.state['y'],base.normalize_timings(mode,np.ctypeslib.as_array(values).reshape(4,repeats),repeats)


if __name__=='__main__':
    paired_main(driver_cls=Driver,default_gpu_build=Path('reports/o378_roof_v73_codegen'),
        labels=('v67_fullK_same_v73_preparation','v78_eight_chain_merge_same_v73_preparation'),
        experiment='eight_chain_merge',banner='EIGHT CHAIN',contract=timing_contract,description=__doc__)
