#!/usr/bin/env python3
"""v83 compile gate: reuse A over N256, no runtime/default claim or tile sweep."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from probe_o78_eight_chain_codegen import generated_header as eight_header, SYMBOL as CONTROL
from probe_roof_fullk_integer_codegen import static_entries

ROOT=Path(__file__).resolve().parents[1]
SYMBOL='adangel_roof_o78_n256_candidate'
SUBSTITUTIONS=(
    ('O3AmpereConfig<64,128,128,false,2,true,2>', 'O3AmpereConfig<64,256,128,false,2,true,2>',1),
    ('weight[2][128*64]', 'weight[2][256*64]',1),
    ('weight_factors[2][128]', 'weight_factors[2][256]',1),
    ('sizeof(Storage)==34304', 'sizeof(Storage)==51712',1),
    ('C::ByteLayout<128>', 'C::ByteLayout<256>',1),
    ('o1_static_for<0,4>([&](auto chunk)', 'o1_static_for<0,8>([&](auto chunk)',1),
    ('blockIdx.x*128', 'blockIdx.x*256',5),
    ('if(threadIdx.x<32) copy16(s.weight_factors', 'if(threadIdx.x<64) copy16(s.weight_factors',1),
    ('cute::make_shape(cute::_64{},cute::_128{})', 'cute::make_shape(cute::_64{},cute::_256{})',1),
    ('C::NibbleLayout<128>', 'C::NibbleLayout<256>',1),
    ('cute::size(acc))::value==64', 'cute::size(acc))::value==128',1),
    ('o1_static_for<0,2>([&](auto nb)', 'o1_static_for<0,4>([&](auto nb)',1),
)


def generated_header(source):
    text=eight_header(source)
    for old,new,count in SUBSTITUTIONS:
        if text.count(old)!=count:
            raise ValueError(f'source drift for {old}: {text.count(old)} != {count}')
        text=text.replace(old,new)
    return text.replace('o78_eight_chain_experiment','o78_n256_experiment').replace(
        'Reuse the 64 INT32 register slots','Reuse the 128 INT32 register slots')


def loops(sass, symbol, mma_count):
    from analyze_o3_mma_lowering import SASS_FUNCTION, SASS_INSTRUCTION, split_sections
    ops=[(int(m[1],16),m[2],m[3]) for m in SASS_INSTRUCTION.finditer(split_sections(sass,SASS_FUNCTION)[symbol])]
    result=[]
    for pc,opcode,args in ops:
        target=re.search(r'0x([0-9a-fA-F]+)',args) if opcode=='BRA' else None
        if not target or int(target[1],16)>=pc:continue
        counts=Counter(op for addr,op,_ in ops if int(target[1],16)<=addr<=pc)
        if sum(c for op,c in counts.items() if op.startswith('IMMA.'))!=mma_count:continue
        result.append(dict(kind='fp32_fallback' if counts['I2F'] else 'integer',begin=target[0],end=hex(pc),
                           instruction_count=sum(counts.values()),opcodes=dict(sorted(counts.items()))))
    if len(result)!=2 or {r['kind'] for r in result}!={'integer','fp32_fallback'}:
        raise ValueError('expected both complete G128 paths')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('reports/o378_roof_v78_codegen'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audit-existing',action='store_true',help='recover completed cubin/PTX audit, never rebuild/overwrite raw artifacts')
    p.add_argument('--compiled-commit',help='required original compilation commit with --audit-existing')
    a=p.parse_args();out=a.output.resolve()
    if (not out.is_relative_to(ROOT) or (out.exists() != a.audit_existing)
            or bool(a.compiled_commit) != a.audit_existing or (out/'codegen.json').exists()):
        p.error('fresh project output, or unaudited existing build with original compilation commit required')
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    prior=json.loads((a.baseline/'codegen.json').read_text())
    for path,sha in prior['sources'].items():
        if digest(ROOT/path)!=sha:raise ValueError('baseline source drift: '+path)
    if digest(a.baseline/'o78_eight_chain.cubin')!=prior['cubin_sha256']:raise ValueError('baseline binary drift')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    nvcc=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in nvcc or commit!='db1c288993354c88e551c40c19a8fb93a774a241':p.error('pinned toolchain required')
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    header=out/'o78_n256_generated.cuh'
    if a.audit_existing:
        if (header.read_text()!=generated_header(source)
                or (out/'o78_eight_chain_generated.cuh').read_text()!=eight_header(source)):
            raise ValueError('existing generated headers differ from current math/layout')
    else:
        out.mkdir(parents=True)
        (out/'o78_eight_chain_generated.cuh').write_text(eight_header(source))
        header.write_text(generated_header(source))
    files=set(prior['sources'])|{'scripts/probe_o78_n256_codegen.py','csrc/sm80/roof_o78_n256_probe.cu'}
    current_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    compile_commit=a.compiled_commit or current_commit
    if a.audit_existing:
        # The generator's audit handling may change; actual device sources and
        # regenerated headers must still agree with the compilation revision.
        for f in files-{'scripts/probe_o78_n256_codegen.py'}:
            old=subprocess.check_output(['git','show',compile_commit+':'+f])
            if hashlib.sha256(old).hexdigest()!=digest(ROOT/f):raise ValueError('compiled source drift: '+f)
    receipt=dict(scope='compile_gate_not_runtime_MSE_or_performance',production_default_changed=False,
        sources={f:digest(ROOT/f) for f in sorted(files)},source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        compiled_commit=compile_commit,audit_existing_artifacts=a.audit_existing,
        original_compile_script_sha256=hashlib.sha256(subprocess.check_output(['git','show',compile_commit+':scripts/probe_o78_n256_codegen.py'])).hexdigest(),
        generated_header_sha256=digest(header),baseline_cubin_sha256=prior['cubin_sha256'],
        cta_tile=[64,256,128],threads=128,stages=2,shared_bytes=51712,nvcc=nvcc,cutlass_commit=commit,commands=[])
    def run(command,name):
        receipt['commands'].append(command)
        with (out/name).open('w') as f:subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    args=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
          '-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out),str(ROOT/'csrc/sm80/roof_o78_n256_probe.cu')]
    binary=out/'o78_n256.cubin'
    if not a.audit_existing:
        run(args+['-cubin','-o',str(binary),'-Xptxas=-v'],'build.log')
        run(args+['-ptx','-o',str(out/'o78_n256.ptx')],'ptx_build.log')
        run([str(cuda/'cuobjdump'),'--dump-sass',str(binary)],'o78_n256.sass')
        run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(binary)],'resources.txt')
        try:
            run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(binary)],'liveness.txt')
        except subprocess.CalledProcessError:
            if (out/'liveness.txt').read_text().strip()!="nvdisasm fatal   : Invalid register count : '255'":raise
    live=(out/'liveness.txt').read_text()
    receipt['liveness_available']='nvdisasm fatal' not in live
    if not receipt['liveness_available'] and live.strip()!="nvdisasm fatal   : Invalid register count : '255'":
        raise ValueError('unrecognized disassembler failure')
    sass=(out/'o78_n256.sass').read_text();ptx=(out/'o78_n256.ptx').read_text()
    entries=static_entries(sass,'^'+SYMBOL+'$',{SYMBOL})
    if not all(e['native_u4_s4'] and e['native_s4_s4'] and e['all_copies_bypass_l1'] and not e['int8_mma'] for e in entries.values()):
        raise ValueError('native two-INT4/async audit failed')
    entry=next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+SYMBOL+'('))
    if not all(s in entry for s in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32')):
        raise ValueError('PTX same-entry audit failed')
    result=compare((a.baseline/'o78_eight_chain.sass').read_text(),sass,'^'+CONTROL+'$')
    if not result['passed']:raise ValueError('included v78 control machine code changed')
    receipt.update(cubin_sha256=digest(binary),entries=entries,control_comparison=result,
        loop_counts={CONTROL:loops(sass,CONTROL,64),SYMBOL:loops(sass,SYMBOL,128)},
        artifact_sha256={name:digest(out/name) for name in ('build.log','o78_n256.ptx','o78_n256.sass','resources.txt','liveness.txt')})
    (out/'codegen.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print((out/'build.log').read_text(),flush=True)
    print(json.dumps(receipt['loop_counts'],indent=2),flush=True)


if __name__=='__main__':main()
