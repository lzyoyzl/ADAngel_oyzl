#!/usr/bin/env python3
"""v126 packed E4M3->Q8, against best measured LUT A and packed NV4 W.

No GEMM changes. Preserve v73 metadata/guards/Event contract. One fixed
decoder: exact four-byte RNE, DP4A norm, no per-element float/shuffle/LUT.
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import probe_nv4_swar_codegen as weight
import probe_nv6_swar_codegen as fp6
import probe_mx8_warp_lut_codegen as lut
from analyze_o78_row_fused_codegen import entries
from compare_a100_codegen import compare

ROOT=Path(__file__).resolve().parents[1]
BASELINE=ROOT/'reports/o378_roof_v106_codegen'
WEIGHT=ROOT/'reports/o378_roof_v118_codegen'


def scalar(code):
    return lut.legacy_q(code)


def reference(word):
    values=[scalar((word>>(8*j))&255) for j in range(4)]
    return sum((v&255)<<(8*j) for j,v in enumerate(values)),sum(v*v for v in values)


def swar(word):
    if isinstance(word,bool) or not isinstance(word,int) or not 0<=word<=0xffffffff:
        raise ValueError('uint32 packed word required')
    lsb=0x01010101
    mant=(word&0x07070707)|0x08080808
    e0=((word>>3)&lsb)*255;e1=((word>>4)&lsb)*255
    e2=((word>>5)&lsb)*255;e3=((word>>6)&lsb)*255
    q0=((mant+0x07070707)>>4)&lsb
    q1=((mant+0x04040404)>>3)&0x03030303
    q2=((mant+lsb+((mant>>2)&lsb))>>2)&0x07070707
    q3=((mant+((mant>>1)&lsb))>>1)&0x0f0f0f0f
    small=(((q0&~e0)|(q1&e0))&~e1)|(((q2&~e0)|(q3&e0))&e1)
    large01=(mant&~e0)|((mant<<1)&e0)
    large=(large01&~e1)|((large01<<2)&e1)
    magnitude=((small&~e2)|(large&e2))&e3
    sign=(word>>7)&lsb
    q=((magnitude^(sign*127))+sign)^(sign<<7)
    values=[((q>>(8*j))&255) for j in range(4)]
    return q,sum((v if v<128 else v-256)**2 for v in values)


def generated_header():
    # Exactly the same row/group launch, IO, metadata, norms and guard ABI.
    text=fp6.generated_header()
    for old,new in (('nv6_row_swar_probe','mx8_row_swar_probe'),
            ('nv6_swar_probe','mx8_swar_probe'),
            ('adangel_sm80_nv6_swar_metadata','adangel_sm80_mx8_swar_metadata'),
            ('"nv6_swar_conversion.cuh"','"mx8_swar_conversion.cuh"')):
        text=text.replace(old,new)
    return once(text,'if constexpr(Format==Kind::Nv6) {','if constexpr(Format==Kind::Mx8) {')


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('source boundary drift: '+old)
    return text.replace(old,new)


def generated_host():
    text=fp6.generated_host().replace('v123','v126').replace('nv6','mx8').replace('Nv6Swar','Mx8Swar')
    text=once(text,'#include "roof_o78_row_fused_prepare.cu"',
        '#include "nv4_swar_prepare.cu"\n#include "mx8_warp_lut_generated.cuh"')
    text=text.replace('Mx8SwarOnline:RowFusedOnline','Mx8SwarOnline:Nv4SwarOnline')
    text=once(text,'using RowFusedOnline::RowFusedOnline;','using Nv4SwarOnline::Nv4SwarOnline;')
    text=once(text,'void weight(bool) {RowFusedOnline::weight(true);}',
        'void weight(bool) {Nv4SwarOnline::weight(true);}')
    text=text.replace('adangel_sm80_mx8_swar_metadata<Kind::Nv6,16>',
                      'adangel_sm80_mx8_swar_metadata<Kind::Mx8,16>')
    begin=text.index('  void packed_activation() {');end=text.index('\n  void activation(bool candidate)',begin)
    packed=text[begin:end]
    lookup=packed.replace('packed_activation','lookup_activation').replace(
        'mx8_row_swar_probe::adangel_sm80_mx8_swar_metadata',
        'mx8_warp_lut_probe::adangel_sm80_row_warp_lut_metadata')
    text=text[:end]+'\n'+lookup+text[end:]
    text=once(text,'if(!candidate || variant!=8){RowFusedOnline::activation(true);return;}\n'
        '    if(variant==7)convert_metadata<Kind::Mx8>(sa,true,a_mult);\n'
        '    else packed_activation();',
        'if(variant!=7){RowFusedOnline::activation(true);return;}\n'
        '    if(candidate)packed_activation();else lookup_activation();')
    text=once(text,'const int q=vector_probe::mx8((word>>(8*j))&255u);',
        'const int q=__float2int_rn(__fmul_rn(vector_probe::e4m3((word>>(8*j))&255u),.25f));')
    # Actual resources compare the existing best lookup, not the slower scalar A.
    text=text.replace('row_fused_probe::adangel_sm80_row_conversion_metadata<Kind::Nv6,16>',
        'mx8_warp_lut_probe::adangel_sm80_row_warp_lut_metadata<Kind::Mx8,16>')
    return text


def find_mx8_entry(text,name):
    # roof_fused_conversion_api.h: Nv4=0, Mx8=1, Hif4=2, Nv6=3.
    # Match the format and vector width, never a similarly named probe.
    result=[(s,e) for s,e in entries(text).items()
            if name in s and 'GroupedSourceKindE1ELi16E' in s]
    if len(result)!=1:raise ValueError('one MX8/Elements16 entry required: '+name)
    return result[0]


def audit(directory):
    before=(BASELINE/'prepare.sass').read_text();after=(directory/'prepare.sass').read_text()
    old_symbol,a=find_mx8_entry(before,'adangel_sm80_row_warp_lut_metadata')
    symbol,b=find_mx8_entry(after,'adangel_sm80_mx8_swar_metadata')
    comparison=compare(before,after,'^'+re.escape(old_symbol)+'$')
    weight_text=(WEIGHT/'prepare.sass').read_text()
    weights=[s for s in entries(weight_text) if 'adangel_sm80_row_swar_metadata' in s and 'GroupedSourceKindE0E' in s]
    if len(weights)!=1:raise ValueError('one unchanged packed NV4 weight kernel required')
    wc=compare(weight_text,after,'^'+re.escape(weights[0])+'$')
    resources={s:dict(registers=int(r),stack=int(stack),shared=int(shared),local=int(local))
        for s,r,stack,shared,local in re.findall(
        r' Function (\S+):\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)',
        (directory/'resources.txt').read_text())}
    lib=ct.CDLL(str((directory/'libo78_gpu_prepare.so').resolve()))
    fn=lib.roof_mx8_swar_resources;fn.argtypes=[ct.POINTER(ct.c_int)];fn.restype=ct.c_int
    values=(ct.c_int*10)();err=fn(values)
    if err:raise RuntimeError('CUDA resource query '+str(err))
    runtime={name:dict(zip(('registers','local','shared','max_threads','active_blocks_per_sm'),
        list(values)[offset:offset+5])) for name,offset in (('control',0),('candidate',5))}
    for name,sym in (('control',old_symbol),('candidate',symbol)):
        if any(runtime[name][k]!=resources[sym][k] for k in ('registers','local','shared')):
            raise ValueError('binary/runtime resource mismatch')
    r=resources[symbol];ops=b['opcode_counts'];reduction=1-b['instructions']/a['instructions']
    local=sum(v for k,v in ops.items() if k.startswith(('LDL','STL')))
    dots=sum(v for k,v in ops.items() if k.startswith('IDP.4A.S8.S8'))
    barrier=sum(v for k,v in ops.items() if k.startswith('BAR.SYNC'))
    gate=dict(control_encoding_unchanged=comparison['passed'],weight_encoding_unchanged=wc['passed'],
        work_reduction_at_least5pct=reduction>=.05,registers_at_most32=r['registers']<=32,
        no_local=r['stack']==r['local']==local==0,same_shared=r['shared']==256,
        same_residency=runtime['control']['active_blocks_per_sm']==runtime['candidate']['active_blocks_per_sm']==8,
        four_native_DP4A=dots==4,same_barriers=barrier==1)
    return dict(scope='v126_compile_and_resource_gate_not_performance',
        control=dict(symbol=old_symbol,**resources[old_symbol],**a),candidate=dict(symbol=symbol,**r,**b),
        old_controls=comparison,weight_control=wc,runtime_resources=runtime,
        static_instruction_reduction_fraction=reduction,gate=gate,worth_runtime_validation=all(gate.values()),
        no_candidate_kernel_launched=True)


def checked(directory):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((directory/'build.json').read_text())
    for name,value in r['sources'].items():
        if sha(ROOT/name)!=value:raise ValueError('source drift: '+name)
    for name,value in r['artifact_sha256'].items():
        if sha(directory/name)!=value:raise ValueError('artifact drift: '+name)
    for name,value in generated_files().items():
        if (directory/name).read_text()!=value:raise ValueError('generated source drift: '+name)
    if r['GEMM_modified'] or r['production_default_changed']:raise ValueError('scope drift')
    return r


def generated_files():
    return {'mx8_swar_generated.cuh':generated_header(),'mx8_swar_prepare.cu':generated_host(),
        'mx8_warp_lut_generated.cuh':lut.generated_header(),
        'nv4_swar_generated.cuh':weight.generated_header(),'nv4_swar_prepare.cu':weight.generated_host()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh project output required')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    names=set()
    for root in (BASELINE,WEIGHT):
        old=json.loads((root/'build.json').read_text())
        for name,value in old['sources'].items():
            if sha(ROOT/name)!=value:raise ValueError('baseline source drift: '+name)
        if sha(root/'libo78_gpu_prepare.so')!=old['driver_sha256']:raise ValueError('baseline binary drift')
        names.update(old['sources'])
    names.update(('csrc/sm80/mx8_swar_conversion.cuh','csrc/sm80/nv6_swar_conversion.cuh',
        'scripts/probe_mx8_swar_codegen.py','scripts/probe_nv6_swar_codegen.py'))
    cuda=Path('/usr/local/cuda-12.8/bin');version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    if 'V12.8.93' not in version:raise ValueError('pinned CUDA12.8.93 required')
    out.mkdir(parents=True)
    receipt=dict(scope='v126_O7_packed_MXFP8_conversion_against_v106_A_plus_v118_W',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(names)},commands=[],nvcc=version,
        GEMM_modified=False,production_default_changed=False,no_small_performance_screen=True,
        quantization='E4M3_to_Q8_Fminus2_RNE_unchanged',metadata_and_Event_timing='v73_unchanged')
    for name,value in generated_files().items():(out/name).write_text(value)
    lib=out/'libo78_gpu_prepare.so'
    def run(cmd,name):
        receipt['commands'].append(cmd)
        with (out/name).open('w') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
            env=dict(os.environ,TMPDIR=str(ROOT/'tmp')),check=True)
    run([str(cuda/'nvcc'),'-O3','-std=c++17','-lineinfo','-arch=sm_80','-shared','-Xcompiler=-fPIC',
        '-Xptxas=-v','-I'+str(ROOT/'csrc/sm80'),str(out/'mx8_swar_prepare.cu'),'-lcuda','-o',str(lib)],'build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(lib)],'prepare.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(lib)],'resources.txt')
    receipt.update(audit=audit(out),driver_sha256=sha(lib))
    receipt['artifact_sha256']={f.name:sha(f) for f in out.iterdir() if f.is_file()}
    (out/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    checked(out);print(json.dumps(receipt['audit'],indent=2),flush=True)


if __name__=='__main__':main()
