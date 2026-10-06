#!/usr/bin/env python3
"""v95: four existing warps, per-slot SM80 mbarriers, no fifth producer.

Warp0 produces and computes. Full counts32 async completions; empty counts
128 consumer arrivals. Slots cannot be overwritten until all readers exit.
Keep best CTA/math/copies/preparation/fallback. Compile gate before full24.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_grouped_cta_codegen import generated_headers as grouped_headers
from probe_o78_eight_chain_codegen import generated_header as eight_header
from probe_roof_fullk_integer_codegen import static_entries

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=('https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/'
           'index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive')
CONFIG={
    'o3':dict(control='adangel_roof_o3_grouped_cta_candidate',
        symbol='adangel_roof_o3_slot_pipeline_candidate',stem='o3_slot_pipeline',
        baseline='reports/o378_roof_v89_o3_codegen',cubin='o3_grouped_cta.cubin',
        shared=50816,stages=3,old_shared=50688),
    'o78':dict(control='adangel_roof_o78_eight_chain_candidate',
        symbol='adangel_roof_o78_slot_pipeline_candidate',stem='o78_slot_pipeline',
        baseline='reports/o378_roof_v78_codegen',cubin='o78_eight_chain.cubin',
        shared=34432,stages=2,old_shared=34304),
}


def baseline_header(kind):
    if kind=='o3':return grouped_headers('o3')[0]
    if kind=='o78':return eight_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    raise ValueError('O3 or O7/O8 required')


def generated_header(kind):
    cfg=CONFIG[kind];stages=cfg['stages'];text=baseline_header(kind)
    # Preserve the original packed layout and fragment/scale math verbatim.
    text=text.replace('\n};\nstatic_assert(sizeof(Storage)',
        f'\n  alignas(8) uint64_t full[{stages}],empty[{stages}];\n}};\nstatic_assert(sizeof(Storage)',1)
    text=text.replace(f'sizeof(Storage)=={cfg["old_shared"]}',f'sizeof(Storage)=={cfg["shared"]}')
    text=text.replace('static_assert(sizeof(Storage)==sizeof(C::Storage));\n','')
    start=text.index('  C::ByteLayout<64> la;C::ByteLayout<128> lb;')
    end=text.index('\n}',start)
    tile_x='roof_grouped_cta::tile().x' if kind=='o3' else 'blockIdx.x'
    tile_y='roof_grouped_cta::tile().y' if kind=='o3' else 'blockIdx.y'
    factors=(f'''  const auto* factors=reinterpret_cast<const int*>(ws);
  const unsigned first=lane*4;
  copy16(s.factor[slot]+first,factors+group*n+{tile_x}*128+first);'''
        if kind=='o3' else f'''  const unsigned first=lane*4;
  if(lane<16) copy16(s.activation_factors[slot]+first,af+group*m+{tile_y}*64+first);
  copy16(s.weight_factors[slot]+first,wf+group*n+{tile_x}*128+first);''')
    prefetch=f'''  C::ByteLayout<64> la;C::ByteLayout<128> lb;
  const unsigned lane=threadIdx.x&31u;
  // Same16KB payload, repartitioned over the existing producer/consumer warp0.
  o1_static_for<0,8>([&](auto chunk) {{
    unsigned off=lane*16+chunk*512,row=off/64,col=off%64;
    const auto* src=a+group*m*64+({tile_y}*64+row)*64+col;
    copy16(s.low[slot]+la(row,col),src);
    copy16(s.high[slot]+la(row,col),src+m*(k/2));
  }});
  o1_static_for<0,16>([&](auto chunk) {{
    unsigned off=lane*16+chunk*512,row=off/64,col=off%64;
    copy16(s.weight[slot]+lb(row,col),w+group*n*64+({tile_x}*128+row)*64+col);
  }});
{factors}
  roof_slot_barrier::publish(s.full[slot]);'''
    text=text[:start]+prefetch+text[end:]
    args='a,w,ws,m,n,k' if kind=='o3' else 'a,w,af,wf,m,n,k'
    first='  prefetch(s,0,0,'+args+');'
    loop_end='    load_a(slot,a00,a01,h00,h01);' if kind=='o3' else '    cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),cute::_0{})),ld0);'
    lo,hi=text.index(first),text.index(loop_end)
    groups='groups' if kind=='o3' else 'Groups'
    text=text[:lo]+f'''  // One CTA rendezvous for initialization, none in the integer G128 loop.
  if(threadIdx.x==0) {{
    o1_static_for<0,{stages}>([&](auto slot) {{
      roof_slot_barrier::init(s.full[slot],32);
      roof_slot_barrier::init(s.empty[slot],128);
    }});
  }}
  __syncthreads();
  if(threadIdx.x<32) o1_static_for<0,{stages}>([&](auto slot) {{
    prefetch(s,slot,slot,{args});
  }});
  for(int group=0;group<{groups};++group) {{
    const int slot=group%{stages};
    const unsigned phase=(group/{stages})&1u;
    roof_slot_barrier::wait(s.full[slot],phase);
'''+text[hi:]
    tail='    });\n  }\n  '+('auto final_value=' if kind=='o3' else '// Reuse the 64 INT32')
    if text.count(tail)!=1:raise ValueError('mainloop boundary drift')
    replacement=f'''    }});
    // All128 threads release after their LAST shared-memory read.
    roof_slot_barrier::release(s.empty[slot]);
    if(threadIdx.x<32 && group+{stages}<{groups}) {{
      roof_slot_barrier::wait(s.empty[slot],phase);
      prefetch(s,slot,group+{stages},{args});
    }}
  }}
  '''+('auto final_value=' if kind=='o3' else '// Reuse the 64 INT32')
    text=text.replace(tail,replacement)
    namespace='o3_grouped_cta_experiment' if kind=='o3' else 'o78_eight_chain_experiment'
    return text.replace(namespace,kind+'_slot_pipeline_experiment')


def worth_runtime(live):
    loop=next(x for x in live['loops'] if x['kind']=='integer')
    counts=loop['opcode_counts']
    local=sum(v for op,v in counts.items() if op.startswith(('LDL','STL')))
    return (live['allocated_gpr']<=168 and local<=2
        and counts.get('IMMA.16864.S4.S4')==counts.get('IMMA.16864.U4.S4')==32
        and counts.get('LDSM.16.M88.4')==16
        and not any(v for op,v in counts.items() if op.startswith('BAR.')))


def checked(directory,kind):
    cfg=CONFIG[kind];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((directory/'codegen.json').read_text())
    for name,digest in r['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('slot-pipeline source drift: '+name)
    for name,digest in r['artifact_sha256'].items():
        if sha(directory/name)!=digest:raise ValueError('slot-pipeline artifact drift: '+name)
    if (directory/(cfg['stem']+'_generated.cuh')).read_text()!=generated_header(kind):
        raise ValueError('generated slot pipeline drift')
    if not r['control_comparison']['passed'] or r['production_default_changed'] or r['changed_semantics']:
        raise ValueError('control/default/math drift')
    if not all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] and e['all_copies_bypass_l1']
               for e in r['entries'].values()):raise ValueError('same-entry native INT4/copy failed')
    if r['worth_runtime_validation']!=worth_runtime(r['liveness'][cfg['symbol']]):
        raise ValueError('compile gate drift')
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=CONFIG,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=CONFIG[a.kind];out=a.output.resolve();baseline=ROOT/cfg['baseline']
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    prior=json.loads((baseline/'codegen.json').read_text())
    for name,digest in prior['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('best source drift: '+name)
    if sha(baseline/cfg['cubin'])!=prior['cubin_sha256']:raise ValueError('best cubin drift')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA/CUTLASS required')
    out.mkdir(parents=True)
    for f in baseline.iterdir():
        if f.suffix in ('.cu','.cuh'):(out/f.name).write_text(f.read_text())
    (out/(cfg['stem']+'_generated.cuh')).write_text(generated_header(a.kind))
    sources=set(prior['sources'])|{'scripts/probe_slot_pipeline_codegen.py',
        'csrc/sm80/roof_sm80_slot_barrier.cuh',f'csrc/sm80/roof_{a.kind}_slot_pipeline_probe.cu'}
    r=dict(scope='v95_SM80_four_existing_warps_full_empty_slot_handshake',kind=a.kind,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},commands=[],nvcc=version,cutlass_commit=commit,
        reference=REFERENCE,cta_tile=[64,128,128],threads=128,stages=cfg['stages'],shared_bytes=cfg['shared'],
        producer_warp_also_computes=True,full_arrivals=32,empty_arrivals=128,
        global_payload_copy_bytes_changed=False,changed_semantics=False,production_default_changed=False,
        baseline_cubin_sha256=prior['cubin_sha256'])
    def run(cmd,name):
        r['commands'].append(cmd)
        with (out/name).open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
        '-DADANGEL_FULLK_INTEGER=1','-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out),
        str(ROOT/f'csrc/sm80/roof_{a.kind}_slot_pipeline_probe.cu')]
    cubin=out/(cfg['stem']+'.cubin')
    run(flags+['-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+['-ptx','-o',str(out/(cfg['stem']+'.ptx'))],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],cfg['stem']+'.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/(cfg['stem']+'.sass')).read_text();ptx=(out/(cfg['stem']+'.ptx')).read_text()
    symbols={cfg['control'],cfg['symbol']}
    entries=static_entries(sass,'^(?:'+'|'.join(sorted(symbols))+')$',symbols)
    body=next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+cfg['symbol']+'('))
    for token in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32',
                  'cp.async.mbarrier.arrive.noinc.shared.b64','mbarrier.test_wait.parity.shared.b64'):
        if token not in body:raise ValueError('same-entry PTX missing: '+token)
    r.update(entries=entries,liveness={s:analyze((out/'liveness.txt').read_text(),s) for s in sorted(symbols)},
        cubin_sha256=sha(cubin),control_comparison=compare((baseline/cfg['cubin'].replace('.cubin','.sass')).read_text(),
        sass,'^'+cfg['control']+'$'))
    r['worth_runtime_validation']=worth_runtime(r['liveness'][cfg['symbol']])
    r['artifact_sha256']={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='codegen.json'}
    (out/'codegen.json').write_text(json.dumps(r,indent=2)+'\n');checked(out,a.kind)
    print(json.dumps(dict(kind=a.kind,runtime_justified=r['worth_runtime_validation'],
        candidate_liveness=r['liveness'][cfg['symbol']]),indent=2),flush=True)


if __name__=='__main__':main()
