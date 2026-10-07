#!/usr/bin/env python3
"""Freeze v134 compiler/CPU-layout evidence; never execute a candidate GPU."""
import hashlib
import json
from pathlib import Path,PurePosixPath
import tarfile

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'tmp/o378_v134_compile_complete.tar.gz'
EXPECTED='f6b1b5b665c1ab04c47a8f6a599fe24fc5430cf0c0fa10ab1a084d655e3d0b72'
OUT=ROOT/'docs/evidence/a100_o378_roof_v134'


def main():
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()!=EXPECTED:raise ValueError('archive SHA mismatch')
    if OUT.exists():raise ValueError('fresh evidence directory required')
    files=[]
    with tarfile.open(ARCHIVE,'r:gz') as archive:
        selected=[]
        for member in archive.getmembers():
            path=PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or not (
                member.name.startswith('reports/o378_roof_v134_codegen/') or
                member.name.startswith('reports/o378_roof_v134_layout_fixed_codegen/') or
                member.name in ('reports/o378_roof_v134_codegen','reports/o378_roof_v134_layout_fixed_codegen',
                    'tmp/o378_v134_codegen.log','tmp/o378_v134_layout_fixed_codegen.log')):
                raise ValueError('unexpected archive member: '+member.name)
            if not (member.isdir() or member.isfile()):raise ValueError('nonregular archive member')
            if member.isfile() and path.suffix in ('.json','.log','.cuh','.cu','.txt','.sass','.ptx'):
                selected.append(member)
        if not selected:raise ValueError('empty evidence')
        OUT.mkdir(parents=True)
        for member in selected:
            data=archive.extractfile(member).read();target=OUT/member.name
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            files.append(dict(path=member.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    receipt=json.loads((OUT/'reports/o378_roof_v134_layout_fixed_codegen/codegen.json').read_text())
    index=dict(scope='v134_one_address_repair_failed_compile_gate_not_GPU_result',
        archive_sha256=EXPECTED,compile_source_commit=receipt['source_commit'],
        initial_failed_source_commit='2e7f327cef9322165db4510f3d947da814ab4576',
        candidate_GPU_executed=False,new_performance_or_MSE_results=False,new_NCU_result=False,
        production_default_changed=False,candidate_adopted=False,
        formal_extension_sha256='94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462',
        artifact_count=len(files),artifact_bytes=sum(f['bytes'] for f in files),files=sorted(files,key=lambda f:f['path']))
    (OUT/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps({k:v for k,v in index.items() if k!='files'},indent=2))


if __name__=='__main__':main()
