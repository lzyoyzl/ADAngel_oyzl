#!/usr/bin/env python3
"""Freeze v140 compile-only evidence; do not execute any GPU candidate."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'tmp/o378_v140_compile_complete.tar.gz'
EXPECTED = 'ed29830f6ac87a24a8ca94e4166e6ad26b12319183a38f749060b72fdfe8851a'
OUT = ROOT / 'docs/evidence/a100_o378_roof_v140'


def main():
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError('archive SHA mismatch')
    if OUT.exists():
        raise ValueError('fresh evidence destination required')
    files = []
    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        selected = []
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or not (
                member.name.startswith('reports/o378_roof_v140_codegen/') or
                member.name in ('reports/o378_roof_v140_codegen', 'tmp/o378_v140_codegen.log')):
                raise ValueError('unexpected member: ' + member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError('nonregular member')
            if member.isfile() and path.suffix in ('.json', '.log', '.cu', '.cuh', '.txt', '.sass', '.ptx'):
                selected.append(member)
        if not selected:
            raise ValueError('empty evidence')
        receipt = json.load(archive.extractfile('reports/o378_roof_v140_codegen/codegen.json'))
        binary = archive.extractfile('reports/o378_roof_v140_codegen/o3_ring_counter.cubin').read()
        if hashlib.sha256(binary).hexdigest() != receipt['cubin_sha256']:
            raise ValueError('archived cubin identity mismatch')
        OUT.mkdir(parents=True)
        for member in selected:
            data = archive.extractfile(member).read()
            target = OUT / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            files.append(dict(path=member.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    index = dict(scope='v140_recurrent_slot_failed_compile_gate_not_GPU_result',
        archive_sha256=EXPECTED, compile_source_commit=receipt['source_commit'],
        candidate_GPU_executed=False, new_performance_or_MSE_results=False, new_NCU_result=False,
        production_default_changed=False, candidate_adopted=False,
        formal_extension_sha256='94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462',
        artifact_count=len(files), artifact_bytes=sum(f['bytes'] for f in files),
        files=sorted(files, key=lambda f: f['path']))
    (OUT / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps({k: v for k, v in index.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
