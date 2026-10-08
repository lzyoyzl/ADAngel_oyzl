#!/usr/bin/env python3
"""Freeze v141 compile, two complete paired runs and limited safety evidence."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

from analyze_o78_recycled_coefficient import analyze
from inspect_o78_register_liveness import analyze as liveness
from inspect_scaled_partial_overlap import inspect
from probe_o78_recycled_coefficient_codegen import CONTROL,SYMBOL,STEM,gate,dependency_audit

ROOT=Path(__file__).resolve().parents[1]
BUILD='reports/o378_roof_v141_recycled_coefficient_codegen'
EXTENSION='94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
TEXT={'.json','.jsonl','.log','.txt','.sass','.ptx','.cuh'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def verify(directory):
    build=directory/BUILD;r=read(build/'codegen.json')
    sass=(build/(STEM+'.sass')).read_text()
    lives={s:liveness((build/'liveness.txt').read_text(),s) for s in (CONTROL,SYMBOL)}
    if lives!=r['liveness']:raise ValueError('liveness evidence drift')
    deps={s:dependency_audit(sass,s,lives[s]) for s in (CONTROL,SYMBOL)}
    if deps!=r['dependency_audit'] or gate(lives[CONTROL],lives[SYMBOL],deps[SYMBOL])!=r['compile_gate']:
        raise ValueError('compile dependency/cost gate drift')
    result=dict(runs={name:analyze(directory/f'runs/{name}') for name in (
        'o378_roof_v141_paired','o378_roof_v141_paired_r2')},
        scale_overlap={s:inspect(sass,s,lives[s]) for s in (CONTROL,SYMBOL)},
        candidate_adopted=False,production_default_changed=False,
        no_new_conversion_or_end_to_end_claim=True,compile_source_commit=r['source_commit'])
    env=read(directory/'runs/o378_roof_v141_paired/environment.json')
    result['runtime_source_commit']=env['git_commit']
    if any(x['extension_sha256']!=EXTENSION for x in result['runs'].values()):
        raise ValueError('runtime extension identity changed')
    if (directory/'reports/o378_roof_v141_extension_after.txt').read_text().split()[0]!=EXTENSION:
        raise ValueError('post-test extension identity changed')
    validation=read(directory/'runs/o378_roof_v141_validation/validation.json')
    if not validation['passed'] or validation['count']!=64 or validation['edge_count']!=12:
        raise ValueError('incomplete numerical/edge validation')
    for name in ('memcheck','synccheck'):
        if read(directory/f'runs/o378_roof_v141_{name}/validation.json')!=validation:
            raise ValueError('different numerical safety suite')
        if 'ERROR SUMMARY: 0 errors' not in (build/(name+'.log')).read_text():
            raise ValueError('limited sanitizer evidence failed')
    for name in result['runs']:
        if read(directory/f'runs/{name}/validation.json')!=validation:
            raise ValueError('paired run numerical preflight differs')
        if 'RECYCLED COEFFICIENT PAIRED TEST PASSED' not in (directory/f'reports/{name}.log').read_text():
            raise ValueError('missing completion log')
    result.update(formal_extension_sha256=EXTENSION,limited_sanitizers=dict(
        memcheck='0 errors',synccheck='0 errors',
        scope='candidate_entry_filter_small_MN_K4096_not_full4096cubed'),
        synthetic_checks=64,edge_checks=12)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replay-existing',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if (out.exists() and not a.replay_existing) or not out.is_relative_to(ROOT):
        p.error('fresh project output or explicit byte-verified replay required')
    if sha(a.archive)!=a.sha256:raise ValueError('archive hash differs')
    files={};binaries={}
    with tarfile.open(a.archive,'r:gz') as archive:
        members=archive.getmembers();names=set()
        for m in members:
            path=PurePosixPath(m.name)
            if (path.is_absolute() or '..' in path.parts or m.name in names or
                    not (m.isfile() or m.isdir()) or not m.name.startswith(('reports/o378_roof_v141','runs/o378_roof_v141'))):
                raise ValueError('out-of-scope/unsafe/duplicate archive member: '+m.name)
            names.add(m.name)
        out.mkdir(parents=True,exist_ok=a.replay_existing)
        for m in members:
            if not m.isfile():continue
            data=archive.extractfile(m).read();digest=hashlib.sha256(data).hexdigest()
            suffix=PurePosixPath(m.name).suffix
            if suffix=='.cubin':binaries[m.name]=digest;continue
            if suffix not in TEXT:raise ValueError('unexpected artifact type: '+m.name)
            destination=out/m.name;destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():
                if destination.read_bytes()!=data:raise ValueError('existing evidence differs: '+m.name)
            else:destination.write_bytes(data)
            files[m.name]=digest
    build=read(out/BUILD/'codegen.json')
    for name,digest in build['artifact_sha256'].items():
        target=BUILD+'/'+name
        if (binaries if name.endswith('.cubin') else files).get(target)!=digest:
            raise ValueError('build artifact hash mismatch: '+name)
    result=verify(out)
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    index=dict(archive_sha256=a.sha256,artifacts=files,binary_files_not_committed=binaries,
        artifact_count=len(files),analysis_sha256=sha(out/'analysis.json'),
        compile_source_commit=result['compile_source_commit'],runtime_source_commit=result['runtime_source_commit'],
        formal_extension_sha256_after=EXTENSION,candidate_adopted=False,
        production_default_changed=False,candidate_GPU_launched=True,new_performance_result=True)
    (out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps({name:r['table'] for name,r in result['runs'].items()},indent=2))


if __name__=='__main__':main()
