#!/usr/bin/env python3
"""Freeze v138 raw texts; retain binaries only in the SHA-verified archive."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import tarfile

from analyze_hif4_swar import analyze

ROOT=Path(__file__).resolve().parents[1]
RUN='runs/o378_v138_completed24'
BUILD='reports/o378_roof_v138_codegen_r2'
EXTENSION='94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
PREFIXES=('reports/o378_roof_v138_', 'reports/o378_v138_', 'runs/o378_v138_')
ALLOWED=('.json','.jsonl','.txt','.log','.sass','.ptx','.cu','.cuh','.cpp')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_frozen(directory):
    result=analyze(directory/RUN)
    env=json.loads((directory/RUN/'environment.json').read_text())
    build=json.loads((directory/BUILD/'build.json').read_text())
    initial=json.loads((directory/'reports/o378_roof_v138_codegen/build.json').read_text())
    if env['extension_sha256']!=EXTENSION:raise ValueError('formal extension changed')
    if initial['audit']['worth_runtime_validation']:raise ValueError('initial name-audit failure lost')
    if env['codegen']['conversion_swar']!=build:raise ValueError('runtime/build receipt differs')
    validation=json.loads((directory/RUN/'validation.json').read_text())
    if (not validation['passed'] or validation['count']!=32 or validation['edge_count']!=3 or
        validation['packed_word_exhaustive']['words']!=1<<20 or not validation['packed_word_exhaustive']['passed']):
        raise ValueError('numerical proof incomplete')
    recovery=json.loads((directory/RUN/'recovery.json').read_text())
    if (recovery['initial_run']!='runs/o378_v138_full24' or
        recovery['resumed_run']!='runs/o378_v138_resumed4' or
        recovery['completed_samples']!=20 or
        recovery['accepted_initial_records']!=480 or
        recovery['accepted_resumed_records']!=96 or
        recovery['interrupted_incomplete_records_retained_in_source']!=3 or
        not recovery['not_CV_or_performance_selection'] or
        env['acquisition_recovery']!=recovery):
        raise ValueError('interruption provenance incomplete')
    old=directory/recovery['initial_run'];new=directory/recovery['resumed_run']
    for key,path in (('initial_results_sha256',old/'results.jsonl'),
                     ('resumed_results_sha256',new/'results.jsonl'),
                     ('initial_environment_sha256',old/'environment.json'),
                     ('resumed_environment_sha256',new/'environment.json')):
        if digest(path)!=recovery[key]:raise ValueError('recovery source changed')
    decode=lambda p:list(map(json.loads,p.read_text().splitlines()))
    from resume_hif4_swar import completed_prefix
    ids=[r['sample_id'] for r in decode(directory/RUN/'source_provenance.jsonl')]
    start,prefix,incomplete=completed_prefix(decode(old/'results.jsonl'),ids)
    if start!=20 or len(incomplete)!=3:raise ValueError('completion selection changed')
    if prefix+decode(new/'results.jsonl')!=decode(directory/RUN/'results.jsonl'):
        raise ValueError('complete rows changed during recovery')
    old_env=json.loads((old/'environment.json').read_text())
    new_env=json.loads((new/'environment.json').read_text())
    if (old_env['git_commit']!=recovery['initial_git_commit'] or
        new_env['git_commit']!=recovery['resumed_git_commit'] or
        env!=dict(old_env,acquisition_recovery=recovery) or
        new_env['sample_start']!=20):raise ValueError('session provenance changed')
    for key in ('extension_sha256','codegen','gpu_preparation_build','resources','variants',
                'torch','cuda','gpu','prepared_manifest_sha256','raw_manifest_sha256',
                'control','candidate','source_quantization','timing_scope','no_filtering'):
        if old_env[key]!=new_env[key]:raise ValueError('recovery changed kernel/protocol')
    safety={}
    for tool in ('memcheck','synccheck'):
        proof=json.loads((directory/f'runs/o378_v138_{tool}/validation.json').read_text())
        log=(directory/f'reports/o378_v138_{tool}.log').read_text()
        if proof!=validation or 'ERROR SUMMARY: 0 errors' not in log or 'HIF4 PACKED SWAR VALIDATION PASSED' not in log:
            raise ValueError('limited conversion safety check incomplete')
        safety[tool]=dict(passed=True,scope='new_HiF4_conversion_and_decoder_only_small_MN_K4096')
    result.update(formal_extension_sha256=EXTENSION,validation_cases=32,guard_edges=3,
        decoder_words=1<<20,limited_sanitizers=safety,
        compile_source_commit=build['source_commit'],runtime_source_commit=env['git_commit'],
        recovery=recovery)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--replay-existing',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if (out.exists() and not a.replay_existing) or not out.is_relative_to(ROOT):
        p.error('fresh repository output or explicit byte-verified replay required')
    if digest(a.archive)!=a.sha256:raise ValueError('archive SHA mismatch')
    records=[]
    with tarfile.open(a.archive,'r:gz') as archive:
        members=archive.getmembers();names=set()
        for member in members:
            path=PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or member.name in names or not (member.isfile() or member.isdir()):
                raise ValueError('unsafe archive member')
            names.add(member.name)
            if not member.name.startswith(PREFIXES):raise ValueError('out of scope member: '+member.name)
        out.mkdir(parents=True,exist_ok=a.replay_existing)
        for member in members:
            if not member.isfile() or PurePosixPath(member.name).suffix not in ALLOWED:continue
            data=archive.extractfile(member).read();target=out/member.name
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if target.read_bytes()!=data:raise ValueError('existing evidence differs: '+member.name)
            else:target.write_bytes(data)
            records.append(dict(path=member.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    result=validate_frozen(out)
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    index=dict(archive_sha256=a.sha256,files=sorted(records,key=lambda r:r['path']),
        analysis_sha256=digest(out/'analysis.json'),artifact_count=len(records),
        artifact_bytes=sum(r['bytes'] for r in records),binary_artifacts_in_archive_not_git=True,
        compile_source_commit=result['compile_source_commit'],runtime_source_commit=result['runtime_source_commit'],
        formal_extension_sha256=EXTENSION,production_default_changed=False,GEMM_modified=False)
    (out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='conversion_compile_audit'},indent=2))


if __name__=='__main__':main()
