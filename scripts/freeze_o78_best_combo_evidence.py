#!/usr/bin/env python3
"""Freeze downloaded v142 evidence and replay all Events without a GPU."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

from analyze_o78_best_combo import analyze

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(directory):
    before = (directory/'reports/o378_v142_extension_before.txt').read_text()
    after = (directory/'reports/o378_v142_extension_after.txt').read_text()
    if before != after or before.split()[0] != EXTENSION:
        raise ValueError('formal extension changed')
    log = (directory/'reports/o378_v142_driver.log').read_text()
    if 'BEST COMBINATION FULL24 FINISHED' not in log or log.count('BEST COMBINATION PAIRED TEST PASSED') != 2:
        raise ValueError('both complete full24 runs required')
    results, dependencies = {}, {}
    gemm_path = ROOT/'docs/evidence/a100_o378_roof_v99/reports/o378_roof_v99_o78_codegen/codegen.json'
    gemm = read(gemm_path)
    dependencies[str(gemm_path.relative_to(ROOT))] = sha(gemm_path)
    original_source = ROOT/'docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78/source_provenance.jsonl'
    dependencies[str(original_source.relative_to(ROOT))] = sha(original_source)
    for variant, version, build_name, decoder_count in (
        ('o7',126,'o378_roof_v126_codegen_r2',131072),
        ('o8',138,'o378_roof_v138_codegen_r2',1048576),
    ):
        run = directory/f'runs/o378_v142_{variant}_full24'
        result = analyze(run)
        if read(directory/f'reports/o378_v142_{variant}_analysis.json') != result:
            raise ValueError('server/local raw Event replay differs')
        if result['formal_extension_sha256'] != EXTENSION:
            raise ValueError('runtime extension changed')
        env = read(run/'environment.json')
        if env['codegen']['output_streaming'] != gemm:
            raise ValueError('v99 frozen audit receipt changed')
        build = ROOT/f'docs/evidence/a100_o378_roof_v{version}/reports/{build_name}/build.json'
        if env['gpu_preparation_build'] != read(build):
            raise ValueError('retained conversion build changed')
        dependencies[str(build.relative_to(ROOT))] = sha(build)
        old = {r['sample_id']:r for r in map(json.loads, original_source.read_text().splitlines())
               if r['variant'] == variant}
        rows = list(map(json.loads, (run/'source_provenance.jsonl').read_text().splitlines()))
        if len(rows) != 24 or {r['sample_id'] for r in rows} != set(old) or any(r != old[r['sample_id']] for r in rows):
            raise ValueError('actual original FP16/source format identity changed')
        validation = read(run/'validation.json')
        if (not validation['passed'] or validation['count'] != 32 or validation['edge_count'] < 1 or
                not validation['packed_word_exhaustive']['passed'] or
                validation['packed_word_exhaustive']['words'] != decoder_count):
            raise ValueError('numerical/decoder validation incomplete')
        for suffix in ('preflight','memcheck'):
            if read(directory/f'runs/o378_v142_{variant}_{suffix}/validation.json') != validation:
                raise ValueError('preflight/safety numerical scope changed')
        safety = (directory/f'reports/o378_v142_{variant}_memcheck.log').read_text()
        if 'ERROR SUMMARY: 0 errors' not in safety or 'BEST COMBINATION VALIDATION PASSED' not in safety:
            raise ValueError('candidate-entry memcheck failed/incomplete')
        result['limited_memcheck'] = dict(errors=0,
            scope='v99_candidate_entry_small_MN_K4096_not_full4096cubed')
        results[variant] = result
    return dict(runs=results, evidence_dependencies=dependencies,
        formal_extension_sha256=EXTENSION, production_default_changed=False,
        new_CUDA_compilation=False, no_filtering=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replay-existing',action='store_true')
    a = p.parse_args(); out = a.output.resolve()
    if (out.exists() and not a.replay_existing) or not out.is_relative_to(ROOT):
        p.error('fresh project output or explicit byte-checked replay required')
    if sha(a.archive) != a.sha256:
        raise ValueError('downloaded archive SHA differs')
    files = {}
    with tarfile.open(a.archive,'r:gz') as archive:
        members = archive.getmembers(); names = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or member.name in names or
                    not (member.isfile() or member.isdir()) or
                    not member.name.startswith(('reports/o378_v142_', 'runs/o378_v142_'))):
                raise ValueError('unsafe/out-of-scope archive member: '+member.name)
            names.add(member.name)
        out.mkdir(parents=True,exist_ok=a.replay_existing)
        for member in members:
            if not member.isfile():
                continue
            if PurePosixPath(member.name).suffix not in ('.json','.jsonl','.txt','.log'):
                raise ValueError('unexpected evidence type: '+member.name)
            data = archive.extractfile(member).read()
            target = out/member.name; target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError('existing evidence differs: '+member.name)
            else:
                target.write_bytes(data)
            files[member.name] = hashlib.sha256(data).hexdigest()
    result = verify(out)
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (out/'index.json').write_text(json.dumps(dict(archive_sha256=a.sha256,artifacts=files,
        artifact_count=len(files),analysis_sha256=sha(out/'analysis.json'),
        evidence_dependencies=result['evidence_dependencies'],
        production_default_changed=False,new_CUDA_compilation=False),indent=2)+'\n')
    print(json.dumps({v:r['stages'] for v,r in result['runs'].items()},indent=2))


if __name__ == '__main__':
    main()
