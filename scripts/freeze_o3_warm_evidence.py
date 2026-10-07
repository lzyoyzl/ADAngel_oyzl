#!/usr/bin/env python3
"""Freeze and replay the v136 current-best O3 profiling evidence, not timing."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

from profile_o3_best_warm import ROOT
from run_o3_best_warm_ncu import analyze_capture
from analyze_o3_mma_phase_stalls import CODEGEN, analyze_capture as analyze_phases

CAPTURE = 'reports/o378_roof_v136_o3_warm_ncu'
LOG = 'reports/o378_v136_runtime.log'
TEXT_SUFFIXES = {'.csv', '.json', '.txt', '.log', '.sass', '.ptx', '.cu', '.cuh', '.cpp'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(directory):
    capture = directory/CAPTURE
    receipts = [json.loads(p.read_text()) for p in sorted((capture/'pass_receipts').glob('pass_*/receipt.json'))]
    raw = (capture/'o3_warm_raw.csv').read_text(encoding='utf-8-sig')
    source = (capture/'o3_warm_source_sass.csv').read_text(encoding='utf-8-sig')
    result = analyze_capture(raw, source, receipts)
    saved = json.loads((capture/'analysis.json').read_text())
    if any(saved[k] != v for k, v in result.items()):
        raise ValueError('captured analysis does not replay exactly')
    if len(receipts) != 50 or result['dynamic_instructions'] != 88231936:
        raise ValueError('unexpected replay count or best-O3 work')
    if receipts[0]['extension_sha256_before'] != '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462':
        raise ValueError('formal extension identity changed')
    if 'CURRENT BEST O3 WARM NCU IDENTITY/WORK CHECKS PASSED' not in (directory/LOG).read_text():
        raise ValueError('capture did not finish successfully')
    phase = analyze_phases((CODEGEN/'o3_grouped_cta.sass').read_text(),
        (CODEGEN/'liveness.txt').read_text(), source, result)
    phase['inputs'] = {str(p.relative_to(ROOT)): digest(p) for p in (
        CODEGEN/'o3_grouped_cta.sass', CODEGEN/'liveness.txt', capture/'o3_warm_source_sass.csv', capture/'analysis.json')}
    phase['script_sha256'] = digest(ROOT/'scripts/analyze_o3_mma_phase_stalls.py')
    return result, phase, receipts[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        p.error('fresh repository output required')
    if digest(a.archive) != a.sha256: raise ValueError('archive SHA mismatch')
    files = []
    with tarfile.open(a.archive, 'r:gz') as archive:
        members = archive.getmembers(); seen = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or member.name in seen or
                    not (member.isfile() or member.isdir()) or not (
                    member.name == LOG or member.name == CAPTURE or member.name.startswith(CAPTURE+'/'))):
                raise ValueError('unsafe or out-of-scope archive member: '+member.name)
            seen.add(member.name)
        out.mkdir(parents=True)
        for member in members:
            if not member.isfile() or PurePosixPath(member.name).suffix not in TEXT_SUFFIXES: continue
            content = archive.extractfile(member).read(); destination = out/member.name
            destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(content)
            files.append(dict(path=member.name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
    result, phase, first = replay(out)
    (out/'phase_analysis.json').write_text(json.dumps(phase, indent=2, allow_nan=False)+'\n')
    index = dict(files=sorted(files, key=lambda r: r['path']), artifact_count=len(files),
        artifact_bytes=sum(row['bytes'] for row in files), archive_sha256=a.sha256,
        phase_analysis_sha256=digest(out/'phase_analysis.json'), source_commit=first['git_commit'],
        binary_artifacts_retained_in_archive_not_git=True, new_performance_result=False,
        production_default_changed=False)
    (out/'index.json').write_text(json.dumps(index, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in (
        'memory_metrics_raw', 'raw_selected_metrics', 'pc_sampling', 'native_mma_sass', 'sass_instruction_lines')}, indent=2))


if __name__ == '__main__':
    main()
