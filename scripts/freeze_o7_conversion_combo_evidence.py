#!/usr/bin/env python3
"""Freeze v139 raw Events and receipts; reuse SHA-identified audited binaries."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

from analyze_o7_conversion_combo import analyze
from analyze_o78_row_fused_codegen import entries
from compare_a100_codegen import compare

ROOT = Path(__file__).resolve().parents[1]
RUN = 'runs/o378_v139_full24'
EXTENSION = '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
PREFIXES = ('reports/o378_v139_', 'runs/o378_v139_')
ALLOWED = ('.json', '.jsonl', '.log', '.txt')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_frozen(directory):
    result = analyze(directory/RUN)
    env = read(directory/RUN/'environment.json')
    after_extension = (directory/'reports/o378_v139_extension_after.txt').read_text().split()
    if len(after_extension) != 2 or after_extension[0] != EXTENSION:
        raise ValueError('post-run formal extension hash changed')
    receipt = env['codegen']['conversion_combo']
    build_dir = ROOT/'docs/evidence/a100_o378_roof_v126/reports/o378_roof_v126_codegen_r2'
    build = read(build_dir/'build.json')
    if (receipt['frozen_build'] != build or env['extension_sha256'] != EXTENSION or
            receipt['new_CUDA_compilation'] or receipt['frozen_v126_candidate_gate_passed'] or
            receipt['control_route'] != ['roof_o78_nv4_swar_benchmark', 1] or
            receipt['combined_route'] != ['roof_o78_mx8_swar_benchmark', 0]):
        raise ValueError('frozen binary/route/default identity failed')
    dependencies = {str((build_dir/'build.json').relative_to(ROOT)): digest(build_dir/'build.json')}
    for name, expected in build['artifact_sha256'].items():
        if name.endswith('.so'):
            continue  # Runtime verified this exact binary hash; do not commit binaries.
        path = build_dir/name
        if digest(path) != expected:
            raise ValueError('retained v126 artifact changed: '+name)
        dependencies[str(path.relative_to(ROOT))] = expected
    after = (build_dir/'prepare.sass').read_text()
    for label, version, name, suffix in (
        ('packed_weight', 118, 'adangel_sm80_row_swar_metadata', 'GroupedSourceKindE0E'),
        ('lookup_activation', 106, 'adangel_sm80_row_warp_lut_metadata', 'GroupedSourceKindE1E'),
        ('scalar_activation', 73, 'adangel_sm80_row_conversion_metadata', 'GroupedSourceKindE1E'),
        ('guard', 73, 'adangel_o78_prepare_cta_guard', ''),
    ):
        path = ROOT/f'docs/evidence/a100_o378_roof_v{version}/reports/o378_roof_v{version}_codegen/prepare.sass'
        before = path.read_text()
        symbols = [s for s in entries(before) if name in s and suffix in s]
        if len(symbols) != 1:
            raise ValueError('one exact retained entry required: '+label)
        actual = compare(before, after, '^'+re.escape(symbols[0])+'$')
        if not actual['passed'] or actual != receipt['retained_entry_comparisons'][label]:
            raise ValueError('retained encoded SASS differs: '+label)
        dependencies[str(path.relative_to(ROOT))] = digest(path)
    gemm_path = ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/codegen.json'
    gemm = read(gemm_path)
    actual_gemm = env['codegen']['best_GEMM']['eight_chain']['build']
    if (actual_gemm['cubin_sha256'] != gemm['cubin_sha256'] or
            not all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma']
                    for e in actual_gemm['entries'].values())):
        raise ValueError('original two-route native INT4 kernel required')
    dependencies[str(gemm_path.relative_to(ROOT))] = digest(gemm_path)
    for policy in ('0', '1'):
        r = env['resources'][policy]
        if (r['kernel_symbol'] != 'adangel_roof_o78_eight_chain_candidate' or
                r['registers_per_thread'] != 168 or r['threads'] != 128 or
                r['cta_tile'] != [64, 128, 128] or r['pipeline_stages'] != 2 or
                r['active_blocks_per_sm'] != 3 or r['GEMM_modified']):
            raise ValueError('identical v78 runtime resources required')
    validation = read(directory/RUN/'validation.json')
    if (not validation['passed'] or validation['count'] != 32 or validation['edge_count'] != 9 or
            validation['packed_word_exhaustive']['words'] != 131072 or
            not validation['packed_word_exhaustive']['passed']):
        raise ValueError('complete retained decoder/guard validation required')
    if read(directory/'runs/o378_v139_preflight/validation.json') != validation:
        raise ValueError('preflight/full24 numerical proof differs')
    safety = {}
    for tool, log_name in (('memcheck', 'memcheck_r2'), ('synccheck', 'synccheck')):
        log = (directory/f'reports/o378_v139_{log_name}.log').read_text()
        if (read(directory/f'runs/o378_v139_{tool}/validation.json') != validation or
                'ERROR SUMMARY: 0 errors' not in log or
                'O7 CONVERSION COMBINATION VALIDATION PASSED' not in log):
            raise ValueError('filtered conversion sanitizer evidence missing')
        safety[tool] = dict(passed=True, scope='row_conversion_filter_small_MN_K4096_not_full_GEMM')
    failed = (directory/'reports/o378_v139_memcheck.log').read_text()
    if 'Filter ill-formed: invalid filter key' not in failed:
        raise ValueError('initial sanitizer CLI error must be retained')
    old_path = ROOT/'docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78/source_provenance.jsonl'
    old = {r['sample_id']: r for r in map(json.loads, old_path.read_text().splitlines()) if r['variant'] == 'o7'}
    rows = list(map(json.loads, (directory/RUN/'source_provenance.jsonl').read_text().splitlines()))
    if len(rows) != 24 or set(old) != {r['sample_id'] for r in rows} or any(r != old[r['sample_id']] for r in rows):
        raise ValueError('full24 original FP16 source/quantization identity mismatch')
    dependencies[str(old_path.relative_to(ROOT))] = digest(old_path)
    result.update(formal_extension_sha256=EXTENSION, runtime_source_commit=env['git_commit'],
        retained_conversion_compile_commit=build['source_commit'], validation_cases=32,
        guard_edges=9, decoder_words=131072, limited_sanitizers=safety,
        retained_evidence_dependencies=dependencies,
        conversion_library_sha256=build['driver_sha256'], GEMM_cubin_sha256=gemm['cubin_sha256'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay-existing', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if (output.exists() and not args.replay_existing) or not output.is_relative_to(ROOT):
        parser.error('fresh output or explicit byte-verified replay inside project required')
    if digest(args.archive) != args.sha256:
        raise ValueError('archive SHA mismatch')
    records = []
    with tarfile.open(args.archive, 'r:gz') as archive:
        members = archive.getmembers()
        names = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or member.name in names or
                    not (member.isfile() or member.isdir()) or not member.name.startswith(PREFIXES)):
                raise ValueError('unsafe or out-of-scope archive member: '+member.name)
            names.add(member.name)
        output.mkdir(parents=True, exist_ok=args.replay_existing)
        for member in members:
            if not member.isfile() or PurePosixPath(member.name).suffix not in ALLOWED:
                continue
            data = archive.extractfile(member).read()
            path = output/member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError('existing evidence differs: '+member.name)
            else:
                path.write_bytes(data)
            records.append(dict(path=member.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    result = validate_frozen(output)
    (output/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    index = dict(archive_sha256=args.sha256, files=sorted(records, key=lambda r: r['path']),
        artifact_count=len(records), artifact_bytes=sum(r['bytes'] for r in records),
        analysis_sha256=digest(output/'analysis.json'), runtime_source_commit=result['runtime_source_commit'],
        retained_evidence_dependencies=result['retained_evidence_dependencies'],
        formal_extension_sha256=EXTENSION, production_default_changed=False, GEMM_modified=False)
    (output/'index.json').write_text(json.dumps(index, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('conversion_audit', 'guard_summary')}, indent=2))


if __name__ == '__main__':
    main()
