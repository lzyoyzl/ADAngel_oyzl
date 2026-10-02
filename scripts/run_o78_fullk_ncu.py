#!/usr/bin/env python3
"""Capture four exact v67/v69 GEMM entries; preserve profiler provenance."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-build', type=Path, required=True)
    p.add_argument('--gemm-cubins', type=Path, required=True)
    args = p.parse_args()
    directory = args.output.resolve()
    if directory.exists() or not directory.is_relative_to(ROOT):
        p.error('fresh repository output required')
    directory.mkdir(parents=True)
    ncu = '/usr/local/cuda-12.8/bin/ncu'
    commands = []
    def run(cmd, output):
        commands.append(dict(command=cmd, output=str(output)))
        (directory/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        with output.open('w') as stream:
            subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    run([ncu, '--version'], directory/'ncu_version.txt')
    for variant in ('o7','o8'):
        for policy in (0,1):
            stem = f'{variant}_p{policy}'
            prefix = directory / stem
            symbol = 'adangel_roof_o78_fullk_' + ('candidate' if policy else 'control')
            cmd = [ncu, '--set', 'full', '--cache-control', 'all', '--clock-control', 'none',
                   '--kernel-name-base', 'function', '--kernel-name', symbol,
                   '--launch-skip', '50', '--launch-count', '1', '-o', str(prefix),
                   sys.executable, 'scripts/profile_o78_fullk_kernel.py', '--variant', variant,
                   '--policy', str(policy), '--gpu-build', str(args.gpu_build),
                   '--gemm-cubins', str(args.gemm_cubins), '--output', str(prefix)]
            run(cmd, directory/f'{stem}.log')
            for suffix, page in (('raw','raw'),('source_sass','source')):
                export = [ncu, '--import', str(prefix)+'.ncu-rep', '--csv', '--page', page]
                if page == 'source':
                    export += ['--print-source', 'sass']
                run(export, directory/f'{stem}_{suffix}.csv')
            print(variant, policy, 'captured', flush=True)
    run([sys.executable, 'scripts/analyze_o78_fullk_ncu.py', '--directory', str(directory)],
        directory/'analysis.json')
    print('FOUR NCU CAPTURES AND IDENTITY/WORK CHECKS PASSED', flush=True)


if __name__ == '__main__':
    main()
