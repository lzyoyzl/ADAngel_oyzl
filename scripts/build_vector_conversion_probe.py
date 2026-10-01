#!/usr/bin/env python3
"""Compile isolated vector conversions with unchanged scalar best controls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT):
        p.error('fresh output inside repository required')
    a.output.mkdir(parents=True)
    sources = ['csrc/sm80/roof_vector_conversion_probe.cu', 'csrc/sm80/roof_fused_conversion.cu',
               'csrc/sm80/roof_integer_conversion.cu']
    binary = a.output / 'libvector_conversion_probe.so'
    cmd = ['/usr/local/cuda-12.8/bin/nvcc', '-std=c++17', '-O3', '-lineinfo',
           '-gencode=arch=compute_80,code=sm_80', '-Xcompiler=-fPIC', '-shared', '-Xptxas=-v',
           '-Icsrc/include', '-Icsrc/sm80', *sources, '-o', str(binary)]
    with (a.output / 'build.log').open('w') as out:
        subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, check=True)
    for suffix, mode in (('sass', '--dump-sass'), ('resources.txt', '--dump-resource-usage')):
        with (a.output / f'conversion.{suffix}').open('w') as out:
            subprocess.run(['/usr/local/cuda-12.8/bin/cuobjdump', mode, str(binary)],
                           cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, check=True)
    receipt = dict(command=cmd, library=str(binary), library_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                   sources=[dict(file=s, sha256=hashlib.sha256((ROOT/s).read_bytes()).hexdigest()) for s in sources],
                   git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   production_extension_changed=False)
    (a.output / 'build.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(binary)


if __name__ == '__main__':
    main()
