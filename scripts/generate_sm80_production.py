#!/usr/bin/env python3
"""Freeze the accepted v89/v99/v139/v138 bodies; no report/cubin dependency.

Generated headers are committed. This command is a mechanical regeneration,
not an autotuner; --check verifies their exact provenance before release.
"""
import argparse
from pathlib import Path

from probe_grouped_cta_codegen import generated_headers
from probe_output_streaming_codegen import generated_header as streaming
from probe_nv4_swar_codegen import generated_header as nv4
from probe_nv6_swar_codegen import generated_header as nv6
from probe_hif4_swar_codegen import generated_header as hif4
from probe_mx8_warp_lut_codegen import generated_header as mx8

ROOT = Path(__file__).resolve().parents[1]


def contents():
    body, fallback = generated_headers('o3')
    prep = (ROOT/'csrc/sm80/roof_o78_gpu_prepare.cu').read_text()
    # Device-only guard code; deliberately exclude the experimental host driver.
    prep = prep[prep.index('namespace o78_prepare {'):prep.index('\nnamespace {\nusing Kind=')]
    return {'o3.cuh': body, 'o3_fallback.cuh': fallback,
            'o78.cuh': streaming('o78'), 'nv4.cuh': nv4(),
            'nv6.cuh': nv6(), 'hif4.cuh': hif4(), 'mx8.cuh': mx8(),
            'guard.cuh': '#include <climits>\n'+prep}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    directory = ROOT/'csrc/sm80/production_generated'
    if not args.check:
        directory.mkdir(exist_ok=True)
    for name, body in contents().items():
        path = directory/name
        if args.check:
            if not path.exists() or path.read_text() != body:
                raise SystemExit('generated production source drift: '+name)
        else:
            path.write_text(body)
    print('SM80 accepted production bodies verified' if args.check else 'SM80 production bodies generated')


if __name__ == '__main__':
    main()
