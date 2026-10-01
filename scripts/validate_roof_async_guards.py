#!/usr/bin/env python3
"""Finite guard checks for isolated55/56; not formal timing acceptance."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tunes', type=int, nargs='+', default=[55,56])
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error('fresh output file with existing parent required')
    if not args.tunes or len(set(args.tunes))!=len(args.tunes) or any(t not in (55,56,57,58,59,60,62) for t in args.tunes):
        parser.error('unique asynchronous scale candidate55-60 required')
    import torch
    from adangel import _sm80 as native
    if torch.cuda.get_device_capability() != (8,0):
        raise RuntimeError('requires SM80')
    torch.set_num_threads(4)
    rows = []
    for tune in args.tunes:
        for variant in ('o7','o8'):
            m,n,k = 64,128,384
            a = torch.zeros((2*m,k//2),device='cuda',dtype=torch.uint8)
            w = torch.zeros((n,k//2),device='cuda',dtype=torch.uint8)

            def scale(size, offset=0):
                buffer = torch.ones(size*(k//128)+offset,device='cuda')*.125
                return buffer.as_strided((size,k//128),(1,size),offset)

            asc,wsc = scale(m),scale(n)
            good = native._benchmark_roof_candidate(variant,tune,a,asc,w,wsc,0,1)
            assert good['output'].dtype==torch.float32 and torch.count_nonzero(good['output'])==0
            assert good['kernel']['scale_copy_async'] and not good['kernel']['fp32_reassociated']
            for operand,size in (('A',m),('W',n)):
                for case in ('misaligned','nan','negative'):
                    bad = scale(size,1 if case=='misaligned' else 0)
                    expected = '16-byte aligned' if case=='misaligned' else f'invalid {operand} scale'
                    if case=='misaligned':
                        assert bad.data_ptr()%16==4 and bad.stride()==(1,size)
                    else:
                        bad[0,0] = float('nan') if case=='nan' else -1
                    try:
                        native._benchmark_roof_candidate(variant,tune,a,
                            bad if operand=='A' else asc,w,bad if operand=='W' else wsc,0,1)
                    except RuntimeError as exc:
                        assert expected in str(exc), str(exc)
                        rows.append(dict(variant=variant,tune=tune,operand=operand,
                                         case=case,rejected=True,expected_message=expected))
                    else:
                        raise AssertionError((variant,tune,operand,case,'unexpected acceptance'))
        try:
            native._benchmark_roof_candidate('o3',tune,a,asc,w,wsc,0,1)
        except RuntimeError as exc:
            assert 'requires O7/O8' in str(exc),str(exc)
            rows.append(dict(variant='o3',tune=tune,case='wrong_format',rejected=True))
        else:
            raise AssertionError('O3 accepted FP32 scale-copy kernel')
    torch.cuda.synchronize()
    result = dict(passed=True,scope='finite_host_guard_tests_not_exhaustive_memory_safety',
                  binary_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),checks=rows)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('async scale guard checks:',len(rows),'passed')


if __name__=='__main__':
    main()
