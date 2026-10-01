#!/usr/bin/env python3
"""CPU feasibility probe, NOT a CUDA backend or a performance benchmark.

Exhaustively compare integer-only payload decoding/RNE with the independent
floating source-format reference. Effective scale generation is NOT changed.
"""
import json
from pathlib import Path
import runpy


def round_unsigned_pow2(value, shift):
    """Exact RNE(value * 2**shift), for a nonnegative small integer."""
    if value < 0:
        raise ValueError('unsigned significand required')
    if shift >= 0:
        return value << shift
    bits = -shift
    whole, remainder = value >> bits, value & ((1 << bits) - 1)
    half = 1 << (bits - 1)
    return whole + int(remainder > half or (remainder == half and whole & 1))


def integer_fixed(code, kind, micro8=0, micro4=0):
    """Formats and F match the existing O7/O8 conversion contract exactly."""
    sign_bit = {'e2m1': 8, 'e4m3': 128, 'e2m3': 32, 's1p2': 8}[kind]
    if not 0 <= code < sign_bit * 2 or micro8 not in (0, 1) or micro4 not in (0, 1):
        raise ValueError('invalid payload/micro exponent')
    mag = code & (sign_bit - 1)
    if kind == 's1p2':
        # HiF4 internal micro exponents, followed by Q4 F=0.
        q = round_unsigned_pow2(mag, micro8 + micro4 - 2)
    elif kind == 'e4m3':
        if mag == 127:
            raise ValueError('E4M3 NaN code is forbidden')
        e, mant = mag >> 3, mag & 7
        # E4M3 bias7/mantissa3 and F=-2; subnormal exponent is 1-bias.
        q = round_unsigned_pow2(8 + mant if e else mant, e - 12 if e else -11)
    else:
        mbits = 1 if kind == 'e2m1' else 3
        e, mant = mag >> mbits, mag & ((1 << mbits) - 1)
        # E2M1 F=0 and E2M3 F=2 both leave shift e-2.
        q = round_unsigned_pow2((1 << mbits) + mant if e else mant, e - 2 if e else -1)
    return -q if code & sign_bit else q


def probe():
    root = Path(__file__).resolve().parents[1]
    ref = runpy.run_path(str(root / 'python/adangel/quantization/mixed_formats.py'))
    rows = []
    for kind, qbits, fractional in (('e2m1', 4, 0), ('e4m3', 8, -2),
                                     ('s1p2', 4, 0), ('e2m3', 6, 2)):
        checked, mismatches, maximum = 0, [], 0
        micro_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)] if kind == 's1p2' else [(0, 0)]
        for a, b in micro_pairs:
            for code in range(2 * ref['SIGN_BITS'][kind]):
                if kind == 'e4m3' and code & 127 == 127:
                    continue
                value = ref['decode_scalar'](code, kind) * (2 ** (a + b))
                expected = ref['fixed_scalar'](value, qbits, fractional)
                actual = integer_fixed(code, kind, a, b)
                if actual != expected:
                    mismatches.append([code, a, b, actual, expected])
                maximum = max(maximum, abs(actual))
                checked += 1
        rows.append(dict(kind=kind, q_bits=qbits, F=fractional, checked=checked,
                         mismatches=mismatches, max_abs_integer=maximum))
    return dict(passed=not any(r['mismatches'] for r in rows),
                scope='CPU finite payload semantics only; no CUDA execution or timing',
                changes_scale=False, changes_gemm=False, performance_measured=False,
                checks=sum(r['checked'] for r in rows), rows=rows)


if __name__ == '__main__':
    result = probe()
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)
