#!/usr/bin/env python3
"""Compare exact cuobjdump SASS instruction words for preserved SM80 entries.

This is a binary regression check, not a correctness or performance test.
No register/resource conclusion is inferred from an equality result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

DEFAULT = r'adangel_sm80_(?:o3_swizzled_bound2|split_grouped_major)'


def instructions(payload, pattern=DEFAULT):
    result = {}
    for block in re.split(r'(?=Function\s*:\s*)', payload):
        if not block.startswith('Function'):
            continue
        symbol = block.splitlines()[0].split(':', 1)[1].strip()
        if not re.search(pattern, symbol):
            continue
        if symbol in result:
            raise ValueError('duplicate matched symbol; compare one architecture at a time')
        words = re.findall(r'/\*\s*(0x[0-9a-fA-F]{16})\s*\*/', block)
        if not words or len(words) % 2:
            raise ValueError('expected two 64-bit encoded words per SM80 instruction')
        result[symbol] = tuple(word.lower() for word in words)
    if not result:
        raise ValueError('no matched kernel')
    return result


def compare(before, after, pattern=DEFAULT):
    left, right = instructions(before, pattern), instructions(after, pattern)
    missing = sorted(set(left) - set(right))
    changed = [key for key in left if key in right and left[key] != right[key]]
    return dict(passed=not missing and not changed, missing=missing, changed=changed,
                old_symbols=len(left), new_symbols=len(right),
                unchanged=[key for key in left if key in right and left[key] == right[key]],
                scope='exact_SASS_words_of_matched_old_entries_not_full_binary_or_performance')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--pattern', default=DEFAULT)
    args = p.parse_args()
    a, b = args.before.read_bytes(), args.after.read_bytes()
    result = compare(a.decode(), b.decode(), args.pattern)
    result['sources'] = [dict(file=str(path), sha256=hashlib.sha256(data).hexdigest())
                         for path, data in ((args.before, a), (args.after, b))]
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
