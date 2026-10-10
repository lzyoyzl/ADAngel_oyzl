#!/usr/bin/env python3
"""Audit size-specialized production entries, with original K4096 regression."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from audit_sm80_production import audit, PAIRS, ROOT
from compare_a100_codegen import compare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    from adangel import _sm80 as native
    args.output.mkdir(parents=True, exist_ok=False)
    path = Path(native.__file__)
    dumps = {}
    for flag, name in [('--dump-ptx', 'extension.ptx'), ('--dump-sass', 'extension.sass'), ('--dump-resource-usage', 'resources.txt')]:
        dumps[name] = subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump', flag, str(path)], text=True)
        (args.output / name).write_text(dumps[name])
    checks = audit(dumps['extension.ptx'], dumps['extension.sass'])
    for symbol, (file, old) in PAIRS.items():
        if (ROOT / file).exists():
            result = compare((ROOT / file).read_text(), dumps['extension.sass'].replace(symbol + '\n', old + '\n'), '^' + old + '$')
            assert result['passed'], result
            checks[symbol]['accepted_4096_sass_unchanged'] = result
    for k in (512, 1024):
        for prefix in ('adangel_sm80_o3_fullk_grouped', 'adangel_sm80_o78_fullk_streaming'):
            symbol = f'{prefix}_k{k}'
            pb = next(b for b in re.split(r'(?=\.visible \.entry )', dumps['extension.ptx']) if b.startswith('.visible .entry ' + symbol + '('))
            sb = next(b for b in re.split(r'(?=\s*Function\s*:\s*)', dumps['extension.sass']) if re.match(r'\s*Function\s*:\s*' + symbol + r'\s', b))
            values = dict(ptx_u4_s4='.s32.u4.s4.s32' in pb, ptx_s4_s4='.s32.s4.s4.s32' in pb,
                ptx_async='cp.async.cg.shared.global' in pb, native_u4_s4='IMMA.16864.U4.S4' in sb,
                native_s4_s4='IMMA.16864.S4.S4' in sb, sass_async='LDGSTS' in sb,
                no_int8_mma=not bool(re.search(r'IMMA\.\S*(?:S8|U8)', sb)))
            assert all(values.values()), (symbol, values)
            values.update(local_loads=len(re.findall(r'\bLDL\b', sb)), local_stores=len(re.findall(r'\bSTL\b', sb)))
            checks[symbol] = values
    result = dict(passed=True, extension_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), entries=checks,
                  torch=torch.__version__, cuda=torch.version.cuda)
    (args.output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
