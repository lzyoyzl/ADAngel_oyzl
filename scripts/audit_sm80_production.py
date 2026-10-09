#!/usr/bin/env python3
"""Audit actual promoted entries in the rebuilt SM80 extension, not probes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare

ROOT=Path(__file__).resolve().parents[1]
PAIRS={
    'adangel_sm80_o3_fullk_grouped': ('reports/o378_roof_v89_o3_codegen/o3_grouped_cta.sass','adangel_roof_o3_grouped_cta_candidate'),
    'adangel_sm80_o78_fullk_streaming': ('reports/o378_roof_v99_o78_codegen/o78_output_streaming.sass','adangel_roof_o78_output_streaming_candidate'),
}


def audit(ptx,sass):
    result={}
    for symbol in PAIRS:
        p=next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+symbol+'('))
        s=next(b for b in re.split(r'(?=\s*Function\s*:\s*)',sass) if re.match(r'\s*Function\s*:\s*'+symbol+r'\s',b))
        checks=dict(ptx_u4_s4='.s32.u4.s4.s32' in p,ptx_s4_s4='.s32.s4.s4.s32' in p,
            ptx_async='cp.async.cg.shared.global' in p,
            native_u4_s4='IMMA.16864.U4.S4' in s,native_s4_s4='IMMA.16864.S4.S4' in s,
            no_int8_mma=not bool(re.search(r'IMMA\.\S*(?:S8|U8)',s)),sass_async='LDGSTS' in s)
        if not all(checks.values()):raise ValueError((symbol,checks))
        checks['local_loads']=len(re.findall(r'\bLDL\b',s));checks['local_stores']=len(re.findall(r'\bSTL\b',s))
        result[symbol]=checks
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import torch
    from adangel import _sm80 as native
    path=Path(native.__file__);args.output.mkdir(parents=True,exist_ok=False)
    outputs={}
    for flag,name in (('--dump-ptx','extension.ptx'),('--dump-sass','extension.sass'),('--dump-resource-usage','resources.txt')):
        outputs[name]=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump',flag,str(path)],text=True)
        (args.output/name).write_text(outputs[name])
    checks=audit(outputs['extension.ptx'],outputs['extension.sass'])
    for symbol,(file,old) in PAIRS.items():
        if (ROOT/file).exists():
            # Compare encoded instruction words after ONLY renaming the entry.
            checks[symbol]['candidate_sass_comparison']=compare((ROOT/file).read_text(),
                outputs['extension.sass'].replace(symbol,old),'^'+old+'$')
    result=dict(passed=True,extension_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                entries=checks,torch=torch.__version__,cuda=torch.version.cuda)
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
