#!/usr/bin/env python3
"""Audit native A100 O1/O3 MMA and cp.async in each actual GEMM entry."""
import argparse
import json
import re
import subprocess
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    import torch
    from adangel import _sm80
    args.output.mkdir(parents=True,exist_ok=True)
    binary=_sm80.__file__
    dumps={}
    for kind in ("sass","ptx","resource-usage"):
        dump=subprocess.check_output(["cuobjdump",f"--dump-{kind}",binary],text=True)
        (args.output/f"extension.{kind}").write_text(dump)
        dumps[kind]=dump
    records={}
    for name,flag in (("o1","0"),("o3","1")):
        fragments=[s for s in re.split(r"(?=Function\s*:)" ,dumps["sass"])
            if "adangel_sm80_grouped_gemm" in s.splitlines()[0] and f"ILb{flag}E" in s.splitlines()[0]]
        assert len(fragments)==1,(name,"expected one actual SASS function",len(fragments))
        sass=fragments[0]
        (args.output/f"{name}.sass").write_text(sass)
        instructions=re.findall(r"\bIMMA\.[A-Z0-9.]+",sass)
        async_count=len(re.findall(r"\bLDGSTS\b",sass))
        assert async_count>0,(name,"missing asynchronous global-to-shared copy")
        if name=="o3":
            assert any(".U4.S4" in s for s in instructions),"missing native U4xS4"
            assert any(".S4.S4" in s for s in instructions),"missing native S4xS4"
            assert not any(".S8" in s or ".U8" in s for s in instructions),"O3 lowered to INT8"
        else:
            assert any(".S8.S8" in s for s in instructions),"missing native S8xS8"
        records[name]=dict(function=sass.splitlines()[0],mma_counts={s:instructions.count(s) for s in set(instructions)},
            async_copy_count=async_count,local_load_store_count=len(re.findall(r"\b(?:LDL|STL)\b",sass)))
    result=dict(passed=True,binary=binary,device=torch.cuda.get_device_name(),kernels=records)
    (args.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
