#!/usr/bin/env python3
"""Small native O9/O10 corpus for memcheck/racecheck, not performance evidence."""
import argparse
import hashlib
import json
from pathlib import Path

from benchmark_a100_mixed import BINARY_TILES, integer_reference, validate_bitplanes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--tiles",nargs="+",choices=BINARY_TILES,default=list(BINARY_TILES))
    args=p.parse_args()
    if args.output.exists():
        p.error("use a fresh output file")
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(9010)
    a=torch.randn(64,512,device="cuda").half()
    w=torch.randn(128,512,device="cuda").half()
    checks=[]
    for v,family in mf.BINARY_VARIANTS.items():
        wf,af=mf.VARIANTS[family]
        wsrc,asrc=mf.quantize_source(w,wf),mf.quantize_source(a,af)
        wq,ws=mf.to_fixed_reference(wsrc);aq,asc=mf.to_fixed_reference(asrc)
        ref=integer_reference(aq,asc,wq,ws)
        for tile in args.tiles:
            for layout in ("row_major","group_major"):
                r=native._benchmark_mixed(v,"compute_only",wsrc,asrc,0,1,2,tile,layout)
                validate_bitplanes(r,wq,ws,aq,asc,8 if v=="o9" else 6)
                assert torch.equal(r["output"].view(torch.int32),ref.view(torch.int32)), (v,tile,layout)
                checks.append({"variant":v,"tile":tile,"layout":layout,"reference_bitwise":True})
    args.output.write_text(json.dumps({"passed":True,"scope":"small_binary_safety_not_performance",
        "binary_sha256":hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),"checks":checks},indent=2)+"\n")
    print(json.dumps({"passed":True,"checks":len(checks)}))


if __name__=="__main__":
    main()
