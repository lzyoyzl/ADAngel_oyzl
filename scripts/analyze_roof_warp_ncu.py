#!/usr/bin/env python3
"""Recompute v40 work/capacity from one explicitly selected cubin launch."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from analyze_roof_scale_ncu import analyze


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--variant',choices=['o3','o7','o8'],required=True)
    a=p.parse_args()
    rows=[];sources=[]
    for geometry in (0,1):
        texts=[]
        for suffix in ('raw','source_sass'):
            path=a.directory/f'ncu_{a.variant}_g{geometry}_{suffix}.csv'
            payload=path.read_bytes()
            sources.append(dict(file=str(path),sha256=hashlib.sha256(payload).hexdigest()))
            texts.append(payload.decode('utf-8-sig'))
        tune=54 if a.variant=='o3' else 59
        symbol='adangel_roof_warp_o3' if a.variant=='o3' else 'adangel_roof_warp_o78'
        row=analyze(*texts,tune,a.variant,True,True,expected_symbol=symbol)
        raw=list(csv.DictReader(io.StringIO(texts[0])))[1]
        block=int(raw['launch__block_size'].replace(',',''))
        if block!=(128 if geometry==0 else 256):
            raise ValueError('NCU launch-skip selected the wrong geometry')
        row['threads']=block
        row['reference_math_tune']=row.pop('tune')
        row['warp_geometry']=geometry
        row['warp_layout']=[2,2 if geometry==0 else 4]
        rows.append(row)
    print(json.dumps(dict(scope='NCU per-launch diagnostic, not Event timing or production acceptance',
        sources=sources,rows=rows),indent=2,allow_nan=False))


if __name__=='__main__': main()
