#!/usr/bin/env python3
"""v80 diagnostic: three existing best kernels; no kernel/format/default edits."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys

from analyze_roof_scale_ncu import analyze
from profile_eight_chain_kernel import ROOT, SYMBOLS


def normalized_counts(counts):
    # cuobjdump sometimes spells operand-less NOP as both NOP and NOP;.
    # Sum aliases, never overwrite one count with a dict comprehension.
    result = Counter()
    for opcode, count in counts.items():
        result[opcode.rstrip(';')] += count
    return result


def analyze_capture(raw, source, receipt):
    variant = receipt['variant']; symbol = SYMBOLS[variant]
    if receipt['expected_kernel'] != symbol or not receipt['numerical_checks_passed'] or not receipt['bitwise_previous_fullk']:
        raise ValueError('identity/correctness receipt mismatch')
    guard = receipt['guard']
    if guard['integer_ctas'] != 2048 or guard['fallback_ctas'] or guard['invalid_ctas']:
        raise ValueError('full integer work required')
    result = analyze(raw,source,54 if variant=='o3' else 59,variant,True,True,
                     expected_symbol=symbol,fullk_integer=True)
    metrics = list(csv.DictReader(io.StringIO(raw)))[1]
    if int(metrics['launch__block_size'].replace(',','')) != 128 or int(metrics['launch__grid_size'].replace(',','')) != 2048:
        raise ValueError('wrong launch shape')
    if result['registers_per_thread'] != receipt['resources']['registers_per_thread']:
        raise ValueError('register identity mismatch')
    entries = receipt['gemm_codegen']['entries']
    actual = Counter()
    for row in csv.DictReader(io.StringIO(source.split('\n',1)[1])):
        match = re.match(r'\s*(?:@!?U?P(?:T|\d+)\s+)?([A-Z][A-Z0-9_]*)',row['Source'])
        if not match: raise ValueError('unknown source opcode')
        actual[match[1]] += 1
    expected = entries[symbol]
    if actual != normalized_counts(expected['opcode_counts']) or sum(actual.values()) != expected['instructions']:
        raise ValueError('profile source does not match audited cubin')
    result.update(static_fingerprint_verified=True,mse_vs_paired_fp16=receipt['mse_vs_paired_fp16'],
                  mse_vs_previous_fullk=receipt['mse_vs_previous_fullk'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--analyze-only',action='store_true')
    a = p.parse_args(); out = a.output.resolve()
    if not out.is_relative_to(ROOT) or (out.exists() and not a.analyze_only):
        p.error('fresh repository output required for collection')
    if not a.analyze_only:
        out.mkdir(parents=True)
        ncu = '/usr/local/cuda-12.8/bin/ncu'; commands=[]
        def run(cmd,name):
            commands.append(dict(command=cmd,output=name))
            (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
            with (out/name).open('w') as f:
                subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        run([ncu,'--version'],'ncu_version.txt')
        for variant,symbol in SYMBOLS.items():
            prefix = out/variant
            run([ncu,'--set','full','--cache-control','all','--clock-control','none',
                 '--kernel-name-base','function','--kernel-name',symbol,'--launch-skip','50','--launch-count','1',
                 '-o',str(prefix),sys.executable,'scripts/profile_eight_chain_kernel.py',
                 '--variant',variant,'--output',str(prefix)],variant+'.log')
            for suffix,page in (('raw','raw'),('source_sass','source')):
                cmd=[ncu,'--import',str(prefix)+'.ncu-rep','--csv','--page',page]
                if page=='source': cmd+=['--print-source','sass']
                run(cmd,f'{variant}_{suffix}.csv')
            print(variant,'captured',flush=True)
    rows=[]; hashes={}
    for variant in SYMBOLS:
        paths=[out/f'{variant}_{suffix}.csv' for suffix in ('raw','source_sass')]
        receipt=out/variant/'receipt.json'
        rows.append(analyze_capture(*(f.read_text(encoding='utf-8-sig') for f in paths),json.loads(receipt.read_text())))
        hashes.update({str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [*paths,receipt]})
    (out/'analysis.json').write_text(json.dumps(dict(scope='existing_best_NCU_diagnostic_not_new_speedup',
        rows=rows,input_sha256=hashes,new_performance_result=False,production_default_changed=False),indent=2,allow_nan=False)+'\n')
    print('THREE BEST-KERNEL NCU CAPTURES AND IDENTITY/WORK CHECKS PASSED',flush=True)


if __name__=='__main__': main()
