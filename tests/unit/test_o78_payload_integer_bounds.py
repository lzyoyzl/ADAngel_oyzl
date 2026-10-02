from pathlib import Path
import hashlib
import json
import random
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from inspect_o78_payload_integer_bounds import weighted_norms,inspect_norms,export_norm_evidence,INT32_MAX
from inspect_o78_integer_alignment_feasibility import dyadic


class PayloadBoundTests(unittest.TestCase):
    def test_all_codes_against_exact_decoded_square(self):
        for kind,count in (('ue8m0',255),('e4m3',127),('e6m2',255)):
            for c in range(count):
                got=weighted_norms([[c,c]],[[3,5]],kind)
                m,e=dyadic(c,kind)
                anchor=got['anchors'][0]
                self.assertEqual(got['norm2'][0],8*m*m if m else 0)
                self.assertEqual(anchor,e if m else 0)

    def test_prefix_cauchy_bound(self):
        rng=random.Random(20261002)
        for _ in range(300):
            groups=5;width=7
            a=[[rng.randrange(-128,128) for _ in range(width)] for _ in range(groups)]
            w=[[rng.randrange(-8,8) for _ in range(width)] for _ in range(groups)]
            ac=[rng.randrange(119,133) for _ in range(groups)]
            wc=[rng.randrange(0,64) for _ in range(groups)]
            av=weighted_norms([ac],[[sum(v*v for v in g) for g in a]],'ue8m0')
            wv=weighted_norms([wc],[[sum(v*v for v in g) for g in w]],'e4m3')
            af=[m << (e-av['anchors'][0]) if m else 0 for m,e in [dyadic(c,'ue8m0') for c in ac]]
            wf=[m << (e-wv['anchors'][0]) if m else 0 for m,e in [dyadic(c,'e4m3') for c in wc]]
            prefix=0;absolute=0;bound=av['norm2'][0]*wv['norm2'][0]
            for g in range(groups):
                term=sum(x*y for x,y in zip(a[g],w[g]))*af[g]*wf[g]
                prefix+=term;absolute+=abs(term)
                self.assertLessEqual(prefix*prefix,bound)
                self.assertLessEqual(absolute*absolute,bound)

    def test_safety_count_matches_exhaustive_and_cta_gate(self):
        rng=random.Random(18)
        for _ in range(30):
            a=dict(norm2=[rng.randrange(0,2**33) for _ in range(7)],factor_max=[1]*7)
            w=dict(norm2=[rng.randrange(0,2**33) for _ in range(9)],factor_max=[1]*9)
            r=inspect_norms(a,w,3,4)
            self.assertEqual(r['int32_norm_safe_outputs'],sum(x*y<=INT32_MAX**2 for x in a['norm2'] for y in w['norm2']))
            safe=0
            for mi in range(0,7,3):
                for ni in range(0,9,4):
                    safe+=all(x*y<=INT32_MAX**2 for x in a['norm2'][mi:mi+3] for y in w['norm2'][ni:ni+4])
            self.assertEqual(r['guard_safe_ctas'],safe)

    def test_exact_threshold_zero_and_huge_factors(self):
        a=dict(norm2=[0,1,INT32_MAX**2],factor_max=[1]*3)
        w=dict(norm2=[1,INT32_MAX**2,INT32_MAX**2+1],factor_max=[1]*3)
        self.assertEqual(inspect_norms(a,w,1,1)['int32_norm_safe_outputs'],6)
        v=weighted_norms([[0,254]],[[1,1]],'ue8m0')
        self.assertEqual(v['norm2'][0],1+2**508)
        self.assertFalse(inspect_norms(v,v,1,1)['all_outputs_guaranteed_safe'])
        zero=weighted_norms([[0,0]],[[7,11]],'e4m3')
        self.assertTrue(inspect_norms(zero,zero)['all_outputs_guaranteed_safe'])

    def test_invalid_input(self):
        for c,s,k in (([],[],'ue8m0'),([[255]],[[1]],'ue8m0'),([[1]],[[1,2]],'e4m3'),([[1]],[[-1]],'e4m3')):
            with self.assertRaises(ValueError): weighted_norms(c,s,k)

    def test_export_preserves_hashes_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temporary:
            base=Path(temporary);source=base/'source';source.mkdir()
            side=dict(norm2=[3],anchors=[0],factor_max=[1],scale_codes=[[127]],
                      group_sum_squares=[[3]])
            case=dict(a=side,w=side)
            content=json.dumps(case).encode();digest=hashlib.sha256(content).hexdigest()
            (source/'case.json').write_bytes(content)
            summary=dict(samples=[dict(file='case.json',sha256=digest,
                                      all_outputs_guaranteed_safe=True)])
            (source/'summary.json').write_text(json.dumps(summary))
            export_norm_evidence(source,base/'export')
            got=json.loads((base/'export/case.json').read_text())
            self.assertEqual(got['a']['norm2'],[3])
            self.assertNotIn('scale_codes',got['a'])
            manifest=json.loads((base/'export/summary.json').read_text())
            self.assertEqual(manifest['samples'][0]['full_source_sha256'],digest)
            self.assertEqual(manifest['samples'][0]['sha256'],
                hashlib.sha256((base/'export/case.json').read_bytes()).hexdigest())
            with self.assertRaises(ValueError): export_norm_evidence(source,base/'export')
            (source/'case.json').write_text('{}')
            with self.assertRaises(ValueError): export_norm_evidence(source,base/'tampered')

    def test_committed_evidence_all_counts_and_exceptional_groups(self):
        evidence=ROOT/'docs/evidence/a100_o378_roof_v65/reports'
        first={}
        for suffix,expected in (('screen',4),('trace24',24)):
            path=evidence/f'o378_roof_v65_{suffix}'
            manifest=json.loads((path/'summary.json').read_text())
            self.assertEqual(len(manifest['samples']),expected*2)
            totals={v:dict(unsafe=0,unsafe_ctas=0,safe_samples=0) for v in ('o7','o8')}
            for entry in manifest['samples']:
                content=(path/entry['file']).read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(),entry['sha256'])
                case=json.loads(content)
                stats=inspect_norms(case['a'],case['w'])
                self.assertEqual(stats,case['stats'])
                for key,value in stats.items(): self.assertEqual(entry[key],value)
                for side in ('a','w'):
                    self.assertEqual(len(case[side]['norm2']),4096)
                    if case['group_statistics_retained']:
                        rebuilt=weighted_norms(case[side]['scale_codes'],
                            case[side]['group_sum_squares'],case[side]['kind'])
                        for key,value in rebuilt.items(): self.assertEqual(case[side][key],value)
                key=(case['sample_id'],case['variant'])
                if suffix=='screen': first[key]=case
                elif key in first: self.assertEqual(case,first[key])
                total=totals[case['variant']]
                total['unsafe']+=stats['outputs']-stats['int32_norm_safe_outputs']
                total['unsafe_ctas']+=stats['ctas']-stats['guard_safe_ctas']
                total['safe_samples']+=stats['all_outputs_guaranteed_safe']
                self.assertTrue(stats['coefficient_int32_safe'])
                if not stats['all_outputs_guaranteed_safe']:
                    self.assertEqual(key,('layer_24_o_proj','o8'))
            self.assertEqual(totals['o7'],dict(unsafe=0,unsafe_ctas=0,safe_samples=expected))
            self.assertEqual(totals['o8'],dict(unsafe=15 if expected==24 else 0,
                unsafe_ctas=12 if expected==24 else 0,safe_samples=23 if expected==24 else 4))
