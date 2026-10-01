"""v36 exact packed operations, traversal and opt-in contracts, CPU-only."""
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]


class O3ConversionTests(unittest.TestCase):
    def test_register_lut_all_codes(self):
        expected=[0,0,1,2,2,3,4,6,0,0,-1,-2,-2,-3,-4,-6]
        for code in range(16):
            mag=(0x64322100>>(4*(code&7)))&15
            self.assertEqual((-mag if code&8 else mag),expected[code])

    def test_four_byte_int8_planes(self):
        def compress(v): return (v&15)|((v>>4)&0xf0)|((v>>8)&0xf00)|((v>>12)&0xf000)
        for a in range(256):
            for b in range(256):
                bs=[a,b,a^0x80,b^0xff]
                v=sum(x<<(8*i) for i,x in enumerate(bs))
                lo,hi=compress(v),compress(v>>4)
                for i,x in enumerate(bs):
                    l=(lo>>(4*i))&15;h=(hi>>(4*i))&15
                    self.assertEqual(l+16*(h-16 if h>=8 else h),x-256 if x>=128 else x)

    def test_exact_destination_coverage(self):
        for lanes in (16,64):
            for rows in (1,3,7,16,64):
                for groups in (1,3,32):
                    src=[];dst=[];sc=[]
                    for g in range(groups):
                        for bx in range((rows+256//lanes-1)//(256//lanes)):
                            for t in range(256):
                                row=bx*(256//lanes)+t//lanes;i=t%lanes
                                if row>=rows: continue
                                src.append((row*groups+g)*lanes+i)
                                dst.append((g*rows+row)*lanes+i)
                                if i==0: sc.append(g*rows+row)
                    self.assertEqual(sorted(src),list(range(rows*groups*lanes)))
                    self.assertEqual(sorted(dst),list(range(rows*groups*lanes)))
                    self.assertEqual(sorted(sc),list(range(rows*groups)))

    def test_opt_in_and_accounted_timing(self):
        cu=(ROOT/'csrc/sm80/o1_o3.cu').read_text()
        self.assertIn('conversion_impl==0 || (split && roof_tune==54',cu)
        self.assertIn('py::arg("conversion_impl")=0',cu)
        self.assertIn('vector conversion requires A8/W4 byte alignment',cu)
        self.assertIn('a.numel()<=std::numeric_limits<int>::max()',cu)
        self.assertIn('meta["gemm_math_changed"]=false',cu)
        self.assertIn('if(roof_fused_payload(roof_tune) || conversion_impl)',cu)
        self.assertLess(cu.index('auto cvw='),cu.index('for(int j=0;j<warmup'))
        self.assertGreater(cu.index('aa=roof_a.permute'),cu.index('auto wt=weight?batch'))
        kernel=(ROOT/'csrc/sm80/roof_o3_conversion.cu').read_text()
        for banned in ('mma.sync','__float2int','atomicAdd','malloc'):
            self.assertNotIn(banned,kernel)
        self.assertIn('grouped_scale[dest_group]=scale[source_group]',kernel)

    def test_audit_rejects_missing_resources_or_local(self):
        analyze=runpy.run_path(str(ROOT/'scripts/audit_o3_conversion_pipeline.py'))['analyze']
        symbols=[f'adangel_sm80_o3_tiled_conversion_test{i}' for i in range(4)]
        sass='\n'.join(f'Function : {s}\n IADD3 R1,R2,R3,R4;' for s in symbols)
        resource='\n'.join(f'Function {s}: REG:20 STACK:0 LOCAL:0' for s in symbols)
        self.assertTrue(analyze(sass,resource)['passed'])
        self.assertFalse(analyze(sass,'')['passed'])
        for op in ('LDL','STL','F2I','MUFU.RCP'):
            self.assertFalse(analyze(sass.replace('IADD3',op,1),resource)['passed'])


if __name__=='__main__': unittest.main()
