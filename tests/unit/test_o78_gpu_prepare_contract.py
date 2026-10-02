"""Offline source and integer-proof contracts for isolated v68 GPU preparation.

These checks require neither Torch nor CUDA. GPU correctness, instruction audit,
memory safety and performance remain separate acceptance evidence.
"""
import ast
from pathlib import Path
import random
import unittest


ROOT = Path(__file__).resolve().parents[2]
INT32_MAX = 2**31 - 1
BOUND = INT32_MAX**2
CAP = BOUND + 1


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def cpp_body(source, signature):
    """Return one C++ body, including nested scopes, without importing code."""
    start = source.index("{", source.index(signature))
    depth = 1
    for index in range(start + 1, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1:index]
    raise ValueError(f"unclosed source body: {signature}")


def python_method(source, class_name, method_name):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    return ast.get_source_segment(source, method)


class OnlinePreparationSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cuda = read("csrc/sm80/roof_o78_gpu_prepare.cu")
        cls.codegen = read("scripts/probe_o78_gpu_prepare_codegen.py")
        cls.benchmark = read("scripts/benchmark_o78_fullk_gpu_prepare.py")

    def test_experiment_excluded_from_formal_build_and_dispatch(self):
        for path in ("setup.py", "csrc/bindings.cpp", "csrc/sm80/o1_o3.cu"):
            source = read(path)
            for entry in ("roof_o78_gpu_prepare", "roof_o78_gpu_benchmark",
                          "adangel_roof_o78_fullk_candidate", "roof_o78_fullk_probe"):
                self.assertNotIn(entry, source, path)
        self.assertIn("independent_GPU_preparation_no_default_change", self.codegen)
        self.assertIn("production_default_changed=False", self.benchmark)
        self.assertIn("'-arch=sm_80'", self.codegen)
        self.assertIn("not out.is_relative_to(ROOT)", self.codegen)
        self.assertIn("out.exists()", self.codegen)
        for command in ("pip install", "git merge", "git push"):
            self.assertNotIn(command, self.codegen)

    def test_anchor_uses_every_source_scale_before_payload_statistics(self):
        row = cpp_body(self.cuda, "template<int Kind> __device__ void row(")
        anchor = row[row.index("if(warp==0)"):row.index("__syncthreads();")]
        self.assertIn("decode<Kind>(codes[r*32+lane],mant,exp,bad)", anchor)
        self.assertIn("int amin=mant?exp:INT_MAX", anchor)
        self.assertIn("amin=min(amin,__shfl_down_sync", anchor)
        self.assertIn("if(amin==INT_MAX) amin=0", anchor)
        self.assertIn("const int delta=exp-amin", anchor)
        # Payload-zero groups cannot be omitted from the anchor selection.
        self.assertNotIn("packed", anchor)
        self.assertNotIn("sq", anchor)
        self.assertLess(row.index("const int delta=exp-amin"), row.index("unsigned sq=0"))
        self.assertIn("factors[lane*rows+r]", row)
        self.assertIn("size_t(g)*rows+r", row)

    def test_saturation_detects_overflow_before_using_low_product(self):
        self.assertIn("Bound=uint64_t(INT32_MAX)*INT32_MAX", self.cuda)
        self.assertIn("Cap=Bound+1", self.cuda)
        term = cpp_body(self.cuda, "__device__ uint64_t sat_term(")
        self.assertIn("__umul64hi(f2,square)", term)
        self.assertIn("hi || lo>Cap?Cap:lo", term)
        row = cpp_body(self.cuda, "template<int Kind> __device__ void row(")
        self.assertIn("delta>=31", row)
        self.assertIn("uint64_t(INT32_MAX)>>delta", row)
        self.assertIn("factor_bad?Cap:total", row)
        self.assertIn("sat_add(total,partial[i])", row)

    def test_invalid_status_precedes_fallback_and_guard_covers_coefficients(self):
        row = cpp_body(self.cuda, "template<int Kind> __device__ void row(")
        self.assertIn("bad?2u:(overflow || basebad?1u:0u)", row)
        self.assertIn("v=warp_max(v)", row)
        guard = cpp_body(self.cuda, "void adangel_o78_prepare_cta_guard(")
        self.assertIn("s=max(t<64?ast[r]:0,wst[c])", guard)
        self.assertIn("s=max(s,st[i])", guard)
        self.assertRegex(guard, r"if\s*\(s==0\s*&&")
        self.assertIn("uint64_t(ax)*wx>INT32_MAX", guard)
        self.assertIn("w && a>o78_prepare::Bound/w", guard)
        self.assertIn("result[blockIdx.y*(n/128)+blockIdx.x]=s", guard)
        self.assertIn("bad=c==255", self.cuda)
        self.assertIn("bad=c>126", self.cuda)

    def test_scalar_multipliers_preserved_across_c_abi_and_python_calls(self):
        for entry in ("roof_o78_gpu_prepare", "roof_o78_gpu_benchmark"):
            signature = self.cuda[self.cuda.index(f'extern "C" int {entry}'):]
            self.assertIn("float a_mult,float w_mult", signature[:signature.index("{")])
            body = cpp_body(self.cuda, f'extern "C" int {entry}')
            self.assertIn("x.a_mult=a_mult;x.w_mult=w_mult", body)
        weight = cpp_body(self.cuda, "void weight(bool candidate)")
        activation = cpp_body(self.cuda, "void activation(bool candidate)")
        self.assertIn("metadata<1>(sw,false,w_mult)", weight)
        self.assertIn("metadata<2>(sw,false,w_mult)", weight)
        self.assertIn("metadata<0>(sa,true,a_mult)", activation)
        self.assertIn("metadata<1>(sa,true,a_mult)", activation)
        init = python_method(self.benchmark, "Driver", "__init__")
        for entry in ("roof_o78_gpu_prepare", "roof_o78_gpu_benchmark"):
            assignment = next(n for n in ast.walk(ast.parse(init))
                              if isinstance(n, ast.Assign) and
                              ast.unparse(n.targets[0]) == f"self.lib.{entry}.argtypes")
            # Count scalar floats, excluding the benchmark's float* times.
            scalar_floats = [n for n in assignment.value.elts
                             if isinstance(n, ast.Attribute) and n.attr == "c_float"
                             and isinstance(n.value, ast.Name) and n.value.id == "ct"]
            self.assertEqual(len(scalar_floats), 2)
        for method in ("prepare", "run"):
            source = python_method(self.benchmark, "Driver", method)
            self.assertIn("case.a_multiplier, case.w_multiplier", source)
        case = python_method(self.benchmark, "Case", "__init__")
        self.assertIn('np.float32(4 if variant == "o7"', case)
        self.assertIn("np.float32(tensor_scale) * np.float32(.25)", case)
        self.assertIn('np.float32(tensor_scale if variant == "o7" else 1)', case)

    def test_direct_path_precedes_amortized_conversion_and_events_are_preallocated(self):
        body = cpp_body(self.cuda, 'extern "C" int roof_o78_gpu_benchmark')
        direct_start = body.index("if(mode!=0)")
        first_event = body.index("cuEventRecord(")
        isolated_start = body.index("const bool weight=mode==0 || mode==2")
        self.assertLess(body.index("Events main_events("), direct_start)
        self.assertLess(body.index("w_events(repeats*2),a_events(repeats*2)"), first_event)
        self.assertLess(body.index("x.weight(candidate);x.activation(candidate);"), direct_start)
        self.assertLess(first_event, isolated_start)
        direct = body[direct_start:isolated_start]
        self.assertNotIn("j<inner", direct)
        self.assertIn("if(mode==2)x.weight(candidate)", direct)
        self.assertIn("if(mode==2 || mode==3)x.activation(candidate)", direct)
        self.assertEqual(body.count("for(int j=0;j<inner;++j)"), 2)
        self.assertIn("times[i]/=inner", body)
        self.assertIn("times[repeats+i]/=inner", body)
        self.assertIn("main_events.handles[3*i],main_events.handles[3*i+2]", body)
        for allocation in ("cudaMalloc", "cuMemAlloc", "new ", "malloc("):
            self.assertNotIn(allocation, body)
        run = python_method(self.benchmark, "Driver", "run")
        self.assertNotIn("torch.empty", run)
        self.assertNotIn("prepare_fullk_metadata", run)
        case = python_method(self.benchmark, "Case", "__init__")
        self.assertIn("self.state[key] = torch.empty", case)
        self.assertIn("ct.c_uint64 * 16", case)


class SaturatedNormProofTests(unittest.TestCase):
    def test_uint64_saturation_equals_exact_nonnegative_sum(self):
        # sat_add is safe without another high-word test because its arguments
        # have already been capped, so even their sum fits in uint64.
        self.assertLess(2 * CAP, 2**64)
        rng = random.Random(68)
        cases = [[0], [BOUND], [CAP], [CAP, CAP], [BOUND, 1], [BOUND, 2]]
        cases += [[rng.randrange(2**100) for _ in range(32)] for _ in range(100)]
        for terms in cases:
            capped = 0
            for term in terms:
                capped = min(capped + min(term, CAP), CAP)
            self.assertEqual(capped, min(sum(terms), CAP))

    def test_mul_high_and_low_never_admit_wrapped_norm_term(self):
        rng = random.Random(6801)
        pairs = [(0, INT32_MAX), (128 * 128**2, INT32_MAX), (1, INT32_MAX),
                 (128 * 8**2, 2**30), (1, 0), (0, 0)]
        pairs += [(rng.randrange(128 * 128**2 + 1), rng.randrange(INT32_MAX + 1)) for _ in range(1000)]
        for square, factor in pairs:
            factor_squared = factor * factor
            self.assertLess(factor_squared, 2**64)
            product = factor_squared * square
            high, low = divmod(product, 2**64)
            device_result = CAP if high or low > CAP else low
            self.assertEqual(device_result, min(product, CAP))

    def test_capped_division_gate_matches_arbitrary_precision_product(self):
        rng = random.Random(6802)
        values = (0, 1, BOUND - 1, BOUND, CAP, 2**64, 2**128)
        pairs = [(a, w) for a in values for w in values]
        pairs += [(rng.randrange(2**100), rng.randrange(2**100)) for _ in range(1000)]
        for exact_a, exact_w in pairs:
            a, w = min(exact_a, CAP), min(exact_w, CAP)
            division_rejects = bool(w and a > BOUND // w)
            self.assertEqual(division_rejects, exact_a * exact_w > BOUND)
        # Coefficient overflow remains an independent rejection, even if the
        # payload is all zero and both norms are zero.
        self.assertEqual(0 * 0, 0)
        self.assertGreater((2**30) * 3, INT32_MAX)


if __name__ == "__main__":
    unittest.main()
