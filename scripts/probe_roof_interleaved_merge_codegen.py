#!/usr/bin/env python3
"""v63: four independent N atoms, each using one high-then-low INT32 fragment.

Do not change the production headers or dispatch. The old per-atom Merge path
serializes all four IMMA operations; this probe interleaves four N atoms while
halving their live partial storage. A/B reuse and scale/FP32 order are unchanged.
Compilation/ISA evidence is not numerical or performance acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from probe_roof_fullk_integer_codegen import static_entries

ROOT = Path(__file__).resolve().parents[1]
HEADERS = {"o3": "o3_row_scale_epilogue_candidate.cuh",
           "o78": "o78_unsigned_payload_candidate.cuh"}
START = "              auto pls=cute::make_tensor<int>(cute::make_shape(cute::_4{},cute::Int<NAtoms>{}));"
END = "            } else {\n              o1_static_for<0,NAtoms>([&](auto ni) {"
MERGED = '''              // Four independent N atoms preserve inter-atom ILP. Each
              // fragment becomes 16*high first, then receives both low MMAs.
              // All integer intermediates are < 2^18 for a G128 INT8 x INT4
              // dot; this is exact, including negative high and -128 inputs.
              auto pls=cute::make_tensor<int>(cute::make_shape(cute::_4{},cute::Int<NAtoms>{}));
              cute::clear(pls);
              o1_static_for<0,NAtoms>([&](auto ni) {
                auto pl=pls(cute::_,ni);
                cute::gemm(HA{},pl,rh(cute::_,mi,cute::_0{}),br0(cute::_,ni,cute::_0{}),pl);
              });
              o1_static_for<0,NAtoms>([&](auto ni) {
                auto pl=pls(cute::_,ni);
                cute::gemm(HA{},pl,rh1(cute::_,mi,cute::_0{}),br1(cute::_,ni,cute::_0{}),pl);
              });
              o1_static_for<0,NAtoms>([&](auto ni) {
                o1_static_for<0,4>([&](auto vi) { pls(vi,ni)*=16; });
              });
              o1_static_for<0,NAtoms>([&](auto ni) {
                auto pl=pls(cute::_,ni);
                cute::gemm(LA{},pl,ra(cute::_,mi,cute::_0{}),br0(cute::_,ni,cute::_0{}),pl);
              });
              o1_static_for<0,NAtoms>([&](auto ni) {
                auto pl=pls(cute::_,ni);
                cute::gemm(LA{},pl,ra1(cute::_,mi,cute::_0{}),br1(cute::_,ni,cute::_0{}),pl);
              });
              o1_static_for<0,NAtoms>([&](auto ni) {
                auto pl=pls(cute::_,ni);finish(ni,pl,pl);
              });
'''


def generated_header(source, policy):
    if policy not in (0, 1):
        raise ValueError("only control and one merged candidate")
    if policy == 0:
        return source
    if source.count(START) != 1 or source.count(END) != 1:
        raise ValueError("source branch drift; inspect before substitution")
    first = source.index(START)
    last = source.index(END, first)
    source = source[:first] + MERGED + source[last:]
    old = "                const int partial=pl(vi)+16*ph(vi);"
    if source.count(old) != 1:
        raise ValueError("finish source drift")
    return source.replace(old, "                const int partial=pl(vi);")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        p.error("fresh repository output required")
    cuda = Path("/usr/local/cuda-12.8/bin")
    cutlass = ROOT / "third_party/cutlass-src"
    version = subprocess.check_output([str(cuda / "nvcc"), "--version"], text=True)
    commit = subprocess.check_output(["git", "-C", str(cutlass), "rev-parse", "HEAD"], text=True).strip()
    if "release 12.8" not in version or commit != "db1c288993354c88e551c40c19a8fb93a774a241":
        p.error("requires CUDA 12.8 and pinned CUTLASS")
    out.mkdir(parents=True)
    wrapper = ROOT / "csrc/sm80/roof_m128_probe.cu"
    result = dict(scope="compile_only_not_runtime_or_MSE", variants={}, generated_headers={},
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),
                  nvcc=version, cutlass_commit=commit, commands=[])
    def run(cmd, name):
        result["commands"].append(cmd)
        with (out/name).open("w") as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    symbols = {"adangel_roof_m128_o3", "adangel_roof_m128_o78"}
    for policy in (0, 1):
        generated = out / f"policy_{policy}"
        generated.mkdir()
        for family, name in HEADERS.items():
            src = ROOT / "csrc/sm80" / name
            target = generated / f"{family}_m128_generated.cuh"
            target.write_text(generated_header(src.read_text(), policy))
            result["generated_headers"][f"{family}_{policy}"] = dict(
                source=str(src.relative_to(ROOT)), source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                generated=str(target.relative_to(out)), generated_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        stem = f"interleaved_merge_{policy}"
        cmd = [str(cuda/"nvcc"), "-O3", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo", "-arch=sm_80",
               "-DADANGEL_PROBE_M=64", "-I"+str(cutlass/"include"), "-I"+str(generated), str(wrapper)]
        cubin = out / f"{stem}.cubin"
        run(cmd+["-cubin", "-o", str(cubin), "-Xptxas=-v"], f"{stem}_build.log")
        run(cmd+["-ptx", "-o", str(out/f"{stem}.ptx")], f"{stem}_ptx_build.log")
        run([str(cuda/"cuobjdump"), "--dump-sass", str(cubin)], f"{stem}.sass")
        run([str(cuda/"cuobjdump"), "--dump-resource-usage", str(cubin)], f"{stem}_resources.txt")
        entries = static_entries((out/f"{stem}.sass").read_text(), r"^adangel_roof_m128_(?:o3|o78)$", symbols)
        for entry in entries.values():
            if not (entry["native_u4_s4"] and entry["native_s4_s4"] and entry["all_copies_bypass_l1"] and not entry["int8_mma"]):
                raise ValueError("INT4/async ISA audit failed")
        ptx = (out/f"{stem}.ptx").read_text()
        for symbol in symbols:
            block = next(b for b in re.split(r"(?=\.visible \.entry )", ptx)
                         if b.startswith(f".visible .entry {symbol}("))
            if not all(marker in block for marker in ("cp.async.cg.shared.global", ".s32.u4.s4.s32", ".s32.s4.s4.s32")):
                raise ValueError("same-entry PTX audit failed")
        result["variants"][str(policy)] = dict(cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(), entries=entries,
            cta_tile=[64,128,128], threads=128, shared_memory={"o3":50688,"o78":34304})
    previous = ROOT/"docs/evidence/a100_o378_roof_v57/reports/o378_roof_v57/m128_64.sass"
    result["control_comparison"] = compare(previous.read_text(), (out/"interleaved_merge_0.sass").read_text(), r"^adangel_roof_m128_(?:o3|o78)$")
    (out/"codegen.json").write_text(json.dumps(result, indent=2)+"\n")
    if not result["control_comparison"]["passed"]:
        raise SystemExit("control differs from current best SASS")
    print("compile/same-entry audit passed; assess resources before runtime testing")


if __name__ == "__main__":
    main()
