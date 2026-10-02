#!/usr/bin/env python3
"""One isolated larger-M reuse candidate; compile/audit gate before GPU timing."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from probe_roof_stream_width_codegen import static_entries as width_entries

ROOT = Path(__file__).resolve().parents[1]
HEADERS = {"o3": "o3_row_scale_epilogue_candidate.cuh", "o78": "o78_unsigned_payload_candidate.cuh"}
OLD = "static_assert(M==64 && WN==2 && K==128);"
NEW = "static_assert((M==64 || M==128) && WN==2 && K==128);"


def generated_header(text):
    if text.count(OLD) != 1:
        raise ValueError("unexpected source; inspect before applying mechanical config change")
    return text.replace(OLD, NEW)


def counts(sass):
    renamed = sass.replace("adangel_roof_m128_", "adangel_roof_stream_width_")
    return {k.replace("stream_width", "m128"): v for k, v in width_entries(renamed).items()}


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
    manifest = dict(scope="compile_only_not_runtime_or_MSE_evidence", variants={}, generated_headers={},
                    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    nvcc=version, cutlass_commit=commit, commands=[])
    for family, name in HEADERS.items():
        source = ROOT / "csrc/sm80" / name
        target = out / f"{family}_m128_generated.cuh"
        target.write_text(generated_header(source.read_text()))
        manifest["generated_headers"][family] = dict(source=str(source.relative_to(ROOT)),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            generated_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            exact_substitution=[OLD, NEW])

    def run(cmd, name):
        manifest["commands"].append(cmd)
        with (out/name).open("w") as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    for m in (64, 128):
        prefix = f"m128_{m}"
        cmd = [str(cuda/"nvcc"), "-O3", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo", "-arch=sm_80",
               f"-DADANGEL_PROBE_M={m}", "-I"+str(cutlass/"include"), "-I"+str(out),
               "csrc/sm80/roof_m128_probe.cu"]
        cubin = out / f"{prefix}.cubin"
        run(cmd+["-cubin", "-o", str(cubin), "-Xptxas=-v"], f"{prefix}_build.log")
        run(cmd+["-ptx", "-o", str(out/f"{prefix}.ptx")], f"{prefix}_ptx_build.log")
        run([str(cuda/"cuobjdump"), "--dump-sass", str(cubin)], f"{prefix}.sass")
        run([str(cuda/"cuobjdump"), "--dump-resource-usage", str(cubin)], f"{prefix}_resources.txt")
        sass = (out/f"{prefix}.sass").read_text()
        entries = counts(sass)
        for entry in entries.values():
            if not (entry["native_u4_s4"] and entry["native_s4_s4"] and entry["all_copies_bypass_l1"] and not entry["int8_mma"]):
                raise ValueError("INT4/async ISA audit failed")
        ptx = (out/f"{prefix}.ptx").read_text()
        blocks = re.split(r"(?=\.visible \.entry )", ptx)
        for family in ("o3", "o78"):
            block = next(b for b in blocks if b.startswith(f".visible .entry adangel_roof_m128_{family}("))
            for marker in ("cp.async.cg.shared.global", ".s32.u4.s4.s32", ".s32.s4.s4.s32"):
                if marker not in block:
                    raise ValueError("same-entry PTX audit failed")
        manifest["variants"][str(m)] = dict(cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(), entries=entries,
            expected_grid_for_4096=[32, 4096//m], threads=128,
            shared_memory={"o3": 50688 if m==64 else 75264, "o78": 34304 if m==64 else 51200})
    # The control must be identical to previously measured native best controls.
    previous = ROOT/"docs/evidence/a100_o378_roof_v55b/reports/o378_roof_v55b/fullk_integer_0.sass"
    control = (out/"m128_64.sass").read_text().replace("adangel_roof_m128_", "adangel_roof_fullk_integer_")
    result = compare(previous.read_text(), control, r"^adangel_roof_fullk_integer_(?:o3|o78)$")
    manifest["control_comparison"] = result
    (out/"codegen.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))
    if not result["passed"]:
        raise SystemExit("control differs from current best SASS")


if __name__ == "__main__":
    main()
