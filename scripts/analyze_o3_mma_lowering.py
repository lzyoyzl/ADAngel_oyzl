#!/usr/bin/env python3
"""Attribute static PTX/SASS instructions in the O3 MMA microbenchmark."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SASS_INSTRUCTION = re.compile(
    r"^\s*/\*([0-9a-fA-F]+)\*/\s+(?:@\S+\s+)?"
    r"([A-Z][A-Z0-9_.]*)\s*([^;]*);",
    re.MULTILINE,
)
SASS_FUNCTION = re.compile(r"^\s*Function\s+:\s+(\S+)\s*$", re.MULTILINE)
PTX_ENTRY = re.compile(r"^\.visible\s+\.entry\s+(\S+)\(", re.MULTILINE)
PTX_MMA = re.compile(r"^\s*mma\.sync[^;]*;", re.MULTILINE)


def mnemonic_counts(mnemonics: list[str]) -> dict[str, int]:
    return {
        name: mnemonics.count(name)
        for name in ("IMMA", "LOP3", "SHF", "IMAD", "CALL", "BRA")
    }


def outlined_lowering(sass_body: str) -> dict[str, object] | None:
    """Return the ptxas-outlined helper called once per sub-byte PTX MMA.

    On SM120, ptxas outlines the U4/S4 emulation sequence and emits one call
    site for each PTX MMA.  Counting the whole function and dividing by PTX
    statements is wrong because four call sites share one static helper body.
    """

    instructions = [
        {
            "address": int(match.group(1), 16),
            "mnemonic_full": match.group(2),
            "mnemonic": match.group(2).split(".", 1)[0],
            "operands": match.group(3).strip(),
        }
        for match in SASS_INSTRUCTION.finditer(sass_body)
    ]
    calls = [instruction for instruction in instructions if instruction["mnemonic"] == "CALL"]
    if not calls:
        return None
    target_match = re.search(r"0x([0-9a-fA-F]+)", str(calls[0]["operands"]))
    if target_match is None:
        return None
    target = int(target_match.group(1), 16)
    start = next(
        (index for index, instruction in enumerate(instructions) if instruction["address"] == target),
        None,
    )
    if start is None:
        return None
    helper: list[dict[str, object]] = []
    for instruction in instructions[start:]:
        helper.append(instruction)
        if instruction["mnemonic"] == "RET":
            break
    bases = [str(instruction["mnemonic"]) for instruction in helper]
    return {
        "call_sites": len(calls),
        "target_address_hex": f"0x{target:x}",
        "static_total": len(helper),
        "static": mnemonic_counts(bases),
    }


def split_sections(text: str, pattern: re.Pattern[str]) -> dict[str, str]:
    matches = list(pattern.finditer(text))
    return {
        match.group(1): text[match.end() : matches[index + 1].start()]
        if index + 1 < len(matches)
        else text[match.end() :]
        for index, match in enumerate(matches)
    }


def analyze(directory: Path) -> dict[str, object]:
    records = [
        json.loads(line)
        for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ptx_sections = split_sections(
        (directory / "microbench.ptx").read_text(encoding="utf-8", errors="replace"),
        PTX_ENTRY,
    )
    sass_sections = split_sections(
        (directory / "microbench.sass").read_text(encoding="utf-8", errors="replace"),
        SASS_FUNCTION,
    )

    kernels: dict[str, object] = {}
    for record in records:
        symbol = record["kernel_symbol"]
        ptx_body = ptx_sections.get(symbol, "")
        sass_body = sass_sections.get(symbol, "")
        mnemonics = [
            match.group(2).split(".", 1)[0]
            for match in SASS_INSTRUCTION.finditer(sass_body)
        ]
        ptx_mma = len(PTX_MMA.findall(ptx_body))
        counts = mnemonic_counts(mnemonics)
        helper = outlined_lowering(sass_body)
        kernels[symbol] = {
            "shape": record["shape"],
            "kind": record["kind"],
            "chains": record["chains"],
            "ptx_mma_static": ptx_mma,
            "sass_static_total": len(mnemonics),
            "sass_static": counts,
            "outlined_lowering_per_ptx_mma": helper,
            "direct_imma_per_ptx_mma": (
                counts["IMMA"] / ptx_mma if ptx_mma and helper is None else None
            ),
            "median_ms": record["median_ms"],
            "logical_tops": record["logical_tops"],
            "registers_per_thread": record["registers_per_thread"],
            "local_bytes": record["local_bytes"],
            "checksum": record["checksum"],
        }

    result = {
        "directory": str(directory),
        "kernels": kernels,
    }
    (directory / "static_instruction_attribution.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory), indent=2))


if __name__ == "__main__":
    main()
