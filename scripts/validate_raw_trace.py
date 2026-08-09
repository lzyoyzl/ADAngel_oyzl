#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from adangel.trace.raw import RAW_MANIFEST_NAME, validate_raw_trace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a transferred ADAngel raw FP16 trace."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="verify manifest and file hashes without loading every tensor",
    )
    args = parser.parse_args()
    manifest = validate_raw_trace(
        args.input,
        args.config,
        deep=not args.metadata_only,
    )
    print(
        json.dumps(
            {
                "passed": True,
                "manifest": RAW_MANIFEST_NAME,
                "format": manifest["format"],
                "samples": len(manifest["samples"]),
                "input_ids_sha256": manifest["tokenization"]["input_ids_sha256"],
                "deep": not args.metadata_only,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
