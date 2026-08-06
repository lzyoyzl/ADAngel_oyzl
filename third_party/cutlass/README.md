# CUTLASS pin metadata

The experiment is pinned to NVIDIA CUTLASS `v4.5.2`, commit
`db1c288993354c88e551c40c19a8fb93a774a241`. Run `scripts/fetch_cutlass.sh`; it clones the
source into the ignored sibling directory `third_party/cutlass-src` and verifies the full SHA.

An existing checkout is also allowed through `ADANGEL_CUTLASS_ROOT`, but the recorded commit in
`environment.json` must match `PINNED_REVISION` before a run is accepted as formal.
