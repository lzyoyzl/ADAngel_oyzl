#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
destination="$repo_root/third_party/cutlass-src"
repository=https://github.com/NVIDIA/cutlass.git
tag=v4.5.2
expected=db1c288993354c88e551c40c19a8fb93a774a241

if [[ -e "$destination" ]]; then
  if [[ ! -d "$destination/.git" ]]; then
    echo "$destination exists but is not a git checkout" >&2
    exit 1
  fi
else
  git clone --filter=blob:none --no-checkout "$repository" "$destination"
fi

git -C "$destination" fetch --depth=1 origin "refs/tags/$tag"
git -C "$destination" checkout --detach "$expected"
actual=$(git -C "$destination" rev-parse HEAD)
if [[ "$actual" != "$expected" ]]; then
  echo "CUTLASS revision mismatch: expected $expected, got $actual" >&2
  exit 1
fi

echo "CUTLASS $tag ready at $destination"
echo "commit=$actual"
