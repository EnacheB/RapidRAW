#!/usr/bin/env bash
# Fork tooling (never upstream). Build this checkout's test build, runnable in
# place: src-tauri/target/fast/RapidRAW. Profile `fast` is in .cargo/config.toml.
set -euo pipefail
cd "$(dirname "$0")"

npx tauri build --no-bundle -- --profile fast

# Tauri copies bundled resources into the shared build cache instead of next to
# this checkout's binary, where the app looks for them; link this checkout's own.
ln -sfn ../../resources ../../lensfun_db src-tauri/target/fast/
echo "Test build: $PWD/src-tauri/target/fast/RapidRAW"
