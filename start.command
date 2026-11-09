#!/bin/sh
# double-click (macOS) to start the exo dashboard in the browser; closing this window stops it
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv run --group dev start
