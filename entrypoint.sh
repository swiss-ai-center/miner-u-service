#!/usr/bin/env bash
set -euo pipefail

# Install Python dependencies at container start
pip install --no-cache-dir --requirement /app/requirements.txt --requirement /app/requirements-all.txt

# Replace shell with the command from CMD
exec "$@"
