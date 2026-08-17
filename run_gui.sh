#!/usr/bin/env bash
# Ativa o ambiente virtual .venv e abre a GUI do pipeline.
# Ainda sem chmod: rode com  bash run_gui.sh

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

if [[ ! -f .venv/bin/activate ]]; then
  echo "Ambiente virtual não encontrado: $(pwd)/.venv" >&2
  echo "Crie-o com: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate

exec python3 pipeline_gui.py "$@"
