#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

notebooks=()
while IFS= read -r -d '' notebook; do
    if [[ -f "$notebook" ]]; then
        notebooks+=("$notebook")
    fi
done < <(git ls-files -z 'example_notebooks/*.ipynb')

if [[ ${#notebooks[@]} -eq 0 ]]; then
    echo "no retained tracked notebooks found" >&2
    exit 1
fi

nbqa flake8 "${notebooks[@]}" --select=F,E9
