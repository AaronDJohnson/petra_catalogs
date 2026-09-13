"""Resolve imports from ``petra`` in every retained tracked notebook."""

import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]

listed = subprocess.run(
    ["git", "ls-files", "example_notebooks/*.ipynb"],
    cwd=REPOSITORY,
    capture_output=True,
    check=True,
    text=True,
).stdout.splitlines()
notebooks = [name for name in listed if (REPOSITORY / name).is_file()]

if not notebooks:
    sys.exit("no retained tracked notebooks found")

failures: list[str] = []
checked = 0

for name in notebooks:
    notebook = json.loads((REPOSITORY / name).read_text())
    for cell_number, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "\n".join(
            "" if line.lstrip().startswith(("%", "!", "?")) else line
            for line in "".join(cell["source"]).splitlines()
        )
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            failures.append(f"{name} cell {cell_number}: {error}")
            continue

        for node in ast.walk(tree):
            targets: list[tuple[str, str | None]] = []
            if isinstance(node, ast.Import):
                targets = [(alias.name, None) for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                targets = [(node.module, alias.name) for alias in node.names]

            for module, attribute in targets:
                if module != "petra" and not module.startswith("petra."):
                    continue
                checked += 1
                try:
                    imported = importlib.import_module(module)
                    if attribute and attribute != "*" and not hasattr(imported, attribute):
                        importlib.import_module(f"{module}.{attribute}")
                except Exception as error:
                    failures.append(
                        f"{name} cell {cell_number}: cannot import "
                        f"{module}.{attribute or ''}: {error}"
                    )

if checked == 0:
    failures.append("no petra imports found")

for failure in failures:
    print(failure, file=sys.stderr)
print(
    f"resolved {checked} petra import(s) across {len(notebooks)} notebook(s)"
)
sys.exit(bool(failures))
