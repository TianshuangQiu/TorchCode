#!/usr/bin/env python3
"""Scaffold a new TorchCode problem.

Usage:
    python scripts/new_problem.py <number> <task_id> "<title>" <difficulty>

Example:
    python scripts/new_problem.py 46 my_op "My Custom Op" Medium

Creates:
    torch_judge/tasks/<task_id>.py         — task definition skeleton
    templates/<number>_<task_id>.ipynb     — blank template notebook
    solutions/<number>_<task_id>_solution.ipynb  — solution notebook skeleton

Also prints the README table row to paste in.
"""

import sys
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIFFICULTY_EMOJI = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}
DIFFICULTY_BADGE = {
    "Easy":   "![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square)",
    "Medium": "![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square)",
    "Hard":   "![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square)",
}
COLAB_BASE = "https://colab.research.google.com/github/duoan/TorchCode/blob/master"
GITHUB_BASE = "https://github.com/duoan/TorchCode/blob/master"


def make_task_file(task_id: str, title: str, difficulty: str) -> str:
    return f'''"""{title} task."""

TASK = {{
    "title": "{title}",
    "difficulty": "{difficulty}",
    "function_name": "{task_id}",
    "hint": "TODO: add a hint that names the primitive (e.g. gather, cumsum, scatter, broadcasting).",
    "tests": [
        {{
            "name": "Basic correctness",
            "code": """
import torch
# TODO: write test
result = {{fn}}(torch.tensor([]))
assert result is not None
""",
        }},
        {{
            "name": "Edge case",
            "code": """
import torch
# TODO: write edge case test
result = {{fn}}(torch.tensor([]))
assert result is not None
""",
        }},
        {{
            "name": "Large input — loopy solution would be slow",
            "code": """
import torch, time
# TODO: write large input test (N >= 10k)
t0 = time.time()
# result = {{fn}}(...)
elapsed = time.time() - t0
assert elapsed < 2.0, f\'Too slow: {{elapsed:.2f}}s\'
""",
        }},
    ],
}}
'''


def make_template_notebook(number: int, task_id: str, title: str, difficulty: str) -> dict:
    emoji = DIFFICULTY_EMOJI.get(difficulty, "🟡")
    template_url = f"{COLAB_BASE}/templates/{number:02d}_{task_id}.ipynb"
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "cell-0",
                "metadata": {},
                "source": [
                    f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({template_url})\n",
                    "\n",
                    f"# {emoji} {difficulty}: {title}\n",
                    "\n",
                    "TODO: problem description, signature, rules, example.\n",
                    "\n",
                    "> **Reduction step (say this before coding):** TODO — state output[i] in terms of inputs in one sentence.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell-1",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install torch-judge in Colab (no-op in JupyterLab/Docker)\n",
                    "try:\n",
                    "    import google.colab\n",
                    "    get_ipython().run_line_magic('pip', 'install -q torch-judge')\n",
                    "except ImportError:\n",
                    "    pass",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell-2",
                "metadata": {},
                "outputs": [],
                "source": ["import torch"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell-3",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# ✏️ YOUR IMPLEMENTATION HERE\n",
                    "\n",
                    f"def {task_id}(*args):\n",
                    "    pass  # Replace this",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell-4",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 🧪 Test your implementation\n",
                    "# TODO: add debug prints",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell-5",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# ✅ SUBMIT — Run this cell to check your solution\n",
                    "from torch_judge import check\n",
                    f'check("{task_id}")',
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def make_solution_notebook(number: int, task_id: str, title: str, difficulty: str) -> dict:
    emoji = DIFFICULTY_EMOJI.get(difficulty, "🟡")
    solution_url = f"{COLAB_BASE}/solutions/{number:02d}_{task_id}_solution.ipynb"
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "cell-0",
                "metadata": {},
                "source": [
                    f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({solution_url})\n",
                    "\n",
                    f"# {emoji} Solution: {title}\n",
                    "\n",
                    "**Primitive:** TODO — name the primitive (e.g. `repeat_interleave`, `gather`, `cumsum`)\n",
                    "\n",
                    "**Reduction:** TODO — one sentence stating output[i] in terms of inputs.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell-1",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install torch-judge in Colab (no-op in JupyterLab/Docker)\n",
                    "try:\n",
                    "    import google.colab\n",
                    "    get_ipython().run_line_magic('pip', 'install -q torch-judge')\n",
                    "except ImportError:\n",
                    "    pass",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell-2",
                "metadata": {},
                "outputs": [],
                "source": ["import torch"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell-3",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# ✅ SOLUTION\n",
                    "# primitive: TODO\n",
                    "\n",
                    f"def {task_id}(*args):\n",
                    "    pass  # TODO: implement",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell-4",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Verify\n",
                    "# TODO: add verification prints",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell-5",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run judge\n",
                    "from torch_judge import check\n",
                    f'check("{task_id}")',
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def readme_row(number: int, task_id: str, title: str, difficulty: str) -> str:
    badge = DIFFICULTY_BADGE.get(difficulty, DIFFICULTY_BADGE["Medium"])
    template_gh = f"{GITHUB_BASE}/templates/{number:02d}_{task_id}.ipynb"
    template_colab = f"{COLAB_BASE}/templates/{number:02d}_{task_id}.ipynb"
    colab_img = '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20">'
    return (
        f'| {number} | <a href="{template_gh}" target="_blank">{title}</a> '
        f'<a href="{template_colab}" target="_blank">{colab_img}</a> | '
        f'`{task_id}(...)` | {badge} | ⭐ | TODO: key concepts |'
    )


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)

    number = int(sys.argv[1])
    task_id = sys.argv[2]
    title = sys.argv[3]
    difficulty = sys.argv[4]

    if difficulty not in ("Easy", "Medium", "Hard"):
        print(f"Error: difficulty must be Easy, Medium, or Hard — got '{difficulty}'")
        sys.exit(1)

    # Paths
    task_path = os.path.join(REPO_ROOT, "torch_judge", "tasks", f"{task_id}.py")
    template_path = os.path.join(REPO_ROOT, "templates", f"{number:02d}_{task_id}.ipynb")
    solution_path = os.path.join(REPO_ROOT, "solutions", f"{number:02d}_{task_id}_solution.ipynb")

    # Check nothing already exists
    for path in [task_path, template_path, solution_path]:
        if os.path.exists(path):
            print(f"Error: file already exists: {path}")
            sys.exit(1)

    # Write task file
    with open(task_path, "w") as f:
        f.write(make_task_file(task_id, title, difficulty))
    print(f"Created: {task_path}")

    # Write template notebook
    with open(template_path, "w") as f:
        json.dump(make_template_notebook(number, task_id, title, difficulty), f, indent=1)
    print(f"Created: {template_path}")

    # Write solution notebook
    with open(solution_path, "w") as f:
        json.dump(make_solution_notebook(number, task_id, title, difficulty), f, indent=1)
    print(f"Created: {solution_path}")

    print()
    print("README row to paste into the appropriate section:")
    print()
    print(readme_row(number, task_id, title, difficulty))
    print()
    print("Next steps:")
    print(f"  1. Fill in torch_judge/tasks/{task_id}.py — add hint and tests")
    print(f"  2. Fill in templates/{number:02d}_{task_id}.ipynb — add problem description and skeleton")
    print(f"  3. Fill in solutions/{number:02d}_{task_id}_solution.ipynb — add solution and primitive comment")
    print(f"  4. Paste the README row above into the right section of README.md")
    print(f"  5. Update the badge count in README.md (line: problems-N-orange)")


if __name__ == "__main__":
    main()
