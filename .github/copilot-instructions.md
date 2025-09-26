This repository contains LaTeX reports/presentations, Jupyter notebooks used for experiments, and a few small Python helpers. The goal of these instructions is to give an AI coding agent the minimal, directly-actionable knowledge needed to be productive in this codebase.

Key things to know (big picture)
- The repo is primarily an authoring / research repo for quantum computing coursework and experiments. The most important artifact types are:
  - LaTeX projects (see `QT207/qt207.tex`, `Report/report.tex`, `old/Presentation/presentation.tex`) — built with `pdflatex`/`latexmk`.
  - Jupyter notebooks under `old/` and `qft_on_qc/` — these contain experiments and narrative code; treat notebooks as primary runnable examples.
  - A couple of small Python scripts (e.g. `QT207/stadium_test_runner.py`, `old/stadium_test_runner.py`) which are either stubs or single-file runners.

Build / run / test workflows (explicit)
- To build papers/presentations: run `pdflatex` or `latexmk` in the directory containing the `.tex` file. Example (from repo root):

  - Build `QT207/qt207.tex`:
    - cd `QT207` && `pdflatex qt207.tex` (repeat or use `latexmk -pdf qt207.tex` for automatic runs)

  - Old presentation build artifacts are in `old/Presentation/` (the repo contains generated `.fdb_latexmk`, `.fls`, and `presentation.pdf` — safe to regenerate with `latexmk`).

- To run notebooks: open the `.ipynb` files (primary ones in `old/` and `qft_on_qc/`) in Jupyter / VS Code. There is no requirements manifest; infer dependencies from notebook imports (common libs: numpy, matplotlib, qiskit in some notebooks). Before executing, create a virtual env and pip-install packages found in cells.

Project-specific patterns and conventions
- Notebooks are the canonical source of runnable examples and reported outputs (figures under `old/` were produced by notebooks). Prefer edits in the original `.ipynb` rather than converting to scripts unless requested.
- LaTeX sources are authoritative for finalized reports. PDF outputs and `.fdb_latexmk` files are checked in — when regenerating, keep build artifacts out of commits unless explicitly asked.
- Python scripts in `QT207/` are lightweight test runners or helpers; many are empty/stubbed. Check file content before assuming behavior.

Important files to reference when editing or extending
- `QT207/qt207.tex` — example LaTeX assignment (complete source).
- `old/Presentation/presentation.tex` and `old/Presentation/presentation.pdf` — canonical slide deck and generated PDF.
- `qft_on_qc/Question-1.ipynb` and other notebooks under `old/` — experiments and Qiskit examples.
- `QT207/stadium_test_runner.py` and `old/stadium_test_runner.py` — small runner scripts (often empty/stubbed); use these only after inspecting contents.

Integration points & external dependencies
- No top-level package manifest (no `requirements.txt`, `pyproject.toml`, or `environment.yml`) — infer dependencies from notebooks and `.tex` (LaTeX packages). Notebooks mention Qiskit in older files.
- LaTeX requires a TeX distribution (pdflatex/latexmk) on the host.

Editing guidance for AI agents (concise, actionable)
- When changing notebooks, preserve outputs only when updating figures intentionally; otherwise, clear outputs and leave a short summary cell explaining the change.
- When adding Python dependencies, add a `requirements.txt` at the project root listing packages and pinned versions inferred from imports (ask if unsure). Example minimal: `numpy matplotlib qiskit`.
- For LaTeX changes: edit the `.tex` source under the appropriate directory and rebuild with `latexmk -pdf` to verify no compilation errors. If you add images, place them near the source (`old/Presentation/fig-*.png` pattern used).
- Run a quick sanity check after edits: open the affected notebook or run `pdflatex` on the modified `.tex`. Report any build errors and suggest fixes.

Examples from the codebase (patterns to follow)
- Notebook-first edits: modify `qft_on_qc/Question-1.ipynb` when updating QFT experiment code and re-run the kernel to regenerate plots.
- LaTeX build: `old/Presentation/` contains `presentation.fdb_latexmk` and `presentation.pdf`; regenerate with `latexmk -pdf presentation.tex` to reproduce `presentation.pdf`.

Do NOT assume
- There is a test harness or CI configured for running notebooks or Python scripts.
- Any particular virtualenv or package manager is preferred — prefer venv + pip unless the user specifies otherwise.

If something is unclear
- Ask where new runnable code should go (script or notebook). Ask whether generated build artifacts (PDFs, auxiliary LaTeX files) should be committed.

Feedback request
- Tell me if you'd like this file to (a) include an explicit `requirements.txt` generated from the notebooks, (b) add lightweight CI to run notebook cells / LaTeX builds, or (c) expand on any particular notebook or script for immediate fixes.
