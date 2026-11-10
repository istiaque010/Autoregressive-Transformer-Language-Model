# Article: Decoding Strategies and Error Analysis for Small Autoregressive Transformer Language Models in Story Generation

Target journal: IEEE Access.

| File | What it is |
|---|---|
| `main.tex` | the article (LaTeX, generic `IEEEtran` journal class) |
| `references.bib` | bibliography (IEEE style through `IEEEtran.bst`) |
| `figures/` | the 6 figures as vector PDF (used by LaTeX) and 300-dpi PNG (for viewing) |
| `analysis/paper_analysis.py` | recomputes every statistic and figure from the saved results and checkpoints |
| `analysis/paper_stats.json` | the statistics it produced (bootstrap and Wilson intervals, tests, self-NLL, nucleus sizes, error counts) |
| `Autoregressive Transformer Language_references.csv` | Consensus literature export used for the related work |

## Build

Overleaf: upload this folder (main.tex, references.bib, figures/) and compile with pdfLaTeX + BibTeX.
Locally (needs a TeX distribution): `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

Regenerate the numbers and figures (from the project root, about 5 minutes on the CPU):
`python project1_paper/analysis/paper_analysis.py`

## Moving to the official IEEE Access template

Download the IEEE Access LaTeX template from the IEEE Access author resources page, then copy the abstract,
keywords, sections, tables, figures and bibliography from `main.tex` into it. The body uses only standard packages
(amsmath, graphicx, booktabs, multirow, algorithm, algpseudocode), so no changes to the content should be needed.
IEEE Access also asks for author biographies and photographs at the end.

## Still required before submission

1. Author names, affiliations, funding, repository URL (placeholders in `[ ]`).
2. An independent human check of the 60 error labels (and ideally a second annotator, with Cohen's kappa).
   The current labels were produced with AI assistance; update Section VI-E and the Limitations if they change.
3. Recommended for reviewers: 2 more training seeds per model, more prompts, and hand labels for the baseline.
