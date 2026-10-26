# CSE 499A report

*Cross-Dataset Generalization and Robustness of Deepfake Detection: A Leave-One-Out Study with Explainability and Statistical Validation.* Kawser Sikder and Tasneem Abdullah Aurnobb, North South University (advisor: Dr. Riasat Khan).

| File | Contents |
|---|---|
| `main.tex` | The full report in one self-contained IEEE-style file (TikZ figures inline) |
| `refs.bib` | BibTeX source of the references (the bibliography is inlined in `main.tex`, so the build does not read it) |
| `slides_499a.pptx` | Final presentation (19 slides) |
| `flowchart.jpg` | Image export of the system flowchart (`main.tex` draws it in TikZ) |

## Building

Upload `main.tex` to Overleaf (pdfLaTeX), or build locally:

```bash
latexmk -pdf main.tex
```

All packages are in a standard TeX Live installation.
