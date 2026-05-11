# SAAM Report and Sales Pitch — Overleaf instructions

This folder contains the two LaTeX deliverables:

- `report.tex` — full project report (~10 pages once compiled, well under
  the 30-page cap).
- `sales_pitch.tex` — one-page institutional pitch.

Both documents pull figures from `../outputs_part1/` and `../outputs_part2/`
through the `\graphicspath` directive in `report.tex`.

## Compiling on Overleaf

1. On Overleaf, create a new project → **Upload Project (zip)**.
2. Zip the entire repository root (or at minimum the folders `report/`,
   `outputs_part1/`, `outputs_part2/`).
3. Open `report/report.tex` as the main document (Overleaf will detect it
   automatically once you mark it as "main file" from the menu).
4. Compiler: **pdfLaTeX**. Default Overleaf settings work; the document
   uses only stable, widely available packages (booktabs, siunitx,
   natbib, hyperref).

`sales_pitch.tex` is independent and compiles the same way.

## Local compilation (optional)

If you have a TeX distribution (MacTeX, TeX Live, MiKTeX), from the
project root:

```bash
cd report
pdflatex report.tex && pdflatex report.tex   # twice for the ToC
pdflatex sales_pitch.tex
```

The second pass is needed so that `\tableofcontents` picks up the
correct page numbers.
