# Adversarial Machine Learning Research Paper - LaTeX Files

This repository contains the LaTeX source files for the paper "The Evolution of Adversarial Machine Learning Research".

## Files

- `main.tex` - Main LaTeX document containing the paper content
- `references.bib` - BibTeX bibliography file with all references

## How to Use with Overleaf

1. Go to [Overleaf](https://www.overleaf.com)
2. Create a new project (New Project → Upload Project)
3. Upload both `main.tex` and `references.bib` files
4. Compile the document (Overleaf will automatically detect it's a LaTeX document)

## Compiling Locally

If you want to compile locally instead of using Overleaf:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Note

This is a draft version - the content is preserved exactly as written in the original document.
