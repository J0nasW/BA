(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("lrt_thesis" "graybox" "envcountchap" "twoside" "deutsch")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("babel" "ngerman") ("amsmath" "fleqn") ("cases" "fleqn") ("natbib" "numbers") ("rotating" "figuresright") ("footmisc" "bottom") ("inputenc" "utf8")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "abstract"
    "chap1"
    "appendix"
    "lrt_thesis"
    "lrt_thesis10"
    "etex"
    "babel"
    "subfigure"
    "amsmath"
    "amsthm"
    "amssymb"
    "amsfonts"
    "cases"
    "tabularx"
    "caption"
    "fancyhdr"
    "multicol"
    "longtable"
    "booktabs"
    "natbib"
    "dsfont"
    "rotating"
    "ifthen"
    "trfsigns"
    "footmisc"
    "inputenc"
    "hyperref")
   (LaTeX-add-environments
    '("eqnbox" LaTeX-env-args ["argument"] 0))
   (LaTeX-add-bibliographies)))

