(TeX-add-style-hook
 "chap1"
 (lambda ()
   (LaTeX-add-labels
    "eq:gleichung1"
    "fig:abbildung1")
   (LaTeX-add-environments
    '("eqnbox" LaTeX-env-args ["argument"] 0))))

