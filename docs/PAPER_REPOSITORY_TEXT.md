# Paper text and versioned citation

Use the final public URL and the actual published release tag. The following is a template, not a claim that a repository has already been published.

Suggested concise statement for the current coverage:

> Code for the generated-flow NL/LIN/FRZ protocols, initial Fourier checkpoints, phase-mask identifiers, and the processed data and scripts used to reconstruct the figures are available at [repository URL], release [tag]. The source isotropic-turbulence fields are available from JHTDB.

If retaining an explicit scope sentence in the Supplement:

> The archived figure workflow reproduces the reported panels from processed records. Complete static-field recomputation requires the original cutouts and the upstream modules identified in the repository documentation.

Do not write “all simulations and raw data are fully reproducible” until the complete static pipeline and full dynamic rerun have been verified. State the actual released coverage once, in the availability statement or Supplement, rather than adding a disclaimer to every result paragraph.

Example LaTeX after publication:

```latex
Code for the generated-flow NL/LIN/FRZ protocols, initial Fourier
checkpoints, phase-mask identifiers, and the processed data and scripts
used to reconstruct the figures are available at
\url{https://github.com/OWNER/REPOSITORY}, release \texttt{v0.1.0}.
The source isotropic-turbulence fields are available from JHTDB.
```

Replace OWNER/REPOSITORY with the actual address. A tagged release and its commit identify the paper version; record an archive DOI only after one exists. Author and affiliation information remains unset, as requested. `CITATION.cff.example` is inactive until those details are resolved.
