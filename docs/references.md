# References

Web links are **access-date dependent**. Retrieved 2026-09-11. DOIs are omitted unless taken from the publisher page at access time — do not invent them.

## Software and project overviews

1. NASA. *PuMA: Porous Microstructure Analysis*. GitHub repository.  
   [https://github.com/nasa/puma](https://github.com/nasa/puma)

2. NASA. *PuMA documentation*.  
   [https://puma-nasa.readthedocs.io/](https://puma-nasa.readthedocs.io/)

3. NASA Advanced Supercomputing Division. *Microscale modeling of thermal protection system materials* (SC22 project overview).  
   [https://www.nas.nasa.gov/SC22/research/project27.html](https://www.nas.nasa.gov/SC22/research/project27.html)

## Publications (attribution; not a claim that this repo implements those models)

4. Ferguson, J. C., Panerai, F., Borner, A., and Mansour, N. N. (2018).  
   “PuMA: The Porous Microstructure Analysis software.” *SoftwareX*.  
   Journal page via ScienceDirect / *SoftwareX* (search the title at the publisher; do not rely on an unverified DOI).

5. Ferguson, J. C., Panerai, F., Lachaud, J., Martin, A., Bailey, S. C. C., and Mansour, N. N. (2016).  
   “Modeling the oxidation of low-density carbon fiber material based on micro-tomography.” *Carbon*.  
   Journal page via *Carbon* / ScienceDirect (search the title at the publisher).

## How this repository uses the above

- **Attribution only** for inspiration and for optional future PuMA comparisons.
- The numerical baseline here is a **simplified** diffusion–reaction + finite-difference conductivity model. It is **not** a reimplementation of Ferguson et al. (2016) oxidation kinetics and **not** a wrapper around PuMA solvers.
- Material constants in YAML files are illustrative unless a source is added in this file with a full citation.

## Standard numerical references (textbook-level, not TPS-specific)

Closed-form 1-D diffusion in a slab and series/parallel thermal-resistor rules are classical; see any conduction or transport textbook (e.g. Crank, *The Mathematics of Diffusion*; Incropera & DeWitt, *Fundamentals of Heat and Mass Transfer*). They are used only for solver verification in `docs/verification.md`.
