# Change Log

## [v0.14.1](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.14.1) (2018-01-10)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.14.0...v0.14.1)

**Implemented enhancements:**

- ENH: Add simplex tools [\#183](https://github.com/QuantEcon/QuantEcon.jl/pull/183) ([cc7768](https://github.com/cc7768))

**Fixed bugs:**

- WIP: Fix utility functions to branch [\#184](https://github.com/QuantEcon/QuantEcon.jl/pull/184) ([cc7768](https://github.com/cc7768))

**Closed issues:**

- Is PDF export of the lecture notes currently broken? [\#197](https://github.com/QuantEcon/QuantEcon.jl/issues/197)
- Drop ecdf? [\#186](https://github.com/QuantEcon/QuantEcon.jl/issues/186)
- Add simplex\_grid and simplex\_index [\#178](https://github.com/QuantEcon/QuantEcon.jl/issues/178)
- Cannot precompile QuantEcon with Julia Version 0.7.0-DEV.553 \(2017-06-11 22:13 UTC\) [\#159](https://github.com/QuantEcon/QuantEcon.jl/issues/159)
- compute\_fixed\_point warning [\#150](https://github.com/QuantEcon/QuantEcon.jl/issues/150)
- PY: ivp [\#34](https://github.com/QuantEcon/QuantEcon.jl/issues/34)

**Merged pull requests:**

- Add `next\_k\_array!` and `k\_array\_rank`. [\#199](https://github.com/QuantEcon/QuantEcon.jl/pull/199) ([shizejin](https://github.com/shizejin))
- Add quadrature method [\#198](https://github.com/QuantEcon/QuantEcon.jl/pull/198) ([Shunsuke-Hori](https://github.com/Shunsuke-Hori))
- QuantEcon.ecdf calls StatsBase.ecdf [\#196](https://github.com/QuantEcon/QuantEcon.jl/pull/196) ([a-parida12](https://github.com/a-parida12))
- Remove warning message when converged with max\_iter [\#195](https://github.com/QuantEcon/QuantEcon.jl/pull/195) ([a-parida12](https://github.com/a-parida12))
- \[WIP\] Add quantile and quadrature method [\#194](https://github.com/QuantEcon/QuantEcon.jl/pull/194) ([Shunsuke-Hori](https://github.com/Shunsuke-Hori))
- BUG: Fix `simplex\_grid` [\#193](https://github.com/QuantEcon/QuantEcon.jl/pull/193) ([oyamad](https://github.com/oyamad))
- ENH: added `@def\_sim` macro that I use in a few research projects [\#188](https://github.com/QuantEcon/QuantEcon.jl/pull/188) ([sglyon](https://github.com/sglyon))
- Fix deprecations [\#187](https://github.com/QuantEcon/QuantEcon.jl/pull/187) ([femtocleaner[bot]](https://github.com/apps/femtocleaner))
- FIX: Remove 0.0 from `candidates` in `solve\_discrete\_riccati` [\#185](https://github.com/QuantEcon/QuantEcon.jl/pull/185) ([oyamad](https://github.com/oyamad))

## [v0.14.0](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.14.0) (2017-10-21)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.13.0...v0.14.0)

**Implemented enhancements:**

- Model tools: Add various utility functions [\#179](https://github.com/QuantEcon/QuantEcon.jl/pull/179) ([cc7768](https://github.com/cc7768))

**Merged pull requests:**

- update syntax [\#182](https://github.com/QuantEcon/QuantEcon.jl/pull/182) ([Shunsuke-Hori](https://github.com/Shunsuke-Hori))
- Sl/sparse ddp q [\#181](https://github.com/QuantEcon/QuantEcon.jl/pull/181) ([sglyon](https://github.com/sglyon))
- BUG: evaluating a LinInterp at non-consecutive columns was buggy [\#177](https://github.com/QuantEcon/QuantEcon.jl/pull/177) ([sglyon](https://github.com/sglyon))
- WIP: Add short doc on golden\_method [\#176](https://github.com/QuantEcon/QuantEcon.jl/pull/176) ([cc7768](https://github.com/cc7768))
- TEST: update travis to install apt deps [\#175](https://github.com/QuantEcon/QuantEcon.jl/pull/175) ([sglyon](https://github.com/sglyon))
- Stability tests for LSS class [\#171](https://github.com/QuantEcon/QuantEcon.jl/pull/171) ([vgregory757](https://github.com/vgregory757))

## [v0.13.0](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.13.0) (2017-08-19)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.12.1...v0.13.0)

**Closed issues:**

- Minimum version numbers on Optim and NLopt [\#169](https://github.com/QuantEcon/QuantEcon.jl/issues/169)

**Merged pull requests:**

- Add minimum version for Optim and NLopt [\#170](https://github.com/QuantEcon/QuantEcon.jl/pull/170) ([Shunsuke-Hori](https://github.com/Shunsuke-Hori))
- Add discreteVAR of Farmer and Toda [\#163](https://github.com/QuantEcon/QuantEcon.jl/pull/163) ([Shunsuke-Hori](https://github.com/Shunsuke-Hori))

## [v0.12.1](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.12.1) (2017-07-20)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.12.0...v0.12.1)

**Closed issues:**

- Question: change `draw\(::DiscreteRV\)` to `rand\(::DiscreteRV\)`? [\#162](https://github.com/QuantEcon/QuantEcon.jl/issues/162)

**Merged pull requests:**

- Latex rendering in docs + other improvements [\#168](https://github.com/QuantEcon/QuantEcon.jl/pull/168) ([natashawatkins](https://github.com/natashawatkins))
- Allow for noisy observations in LSS class [\#167](https://github.com/QuantEcon/QuantEcon.jl/pull/167) ([vgregory757](https://github.com/vgregory757))
- add latex rendering in arma.jl [\#166](https://github.com/QuantEcon/QuantEcon.jl/pull/166) ([natashawatkins](https://github.com/natashawatkins))
- add latex support to documenter [\#165](https://github.com/QuantEcon/QuantEcon.jl/pull/165) ([natashawatkins](https://github.com/natashawatkins))

## [v0.12.0](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.12.0) (2017-07-13)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.11.0...v0.12.0)

**Closed issues:**

- Julia 0.6 dependency warnings [\#140](https://github.com/QuantEcon/QuantEcon.jl/issues/140)

**Merged pull requests:**

- ENH: use rand instead of draw for drv sampling  [\#164](https://github.com/QuantEcon/QuantEcon.jl/pull/164) ([sglyon](https://github.com/sglyon))
- Discrete Markov chain estimation [\#161](https://github.com/QuantEcon/QuantEcon.jl/pull/161) ([cc7768](https://github.com/cc7768))

## [v0.11.0](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.11.0) (2017-07-07)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.10.1...v0.11.0)

**Merged pull requests:**

- ENH: LinInterp for mulitple functions [\#160](https://github.com/QuantEcon/QuantEcon.jl/pull/160) ([sglyon](https://github.com/sglyon))

## [v0.10.1](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.10.1) (2017-05-01)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.10.0...v0.10.1)

**Merged pull requests:**

- BUG: Fix MVNSampler [\#158](https://github.com/QuantEcon/QuantEcon.jl/pull/158) ([oyamad](https://github.com/oyamad))

## [v0.10.0](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.10.0) (2017-04-28)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.9.0...v0.10.0)

**Closed issues:**

- Changed meaning of .+= in Julia 0.5 [\#132](https://github.com/QuantEcon/QuantEcon.jl/issues/132)
- ENH: parallelize the bellman\_operator \(and similar\) [\#13](https://github.com/QuantEcon/QuantEcon.jl/issues/13)

**Merged pull requests:**

- Adding MVNSampler and method to draw for singular covariance matrix [\#157](https://github.com/QuantEcon/QuantEcon.jl/pull/157) ([Shunsuke-Hori](https://github.com/Shunsuke-Hori))
- Update README.md [\#155](https://github.com/QuantEcon/QuantEcon.jl/pull/155) ([Balinus](https://github.com/Balinus))
- Sl/0.6 [\#154](https://github.com/QuantEcon/QuantEcon.jl/pull/154) ([sglyon](https://github.com/sglyon))

## [v0.9.0](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.9.0) (2017-02-05)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.8.0...v0.9.0)

**Implemented enhancements:**

- Merge in CompEcon function approximation tools [\#45](https://github.com/QuantEcon/QuantEcon.jl/issues/45)

**Closed issues:**

- 1D linear interpolation convenience function [\#152](https://github.com/QuantEcon/QuantEcon.jl/issues/152)
- MarkovChain: state\_values related [\#119](https://github.com/QuantEcon/QuantEcon.jl/issues/119)
- Move in some numerical tools [\#112](https://github.com/QuantEcon/QuantEcon.jl/issues/112)

**Merged pull requests:**

- ENH: add linear interpolation routine [\#153](https://github.com/QuantEcon/QuantEcon.jl/pull/153) ([sglyon](https://github.com/sglyon))
- add back 0..4 testing [\#151](https://github.com/QuantEcon/QuantEcon.jl/pull/151) ([tkelman](https://github.com/tkelman))
- Starting new documentation [\#138](https://github.com/QuantEcon/QuantEcon.jl/pull/138) ([stephenbnicar](https://github.com/stephenbnicar))

## [v0.8.0](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.8.0) (2016-11-14)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.7.0...v0.8.0)

**Closed issues:**

- qnwnorm not type stable [\#148](https://github.com/QuantEcon/QuantEcon.jl/issues/148)

**Merged pull requests:**

- ENH: make quad routines type stable [\#149](https://github.com/QuantEcon/QuantEcon.jl/pull/149) ([sglyon](https://github.com/sglyon))
- Fix typo in gridmake and add method for a tuple of ints [\#147](https://github.com/QuantEcon/QuantEcon.jl/pull/147) ([sglyon](https://github.com/sglyon))
- ENH: added iterators for MarkovChain simulation [\#146](https://github.com/QuantEcon/QuantEcon.jl/pull/146) ([sglyon](https://github.com/sglyon))

## [v0.7.0](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.7.0) (2016-10-25)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.6.0...v0.7.0)

**Closed issues:**

- Loss of significance in Rouwenhorst AR\(1\) approximation? [\#145](https://github.com/QuantEcon/QuantEcon.jl/issues/145)
- tauchen broken by 0.5? [\#142](https://github.com/QuantEcon/QuantEcon.jl/issues/142)
- HEADS UP: Breaking change in LightGraphs API [\#141](https://github.com/QuantEcon/QuantEcon.jl/issues/141)

**Merged pull requests:**

- edited warnings [\#144](https://github.com/QuantEcon/QuantEcon.jl/pull/144) ([jstac](https://github.com/jstac))
- Minor issue with warnings [\#143](https://github.com/QuantEcon/QuantEcon.jl/pull/143) ([jstac](https://github.com/jstac))

## [v0.6.0](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.6.0) (2016-09-05)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.5.1...v0.6.0)

**Merged pull requests:**

- Update markov\_approx.jl [\#139](https://github.com/QuantEcon/QuantEcon.jl/pull/139) ([colbec](https://github.com/colbec))

## [v0.5.1](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.5.1) (2016-08-23)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.5.0...v0.5.1)

**Closed issues:**

- Forcing 2d arrays in robustlq.jl [\#134](https://github.com/QuantEcon/QuantEcon.jl/issues/134)
- QuantEcon failing to load on Ubuntu LTS 16.04 [\#108](https://github.com/QuantEcon/QuantEcon.jl/issues/108)

**Merged pull requests:**

- Redesign of the MarkovChain simulation methods [\#137](https://github.com/QuantEcon/QuantEcon.jl/pull/137) ([vgregory757](https://github.com/vgregory757))
- Relaxed getZ type [\#135](https://github.com/QuantEcon/QuantEcon.jl/pull/135) ([albep](https://github.com/albep))
- Use dirname\(@\_\_FILE\_\_\) instead of Pkg.dir [\#133](https://github.com/QuantEcon/QuantEcon.jl/pull/133) ([tkelman](https://github.com/tkelman))
- Upgraded code to handle 1 dimensional cases [\#130](https://github.com/QuantEcon/QuantEcon.jl/pull/130) ([albep](https://github.com/albep))
- add minimum version requirement for Compat.view [\#129](https://github.com/QuantEcon/QuantEcon.jl/pull/129) ([tkelman](https://github.com/tkelman))
- bug fix in Markov chain simulation [\#128](https://github.com/QuantEcon/QuantEcon.jl/pull/128) ([amckay](https://github.com/amckay))

## [v0.5.0](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.5.0) (2016-07-15)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.4.2...v0.5.0)

**Implemented enhancements:**

- Add `stationary\_distributions` [\#115](https://github.com/QuantEcon/QuantEcon.jl/pull/115) ([oyamad](https://github.com/oyamad))

**Closed issues:**

- MAT is not automatically installed [\#114](https://github.com/QuantEcon/QuantEcon.jl/issues/114)
- Add docstrings to functions in src/util.jl [\#106](https://github.com/QuantEcon/QuantEcon.jl/issues/106)
- Clean out dependencies [\#98](https://github.com/QuantEcon/QuantEcon.jl/issues/98)
- Update docs [\#90](https://github.com/QuantEcon/QuantEcon.jl/issues/90)
- TEST: run code from examples/notebooks directories with tests [\#61](https://github.com/QuantEcon/QuantEcon.jl/issues/61)

**Merged pull requests:**

- Fill in policies backwards [\#127](https://github.com/QuantEcon/QuantEcon.jl/pull/127) ([albep](https://github.com/albep))
- Use Primes package [\#126](https://github.com/QuantEcon/QuantEcon.jl/pull/126) ([davidanthoff](https://github.com/davidanthoff))
- Fixed Matrix Multiplication in DDP [\#123](https://github.com/QuantEcon/QuantEcon.jl/pull/123) ([MaximilianJHuber](https://github.com/MaximilianJHuber))
- Fix indices overwr ddp [\#122](https://github.com/QuantEcon/QuantEcon.jl/pull/122) ([albep](https://github.com/albep))
- Fixing bug related to w\_path and some related comments [\#116](https://github.com/QuantEcon/QuantEcon.jl/pull/116) ([albep](https://github.com/albep))
- Modify src/QuantEcon.jl [\#110](https://github.com/QuantEcon/QuantEcon.jl/pull/110) ([myuuuuun](https://github.com/myuuuuun))
- Fix gridmake ordering, add docstrings to util.jl closes \#106 [\#107](https://github.com/QuantEcon/QuantEcon.jl/pull/107) ([sglyon](https://github.com/sglyon))
- Add docs  [\#105](https://github.com/QuantEcon/QuantEcon.jl/pull/105) ([ranjanan](https://github.com/ranjanan))

## [v0.4.2](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.4.2) (2016-03-27)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.4.1...v0.4.2)

**Implemented enhancements:**

- ENH: use constrained optimizer in jv.jl when one is available in pure julia [\#4](https://github.com/QuantEcon/QuantEcon.jl/issues/4)

**Closed issues:**

- \#People link in Readme redirecting to broken HTML site [\#101](https://github.com/QuantEcon/QuantEcon.jl/issues/101)
- Support sparse transition matrix in `MarkovChain` [\#97](https://github.com/QuantEcon/QuantEcon.jl/issues/97)
- Move to new Base.Test framework [\#89](https://github.com/QuantEcon/QuantEcon.jl/issues/89)
- New `meshgrid` method? [\#87](https://github.com/QuantEcon/QuantEcon.jl/issues/87)
- Delete Old Branches [\#81](https://github.com/QuantEcon/QuantEcon.jl/issues/81)
- Question: remove eigen and lu methods for stationary distribution? [\#75](https://github.com/QuantEcon/QuantEcon.jl/issues/75)
- ENH: move BetaBinomial to Distributions.jl [\#67](https://github.com/QuantEcon/QuantEcon.jl/issues/67)
- perf: for some reason coleman\_operator is slower than bellman\_operator in ifp.jl [\#11](https://github.com/QuantEcon/QuantEcon.jl/issues/11)

**Merged pull requests:**

- Implement faster gridmake [\#104](https://github.com/QuantEcon/QuantEcon.jl/pull/104) ([sglyon](https://github.com/sglyon))
- Add SA pairs to ddp [\#100](https://github.com/QuantEcon/QuantEcon.jl/pull/100) ([sglyon](https://github.com/sglyon))
- Refactor markov [\#99](https://github.com/QuantEcon/QuantEcon.jl/pull/99) ([sglyon](https://github.com/sglyon))
- don't need to include compat in tests anymore [\#93](https://github.com/QuantEcon/QuantEcon.jl/pull/93) ([rawls238](https://github.com/rawls238))
- update tests to use BaseTestNext instead of FactCheck Fixes \#89 [\#92](https://github.com/QuantEcon/QuantEcon.jl/pull/92) ([rawls238](https://github.com/rawls238))
- use BetaBinomial from Distributions.jl instead of QuantEcon Fixes \#67 [\#91](https://github.com/QuantEcon/QuantEcon.jl/pull/91) ([rawls238](https://github.com/rawls238))
- Change type in meshgrid function [\#88](https://github.com/QuantEcon/QuantEcon.jl/pull/88) ([cc7768](https://github.com/cc7768))
- ddp: Re-implement 3-dim matrix \* vector with reshape [\#86](https://github.com/QuantEcon/QuantEcon.jl/pull/86) ([oyamad](https://github.com/oyamad))
- Add random\_discrete\_dp [\#82](https://github.com/QuantEcon/QuantEcon.jl/pull/82) ([oyamad](https://github.com/oyamad))
- BUG: Fix bug in probvec [\#80](https://github.com/QuantEcon/QuantEcon.jl/pull/80) ([oyamad](https://github.com/oyamad))
- Move the models [\#79](https://github.com/QuantEcon/QuantEcon.jl/pull/79) ([sglyon](https://github.com/sglyon))
- Add Discrete DP [\#78](https://github.com/QuantEcon/QuantEcon.jl/pull/78) ([mmcky](https://github.com/mmcky))
- MarkovChain simulate: Match with python version [\#77](https://github.com/QuantEcon/QuantEcon.jl/pull/77) ([oyamad](https://github.com/oyamad))
- random\_probvec: Remove temporary array [\#76](https://github.com/QuantEcon/QuantEcon.jl/pull/76) ([oyamad](https://github.com/oyamad))

## [v0.4.1](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.4.1) (2015-10-31)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.3.0...v0.4.1)

## [v0.3.0](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.3.0) (2015-10-29)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.2.1...v0.3.0)

**Merged pull requests:**

- Update arellano [\#74](https://github.com/QuantEcon/QuantEcon.jl/pull/74) ([sglyon](https://github.com/sglyon))
- ENH: added uncertainty traps code [\#73](https://github.com/QuantEcon/QuantEcon.jl/pull/73) ([sglyon](https://github.com/sglyon))
- Arellano [\#72](https://github.com/QuantEcon/QuantEcon.jl/pull/72) ([cc7768](https://github.com/cc7768))
- change unicode to ascii for latex parsing [\#71](https://github.com/QuantEcon/QuantEcon.jl/pull/71) ([mmcky](https://github.com/mmcky))
- Random MarkovChain [\#63](https://github.com/QuantEcon/QuantEcon.jl/pull/63) ([oyamad](https://github.com/oyamad))

## [v0.2.1](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.2.1) (2015-08-27)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.2.0...v0.2.1)

**Closed issues:**

- ERROR: LoadError: LoadError: Invalid doc expression \[eigen\_solve,lu\_solve,gth\_solve\] [\#70](https://github.com/QuantEcon/QuantEcon.jl/issues/70)
- Docstring causes LoadErrors in 0.4 [\#69](https://github.com/QuantEcon/QuantEcon.jl/issues/69)
- BUG: rename src/distributions.jl [\#66](https://github.com/QuantEcon/QuantEcon.jl/issues/66)
- LoadErrors in 0.4 [\#65](https://github.com/QuantEcon/QuantEcon.jl/issues/65)
- using QuantEcon failing on v0.4 [\#64](https://github.com/QuantEcon/QuantEcon.jl/issues/64)
- Makorv chain: Lecture has to be revised [\#57](https://github.com/QuantEcon/QuantEcon.jl/issues/57)

**Merged pull requests:**

- BUG: Fix a bug in mc\_tools 2 [\#68](https://github.com/QuantEcon/QuantEcon.jl/pull/68) ([oyamad](https://github.com/oyamad))
- fixed mc solution notebook [\#60](https://github.com/QuantEcon/QuantEcon.jl/pull/60) ([jstac](https://github.com/jstac))

## [v0.2.0](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.2.0) (2015-07-21)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.1.2...v0.2.0)

**Implemented enhancements:**

- Update lqcontrol to handle cross product terms [\#16](https://github.com/QuantEcon/QuantEcon.jl/issues/16)

**Fixed bugs:**

- BUG: mc\_tools non-determinism [\#49](https://github.com/QuantEcon/QuantEcon.jl/issues/49)

**Closed issues:**

- using QuantEcon results in error [\#55](https://github.com/QuantEcon/QuantEcon.jl/issues/55)
- Error message in mc\_sample\_path [\#52](https://github.com/QuantEcon/QuantEcon.jl/issues/52)
- PY: graph\_tools [\#32](https://github.com/QuantEcon/QuantEcon.jl/issues/32)

**Merged pull requests:**

- Update MarkovChain Tests [\#59](https://github.com/QuantEcon/QuantEcon.jl/pull/59) ([ZacCranko](https://github.com/ZacCranko))
- New features for mc\_tools [\#56](https://github.com/QuantEcon/QuantEcon.jl/pull/56) ([ZacCranko](https://github.com/ZacCranko))
- Update mc\_convergence\_plot.jl [\#54](https://github.com/QuantEcon/QuantEcon.jl/pull/54) ([ranjanan](https://github.com/ranjanan))
- BUG: fixed bugs in mc\_tools [\#53](https://github.com/QuantEcon/QuantEcon.jl/pull/53) ([sglyon](https://github.com/sglyon))

## [v0.1.2](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.1.2) (2015-06-24)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.1.1...v0.1.2)

## [v0.1.1](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.1.1) (2015-06-24)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.1.0...v0.1.1)

**Fixed bugs:**

- BUG: new tuple updates in base break jv.jl [\#42](https://github.com/QuantEcon/QuantEcon.jl/issues/42)
- BUG: fix ifp\_solutions [\#8](https://github.com/QuantEcon/QuantEcon.jl/issues/8)

**Closed issues:**

- Add docstrings [\#47](https://github.com/QuantEcon/QuantEcon.jl/issues/47)
- Deal with existing PRs [\#46](https://github.com/QuantEcon/QuantEcon.jl/issues/46)
- typo in ifp.jl code [\#38](https://github.com/QuantEcon/QuantEcon.jl/issues/38)
- PY: robustlq [\#37](https://github.com/QuantEcon/QuantEcon.jl/issues/37)
- PY: rank\_nullspace [\#36](https://github.com/QuantEcon/QuantEcon.jl/issues/36)
- PY: matrix\_eqn [\#35](https://github.com/QuantEcon/QuantEcon.jl/issues/35)
- PY: cartesian [\#31](https://github.com/QuantEcon/QuantEcon.jl/issues/31)
- QuantEcon Julia: a list of the routines used [\#24](https://github.com/QuantEcon/QuantEcon.jl/issues/24)
- REF: new interpolation package [\#15](https://github.com/QuantEcon/QuantEcon.jl/issues/15)

**Merged pull requests:**

- Docstrings [\#50](https://github.com/QuantEcon/QuantEcon.jl/pull/50) ([sglyon](https://github.com/sglyon))
- LQ Cross Product & Revamp [\#48](https://github.com/QuantEcon/QuantEcon.jl/pull/48) ([ZacCranko](https://github.com/ZacCranko))
- Fixes: Trying this again... Accidentally pushed last PR to master [\#44](https://github.com/QuantEcon/QuantEcon.jl/pull/44) ([cc7768](https://github.com/cc7768))
- Makes QuantEcon.jl importable [\#43](https://github.com/QuantEcon/QuantEcon.jl/pull/43) ([cc7768](https://github.com/cc7768))
- Mc burn [\#40](https://github.com/QuantEcon/QuantEcon.jl/pull/40) ([sglyon](https://github.com/sglyon))
- Test mutating [\#39](https://github.com/QuantEcon/QuantEcon.jl/pull/39) ([sglyon](https://github.com/sglyon))
- Update README.md file ... [\#29](https://github.com/QuantEcon/QuantEcon.jl/pull/29) ([mmcky](https://github.com/mmcky))

## [v0.1.0](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.1.0) (2015-02-13)
[Full Changelog](https://github.com/QuantEcon/QuantEcon.jl/compare/v0.0.1...v0.1.0)

**Closed issues:**

- PY: gth\_solve [\#33](https://github.com/QuantEcon/QuantEcon.jl/issues/33)
- Tauchen routines have different names [\#25](https://github.com/QuantEcon/QuantEcon.jl/issues/25)
- ENH: remove PyPlot.jl dependency [\#10](https://github.com/QuantEcon/QuantEcon.jl/issues/10)
- ENH: quad routines [\#6](https://github.com/QuantEcon/QuantEcon.jl/issues/6)
- ENH: markov type [\#1](https://github.com/QuantEcon/QuantEcon.jl/issues/1)

**Merged pull requests:**

- TEST: remove julia dev from travis.yml [\#27](https://github.com/QuantEcon/QuantEcon.jl/pull/27) ([sglyon](https://github.com/sglyon))
- Add gth support to mc\_compute\_stationary [\#26](https://github.com/QuantEcon/QuantEcon.jl/pull/26) ([ZacCranko](https://github.com/ZacCranko))
- Implement the GTH algorithm [\#23](https://github.com/QuantEcon/QuantEcon.jl/pull/23) ([oyamad](https://github.com/oyamad))
- Markov [\#21](https://github.com/QuantEcon/QuantEcon.jl/pull/21) ([sglyon](https://github.com/sglyon))
- Make Dict syntax compatible to v0.4 [\#19](https://github.com/QuantEcon/QuantEcon.jl/pull/19) ([nilshg](https://github.com/nilshg))
- Fixing links to updated quant-econ website in solutions [\#18](https://github.com/QuantEcon/QuantEcon.jl/pull/18) ([sanguineturtle](https://github.com/sanguineturtle))
- Change Dict Syntax to Julia 0.4 [\#17](https://github.com/QuantEcon/QuantEcon.jl/pull/17) ([nilshg](https://github.com/nilshg))
- added part 1 solutions [\#14](https://github.com/QuantEcon/QuantEcon.jl/pull/14) ([jstac](https://github.com/jstac))

## [v0.0.1](https://github.com/QuantEcon/QuantEcon.jl/tree/v0.0.1) (2014-08-29)
**Closed issues:**

- RFAC: implement a `Models` module to mimic python [\#7](https://github.com/QuantEcon/QuantEcon.jl/issues/7)
- ENH: update ricatti to use new method [\#2](https://github.com/QuantEcon/QuantEcon.jl/issues/2)

**Merged pull requests:**

- Lqnash [\#12](https://github.com/QuantEcon/QuantEcon.jl/pull/12) ([cc7768](https://github.com/cc7768))
- ARMA done [\#9](https://github.com/QuantEcon/QuantEcon.jl/pull/9) ([jstac](https://github.com/jstac))
- Matrix eqn [\#5](https://github.com/QuantEcon/QuantEcon.jl/pull/5) ([cc7768](https://github.com/cc7768))



\* *This Change Log was automatically generated by [github_changelog_generator](https://github.com/skywinder/Github-Changelog-Generator)*