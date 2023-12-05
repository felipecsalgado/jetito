# Changelog of Jetito

## v0.5.5
* Added new ebeam module: espec analysis for the long electron spectrometer at JETi200 TA1.
  * Class and methods to post-process, join three images and calculate the dQ/dE of the long spectrometer.
  * Exaple jupyter notebook attached.

## v0.5.0
* Added new module: breitwheeler
  * Routines for analyising data from the pair-production experiment at CALA
  * Submodule: calorimeter. How to analyise the data from the calorimeters and separated the background noise from the signal using Bayesian statistics

## v0.4.0
* Added root mean square (rms) emittance evaluation from pepper pots using experimental data, or GEANT4 simulations.
  * New examples for the evaluation in the `examples/` folder.

## v0.3.0
* Added calculation of focus spot and its parameters from near-field.
* Added electron beam divergence and charge capabilities.
* Fixed plot problems for focus and farfield analysis, beam charge and divergence.
