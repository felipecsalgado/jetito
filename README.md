# Jetito

A python package to help evaluate data from experiment at the JETi-200 laser system at the Helmholtz Institute Jena (HIJ) and the nonlinear Breit-Wheeler pair production at the Centre for Advanced Laser Applications (CALA).

**For working examples** using real experimental data, please see the folder `examples/`

## Installation

Jetito requires `python3`.

The latest version of jetito should be installed directly from github using `pip` (python package manager):

`pip install --user git+https://github.com/felipecsalgado/jetito.git`

**After installing** the jetito package, users can import the package using `import jetito`.

**Note:**

* Depending on your system python setup, `pip` may default to `python2`. In this case, you will need use substitute it to `pip3` and make sure that jetito is installed for `python3`.
* The `--user` flag allows the package to be installed without `root` privileges by installing it into your user profile.

## Roadmap

The following functionalities are to be added in the future:

* Electron-beam spectrum analysis
  * Include new setup with new calibrated spectrometer
* Emittance laser grating analysis
* Breit-Wheeler pair-production experimental data analysis
  * CsI detectors
  * Beam energy, pointing, and charge

## Changelog

Please check the [changelog](changelog.md) file.

# Credits

**Felipe C. Salgado**
(Friedrich-Schiller-Universität Jena, Helmholtz Institute Jena)

Github: [https://github.com/felipecsalgado](https://github.com/felipecsalgado)

* Farfield and laser focus
* Electron beam
  * Pointing analysis
  * Emittance from pepper pots (code integration in the package)
* Pair-Production CALA experiment
  * Calorimeter signal anaylsis (filtering, Bayesian signal+bakground anaylsis)

**Alperen Kozan**
(Friedrich-Schiller-Universität Jena, Helmholtz Institute Jena)

* Electron beam
  * Emittance from pepper pots (base code creator)

## Acknowledgements

Thanks to our colleagues at the FSU Jena and Helmholtz Institute Jena for providing few experimental data for benchmarking the code. Special thanks to A. Seidel, Harsh, and [D. Seipt](https://github.com/danielseipt).

# Contribute

If you feel like to contribute to the jetito package, please send us your pull requests, bug fixes and new features!

# License

![CC BY-NC-SA 4.0](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).
