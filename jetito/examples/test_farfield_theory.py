from jetito.focusspot import farfield
import os

print(os.getcwd())

"""
Test file showing the basic commands to calculate the theoretical
focus spot parameters of a laser beam using Gaussian optics.
"""

# JETi-200 beam
theory_spot = farfield.farfield_theory(beam_diameter=12,
                                       focal_length=2.5,
                                       wavelength=800e-9)

# Compute the parameters
theory_spot.compute_focus_parameters()

print("")
theory_spot.set_beam_diameter(10)

print("")
theory_spot.set_wavelength(750e-9)

print("")
theory_spot.set_focal_length(5)

# Emittance experiment
print("")
print("Emittance experiment")
theory_spot = farfield.farfield_theory(beam_diameter=6,
                                       focal_length=1,
                                       wavelength=800e-9)

# Compute the parameters
theory_spot.compute_focus_parameters()

# TA2 experiment
print("")
print("TA2")
theory_spot = farfield.farfield_theory(beam_diameter=12,
                                       focal_length=0.18,
                                       wavelength=800e-9)

# Compute the parameters
theory_spot.compute_focus_parameters()
