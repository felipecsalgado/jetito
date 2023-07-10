import cv2
from jetito.focusspot import farfield
#
import os
print(os.getcwd())

"""
Test file showing the basic commands to calculate the far-field
of a laser beam using a near-field image.
"""
basepath = os.path.abspath(os.path.dirname(__file__)) + "/"
file = basepath + "Images/JETi-200/JETi200_near_field.png"

FF = farfield.farfield_calculator(file, image_calib=0.147e-3)

FF.add_points(dim=2**13)

FF.calculate_far_field(wavelength=800e-9, distance=2.5, norm=True)

FF.crop_fields()

FF.plot_fields(save_file=basepath +
               "results/farfield/NF_FF_full_beam_Jeti.png")

FF.ff_2dgaussian_fit()

FF.getQfactor()

FF.plot_fields_fit(save_file=basepath +
                   "results/farfield/NF_FF_full_beam_Jeti_ElecAcc2023_fit.png")
