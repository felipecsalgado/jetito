import os
from jetito.ebeam import pointing

basepath = os.path.abspath(os.path.dirname(__file__)) + "/"
file = basepath + "Images/Emittance_Experiment_2022/ebeam_pointing/" + \
    "shot00450.tif"
ebeam_anaylsis = pointing.pointing_analysis(file, image_calib=18.5e-3)

ebeam_anaylsis.crop_image(left=300, right=1260, top=1350, bottom=1970,
                          save_file=basepath +
                          "results/ebeam/pointing/ebeam_pointing_cropped.png",
                          verbose=True)

ebeam_anaylsis.calculate_pointing_parameters(init_guess=(500, 2.214, 3.7, 385, 175),
                                             output=True, verbose=False)

ebeam_anaylsis.getDivergence(verbose=True)

ebeam_anaylsis.calculate_charge(screen_yield=1.325e9*2,
                                camera_calib=1.4,
                                transmission_loss=1,
                                lens_focal_length=28,
                                lens_fnumber=1.4,
                                dist_cam_screen=15e-2,
                                verbose=True)

ebeam_anaylsis.plot_fields_fit(save_file=basepath +
                               "results/ebeam/pointing/ebeam_pointing_analysis.png",
                               xlim=(-6, 6), ylim=(-5, 5),
                               clim=(0, 600), cmap='magma')
