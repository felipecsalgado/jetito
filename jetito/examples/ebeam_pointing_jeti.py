from jetito.ebeam import pointing
import os

basepath = os.path.abspath(os.path.dirname(__file__)) + "/"
file = basepath + "images/JETi-200/ebeam_pointing/" + \
    "386_LWFA_Zelle_He5N2_Pointing_ESpeklow_182542_.png"
ebeam_anaylsis = pointing.pointing_analysis(file, rescale=True,
                                            image_calib=5/54,
                                            d_target_screen=1.87)

ebeam_anaylsis.crop_image(left=670, right=1140, top=305, bottom=690,
                          save_file=basepath +
                          "results/ebeam/pointing/ebeam_pointing_cropped_jeti.png",
                          verbose=True)

ebeam_anaylsis.calculate_pointing_parameters(init_guess=(5e3, 1.2, 1.2, 385, 175),
                                             output=True, verbose=False)

ebeam_anaylsis.getDivergence(verbose=True)

ebeam_anaylsis.calculate_charge(screen_yield=8.25e9,
                                camera_calib=6.7,
                                transmission_loss=1,
                                lens_focal_length=25,
                                lens_fnumber=4,
                                dist_cam_screen=40e-2,
                                verbose=True)

ebeam_anaylsis.plot_fields_fit(save_file=basepath + "results/ebeam/pointing/" +
                               "ebeam_pointing_analysis_jeti.png",
                               xlim=(-6, 6), ylim=(-7, 1),
                               clim=(0, 1.5e3), cmap='magma')
