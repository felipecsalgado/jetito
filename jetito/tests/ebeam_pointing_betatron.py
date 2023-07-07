from jetito.ebeam import pointing

file = "Images/Betatron/ebeam_pointing/860_FB543.5+ND10+A=2.8_224917_.png"
ebeam_anaylsis = pointing.pointing_analysis(file, rescale=True,
                                            image_calib=5/54,
                                            d_target_screen=0.7)

ebeam_anaylsis.crop_image(left=415, right=665, top=565, bottom=820, 
              save_file="results/ebeam/pointing/ebeam_pointing_cropped_betatron.png", verbose=True)

ebeam_anaylsis.calculate_pointing_parameters(init_guess=(5e3, 1.2, 1.2, 385, 175),
                                 output=True, verbose=False)

ebeam_anaylsis.getDivergence(verbose=True)

ebeam_anaylsis.calculate_charge(screen_yield = 7.61e9,
                                camera_calib = 6.7,
                                transmission_loss = 0.95 * 0.1,
                                lens_focal_length = 12.5,
                                lens_fnumber = 2.8,
                                dist_cam_screen = 0.95,
                                verbose=True)

ebeam_anaylsis.plot_fields_fit(save_file="results/ebeam/pointing/ebeam_pointing_analysis_betatron.png",
                   xlim=(-3, 5), ylim=(-3, 5), clim=(0,8e2), cmap='magma')


