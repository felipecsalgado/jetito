from jetito.ebeam import pointing

file = "Images/Betatron/ebeam_pointing/877_A=2.8_ND08+ND04_195157_.png"
ebeam_anaylsis = pointing.pointing_analysis(file, rescale=True,
                                            image_calib=5/54,
                                            d_target_screen=0.9)

ebeam_anaylsis.RemoveMeanBackground(left=660, 
                                    right=700, 
                                    top=790, 
                                    bottom=820,
                                    verbose=True)

ebeam_anaylsis.crop_image(left=462, right=630, top=620, bottom=730, 
              save_file="results/ebeam/pointing/ebeam_pointing_cropped_betatron.png", verbose=True)

ebeam_anaylsis.calculate_pointing_parameters(init_guess=(5e3, 1.2, 1.2, 385, 175),
                                 output=True, verbose=False)

ebeam_anaylsis.getDivergence(verbose=True)

ebeam_anaylsis.calculate_charge(screen_yield = 7.61e9,
                                camera_calib = 6.7,
                                transmission_loss = 0.95 * 0.063,
                                lens_focal_length = 25,
                                lens_fnumber = 2.8,
                                dist_cam_screen = 1.15,
                                verbose=True)

ebeam_anaylsis.plot_fields_fit(save_file="results/ebeam/pointing/ebeam_pointing_analysis_betatron.png",
                   xlim=(-5, 5), ylim=(-5, 5), clim=(0,8e2), cmap='magma')


