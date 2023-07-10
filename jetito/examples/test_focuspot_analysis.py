from jetito.focusspot import focusanalysis

file = ("Images/TA1_ElecAcc2023/focus/157_Fz=" +
        "-6mm_600mJ_680_ND10ND10ND10ND03Laser+ND40Target191320_.png")
FA = focusanalysis.focusspot_analysis(file, image_calib=0.4)

FA.crop_image(left=750, right=1150, top=700, bottom=950,
              save_file="results/focus_analysis/focus_cropped.png", verbose=True)

FA.calculate_focus_parameters(init_guess=(250, 10, 10, 0, 0),
                              output=False, verbose=False)

FA.getQfactor()

ax = FA.plot_fields_fit(save_file="results/focus_analysis/focus_analysis.png",
                        xlim=(-40, 60), ylim=(-50, 50),
                        clim=(0, 150), cmap='jet')
