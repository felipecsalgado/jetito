from jetito.focusspot import focusanalysis
import glob
import os
import natsort
import numpy as np

# Get the files from the focus experimental results folder

folder = 'Images/TA1_ElecAcc2023/focus/'
# list of images
files_list = glob.glob(folder + "*-6mm*.png")
files_list = natsort.natsorted(files_list, reverse=False)
print("Number of files in the folder: %d" % (len(files_list)))

# lists to calculate the mean and std
fwhm_x = []
fwhm_y = []
q_factor_percentage = []

# Run a loop over the focus images
for idx in range(0, len(files_list)):
    FA = focusanalysis.focusspot_analysis(files_list[idx], image_calib=0.4)

    FA.crop_image(left=750, right=1150, top=700, bottom=950,
                  save_file=None, verbose=False)

    FA.calculate_focus_parameters(init_guess=(250, 10, 10, 0, 0),
                                  output=False, verbose=False)

    FA.getQfactor()

    fwhm_x.append(FA.fwhm_x)
    fwhm_y.append(FA.fwhm_y)
    q_factor_percentage.append(FA.q_factor)

    # Save the analysis
    filename = "results/focus_analysis/shot_series/focus_analysis_" + str(int(idx)) + ".png"
    FA.plot_fields_fit(save_file=filename,
                       xlim=(-40, 60), ylim=(-50, 50), clim=(0, 150),
                       cmap='coolwarm')

    print("Post_processed: " + filename)

# Calculate the mean values and std
print("")
print("Results from the analysis....")
mean_fwhm_x = np.mean(fwhm_x)
std_fwhm_x = np.std(fwhm_x)
print("X-fwhm = ( %.3f +/- %.3f ) um" % (mean_fwhm_x, std_fwhm_x))

mean_fwhm_y = np.mean(fwhm_y)
std_fwhm_y = np.std(fwhm_y)
print("Y-fwhm = ( %.3f +/- %.3f ) um" % (mean_fwhm_y, std_fwhm_y))

mean_q_sigma = np.mean(q_factor_percentage)
std_q_sigma = np.std(q_factor_percentage)
print("Q-factor = ( %.3f +/- %.3f ) %%" % (mean_q_sigma, std_q_sigma))
