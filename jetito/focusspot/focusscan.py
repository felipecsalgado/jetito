"""
This package provides functions to perfom a scan over the focus spot for a given dataset from experiments.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Jetito imports
# Gaussian fit
import jetito.gaussian_fit as gf
# Focu analysis
from jetito.focusspot import focusanalysis
from tqdm.notebook import tqdm

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

ml = MultipleLocator(2)

class focusscan:
    """
    Provides an easy way to perform focus scan analysis for a set of images of the same dataset.
    """

    def __init__(self, files_list, image_calib=0.4, beam_energy=1.7, pulse_duration=23e-15,
                     im_left=735, im_right=1880, im_top=900, im_bottom=1772,
                     fit_p0=(40e3, 10, 10, 0, 0),
                     plot_focus_output_folder=None,
                     plot_xlim=(-80, 80), plot_ylim=(-80, 80), plot_clim=(0, 50e3)):

        # Define the set of images to be analyzed
        self.files_list = files_list

        # Data parameters
        self.image_calib = image_calib
        self.beam_energy = beam_energy
        self.pulse_duration = pulse_duration

        # Image crop settings
        self.im_left = im_left
        self.im_right = im_right
        self.im_top = im_top
        self.im_bottom = im_bottom

        # Gaussian fit parameter
        self.fit_p0 = fit_p0

        # Plot function
        self.plot_focus_output_folder = plot_focus_output_folder
        self.plot_xlim = plot_xlim
        self.plot_ylim = plot_ylim
        self.plot_clim = plot_clim



    def runScan(self):
        """
            Perform the focus analysis for the set of images given in the class constructor with the given
            laser, image_crop, Gaussian fit and plot parameters.
        """

        self.major_fwhm = []
        self.minor_fwhm = []

        self.major_sigma = []
        self.minor_sigma = []

        self.q_factor_percentage = []
        self.intensity_wcm2 = []

        # Run a loop over the focus images
        for idx in tqdm(range(0, len(self.files_list))):
            FA = focusanalysis.focusspot_analysis(self.files_list[idx], image_calib=self.image_calib,
                                                beam_energy=self.beam_energy, pulse_duration=self.pulse_duration)

            FA.crop_image(left=self.im_left, right=self.im_right, top=self.im_top, bottom=self.im_bottom,
                        save_file=None, verbose=False)

            FA.calculate_focus_parameters(init_guess=self.fit_p0,
                                        output=False, verbose=False)

            _= FA.getQfactor(verbose=False)

            _= FA.getIntensity()

            if FA.major_fwhm < 30 and FA.minor_fwhm < 30:
                self.major_fwhm.append(FA.major_fwhm)
                self.minor_fwhm.append(FA.minor_fwhm)

                self.q_factor_percentage.append(FA.q_factor)
                self.intensity_wcm2.append(FA.intensity)

                self.major_sigma.append(FA.major_sigma)
                self.minor_sigma.append(FA.minor_sigma)

            #Save the analysis

            if self.plot_focus_output_folder is not None:
                filename = self.plot_focus_output_folder + str(int(idx)) + ".png"

                FA.plot_fields_fit(save_file=filename,
                                    xlim=self.plot_xlim, ylim=self.plot_ylim, clim=self.plot_clim,
                                    cmap='coolwarm')

        print("End of loop analysis!!")

    def getStats(self):
        """
        Calculate the statistics of the calculated parameters:
        Average and standard deviation of the focus, intensity and q-factor.
        """

        print("")
        print("Results from the analysis....\n")
        print("FWHM values:")
        mean_major_fwhm = np.mean(self.major_fwhm)
        std_major_fwhm = np.std(self.major_fwhm)
        print("Major fwhm = ( %.5f +/- %.5f ) um" % (mean_major_fwhm, std_major_fwhm))

        mean_minor_fwhm = np.mean(self.minor_fwhm)
        std_minor_fwhm = np.std(self.minor_fwhm)
        print("Minor fwhm = ( %.5f +/- %.5f ) um" % (mean_minor_fwhm, std_minor_fwhm))

        print("")
        print("RMS sigma values:")
        mean_major_sigma = np.mean(self.major_sigma)
        std_major_sigma = np.std(self.major_sigma)
        print("Major sigma = ( %.5f +/- %.5f ) um" % (mean_major_sigma, std_major_sigma))

        mean_minor_sigma = np.mean(self.minor_sigma)
        std_minor_sigma = np.std(self.minor_sigma)
        print("Minor sigma = ( %.5f +/- %.5f ) um" % (mean_minor_sigma, std_minor_sigma))

        print("")
        mean_q_sigma = np.mean(self.q_factor_percentage)
        std_q_sigma = np.std(self.q_factor_percentage)
        print("Q-factor = ( %.5f +/- %.5f ) %%" % (mean_q_sigma, std_q_sigma))

        mean_intensity = np.mean(self.intensity_wcm2) * 1e-18
        std_intensity = np.std(self.intensity_wcm2) * 1e-18
        print("Intensity = ( %.5f +/- %.5f ) x 10^{18} W/cm2" % (mean_intensity, std_intensity))