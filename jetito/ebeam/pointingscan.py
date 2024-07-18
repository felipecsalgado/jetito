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
from jetito.ebeam import pointing
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

class pointingscan:
    """
    Provides an easy way to perform focus scan analysis for a set of images of the same dataset.
    """

    def __init__(self, files_list, rescale=True, image_calib=0.0223, d_target_screen=1.03,
                 im_left=700, im_right=1210, im_top=355, im_bottom=830, im_save_file=None,
                 fit_p0=(5e3, 0.7, 0.5, 0, 50),
                 screen_yield=7.61e9,
                 camera_calib=1/0.092,
                 transmission_loss=0.97**3,
                 lens_focal_length=108,
                 lens_fnumber=8,
                 dist_cam_screen=43e-2,
                 filename_save="e-pointing/results/",
                 im_xlim=(-2,2),
                 im_ylim=(-2.5,2.5),
                 im_clim=(0, 4e3)):

        # Initialization
        self.files_list = files_list
        self.rescale = rescale
        self.image_calib = image_calib
        self.d_target_screen = d_target_screen

        # Image crop
        self.im_left = im_left
        self.im_right = im_right
        self.im_top = im_top
        self.im_bottom = im_bottom
        self.im_save_file = im_save_file

        # Gaussian fit
        self.fit_p0 = fit_p0

        # Charge calibration
        self.screen_yield = screen_yield
        self.camera_calib = camera_calib
        self.transmission_loss = transmission_loss
        self.lens_focal_length = lens_focal_length
        self.lens_fnumber = lens_fnumber
        self.dist_cam_screen = dist_cam_screen

        #Plot files
        self.filename_save = filename_save
        self.im_xlim = im_xlim
        self.im_ylim = im_ylim
        self.im_clim = im_clim

    def runScan(self):
        """
        Scan the pointing screen
        """
        self.rms_div_major_mrad = []
        self.rms_div_minor_mrad = []

        self.fwhm_div_major_mrad = []
        self.fwhm_div_minor_mrad = []

        self.charge_array = []


        for idx in tqdm(range(0, len(self.files_list))):
            print(idx)
            ebeam_anaylsis = pointing.pointing_analysis(filename=self.files_list[idx],
                                                        rescale=self.rescale,
                                                        image_calib=self.camera_calib, # mm/px
                                                        d_target_screen=self.d_target_screen) # meters

            ebeam_anaylsis.crop_image(left=self.im_left,
                                      right=self.im_right,
                                      top=self.im_top,
                                      bottom=self.im_bottom,
                                      save_file=self.im_save_file,
                                      verbose=False)

            ebeam_anaylsis.calculate_pointing_parameters(init_guess=self.fit_p0,
                                                        output=False,
                                                        verbose=False)

            #ebeam_anaylsis.getDivergence(verbose=False)

            ebeam_anaylsis.calculate_charge(screen_yield=self.screen_yield,
                                        camera_calib=self.camera_calib,
                                        transmission_loss=self.transmission_loss,
                                        lens_focal_length=self.lens_focal_length,
                                        lens_fnumber=self.lens_fnumber,
                                        dist_cam_screen=self.dist_cam_screen,
                                        verbose=False)

            #if self.filename_save is not None:
            fname = self.filename_save + str(idx) + ".png"
            ebeam_anaylsis.plot_fields_fit(save_file=fname,
                                            xlim=self.im_xlim, ylim=self.im_ylim,
                                            clim=self.im_clim, cmap='magma')

            self.rms_div_major_mrad.append(ebeam_anaylsis.div_major_rms * 1e3)
            self.rms_div_minor_mrad.append(ebeam_anaylsis.div_minor_rms * 1e3)

            self.fwhm_div_major_mrad.append(ebeam_anaylsis.div_major_fwhm * 1e3)
            self.fwhm_div_minor_mrad.append(ebeam_anaylsis.div_minor_fwhm  * 1e3)

            self.charge_array.append(ebeam_anaylsis.beam_charge_PC)

        print("End of loop analysis!!")

    def getStats(self):
        """
        Calculate the statistics of the calculated parameters of the electorn pointing:
        Average and standard deviation of the rms divergence, fwhm divergence, and
        charge.
        """
        print("")
        print("Results from the analysis....\n")
        print("FWHM values:")
        print("FWHM divergence major = (%.5f +- %.5f) mrad" % (np.mean(self.fwhm_div_major_mrad),
                                                              np.std(self.fwhm_div_major_mrad)))
        print("FWHM divergence minor = (%.5f +- %.5f) mrad" % (np.mean(self.fwhm_div_minor_mrad),
                                                              np.std(self.fwhm_div_minor_mrad)))

        print("")
        print("RMS sigma values:")
        print("rms divergence major = (%.5f +- %.5f) mrad" % (np.mean(self.rms_div_major_mrad),
                                                              np.std(self.rms_div_major_mrad)))
        print("rms divergence minor = (%.5f +- %.5f) mrad" % (np.mean(self.rms_div_minor_mrad),
                                                              np.std(self.rms_div_minor_mrad)))

        print("")
        print("Charge = (%.5f +- %.5f) pC" % (np.mean(self.charge_array),
                                              np.std(self.charge_array)))