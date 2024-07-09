import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
import cv2
import png  # pypng
import imageio
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Jetito imports
# Gaussian fit
import jetito.gaussian_fit as gf

# Configures the plots
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 22
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)            # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
ml = MultipleLocator(2)


class pointing_analysis:

    def __init__(self, filename, rescale=False, image_calib=18.3e-3, d_target_screen=1.45):
        """
        Contructor of the pointing_analysis class.

        Args:
            filename (string): Path and file name of the near-field image to calculate the
            far-field distribution.
            image_calib (double, optional): Calibration of the near-field image in units
            of length/pixel (typical: mm/pixels). Defaults to 18.5e-3.
            d_target_screen (double, optional): Distance from the target to the screen
            where the pointing image was recorded (typical: meters). Defaults to 1.45.
        """

        self.file = filename
        self.image_calib = image_calib
        self.dist_target_screen = d_target_screen

        try:
            if rescale:
                self.image_pointing = self.readpng(self.file)
            else:
                self.image_pointing = cv2.imread(self.file, -1)

            print("Image file loaded successfully!")
            print(self.image_pointing.shape[0])

        except (RuntimeError, TypeError, NameError):
            print("Error loading the image file")

    def readpng(self, filename, hotpixelremove=False):
        '''
        Reads a png file and returns appropriate count vales, even if a bit depth
        other than 8 or 16 is used. An example this might be needed is having a
        12-bit png recorded from a 12-bit camera using LabViews IMAQ toolset.
        In this case the PIL (python image library) fails to retrieve the
        original count values.
        '''

        meta = png.Reader(filename)
        meta.preamble()
        significant_bits = ord(meta.sbit)
        ret = imageio.imread(filename)
        ret >>= 16 - significant_bits
        if hotpixelremove:
            import scipy.ndimage
            ret = scipy.ndimage.morphology.grey_opening(ret, size=(3, 3))
        return np.float64(ret)

    def RemoveMeanBackground(self, left=660, right=700, top=790, bottom=820, **kwargs):
        """
        Removes the background from the image assuming that it is uniform.
        It takes the mean of a portion of the screen and subtracts it from the original image.
        Negative values are then truncated to zero.

        Args:
            left (int, optional): Pixel position for starting cropping the image from the left.
            Defaults to 660.
            right (int, optional): Pixel position for starting cropping the image from the right.
            Defaults to 700.
            top (int, optional): Pixel position for starting cropping the image from the top.
            Defaults to 790.
            bottom (int, optional): Pixel position for starting cropping the image from the
            bottom. Defaults to 820.
        """

        if 'verbose' not in kwargs:
            verbose = False
        else:
            verbose = kwargs['verbose']
            kwargs.pop('verbose', None)

        image_crop_bkg = self.image_pointing[top:bottom, left:right]
        mean_bkg = np.mean(image_crop_bkg)
        self.image_pointing = self.image_pointing - mean_bkg

        # truncate to zero
        self.image_pointing[self.image_pointing < 0] = 0

        if verbose:
            print("")
            print("Mean background = %.1f counts" % mean_bkg)

    def crop_image(self, left=100, right=1260, top=795, bottom=1970, save_file=None, **kwargs):
        """Crops the original pointing image to an smaller part for later to perform the
        2D-Gaussian analyis of the ebeam spot.

        Args:
            left (int, optional): Pixel position for starting cropping the image from the left.
            Defaults to 580.
            right (int, optional): Pixel position for starting cropping the image from the right.
            Defaults to 830.
            top (int, optional): Pixel position for starting cropping the image from the top.
            Defaults to 600.
            bottom (int, optional): Pixel position for starting cropping the image from the
            bottom. Defaults to 740.
            save_file (string, optional): Path of the file to save the cropped focus.
            Please provide this path. Defaults to None.
        """

        if 'verbose' not in kwargs:
            verbose = False
        else:
            verbose = kwargs['verbose']
            kwargs.pop('verbose', None)

        if verbose:
            print("")
            print("Crop the image")

        self.image_crop = self.image_pointing[top:bottom, left:right]
        self.image_crop = self.image_crop

        if verbose:
            print("Shape of the cropped image: %d x %d" % (self.image_crop.shape[0],
                                                           self.image_crop.shape[1]))

        if save_file is not None:
            cv2.imwrite(save_file, self.image_crop)
            if verbose:
                print("Saved cropped image at: " + save_file)

    def calculate_pointing_parameters(self, init_guess=(500, 2.214, 3.7, 385, 175), **kwargs):
        """
        Calculates the ebeam parameters of the given ebeam pointing images
        A 2D-Gaussian is fit in the focus profile and the parameters:
        x0, y0, amplitude, sigma_x, sigma_y, rotation angle, fwhm_x, fwhm_y are calculated
        and stored in variables.
        If verbose = True, the fit parameters are also printed in the console.

        Args:
            init_guess (tuple, optional): Initial guess parameter for the fit function.
            (amplitude, sigma_x, sigma_y, theta, offset). Defaults to (100, 10, 10, 0, 0).

        Returns:
            tuple: popt and pcov of the 2D-Gaussian fit
        """

        # initial guess
        # (amplitude, sigma_x, sigma_y, theta, offset)
        # Manage *kwargs
        if 'output' not in kwargs:
            kwargs['output'] = False

        if 'verbose' not in kwargs:
            verbose = False
        else:
            verbose = kwargs['verbose']
            kwargs.pop('verbose', None)

        img_x_size = self.image_crop.shape[1]
        img_y_size = self.image_crop.shape[0]

        x = np.linspace(img_x_size/2, -img_x_size/2, img_x_size) * self.image_calib
        y = np.linspace(-img_y_size/2, img_y_size/2, img_y_size) * self.image_calib
        self.XX, self.YY = np.meshgrid(x, y)

        try:
            # Get the max for initial guess
            sum_vertical = np.sum(self.image_crop, axis=0)
            idx_max_vertical = np.argmax(sum_vertical)

            sum_horizontal = np.sum(self.image_crop, axis=1)
            idx_max_horizontal = np.argmax(sum_horizontal)

            # Used for debugging
            # plt.pcolormesh(self.XX, self.YY, self.image_crop,  vmin=0, vmax=500)
            # plt.colorbar()
            # plt.savefig("results/ebeam/pointing/cropped.png", forecolor="white")

            if verbose:
                print("Maximum at: x = %.3f um and y = %.3f um" % (self.XX[idx_max_horizontal,
                                                                           idx_max_vertical],
                                                                   self.YY[idx_max_horizontal,
                                                                           idx_max_vertical]))
            # Get the centerr by fitting a 2D Gaussian
            # initial_guess = (25e3,idx_max_vertical,idx_max_horizontal,20,20,0,0, 0)
            # amplitude, xo, yo, sigma_x, sigma_y, theta, offset

            if verbose:
                print("")
                print("2D Gaussian fit in the FF distribution starting....")

            self.popt_fit, self.pcov_fit = gf.fit2d(XX=self.XX,
                                                    YY=self.YY,
                                                    ZZ=self.image_crop,
                                                    initial_guess=(init_guess[0],
                                                                   self.XX[idx_max_horizontal,
                                                                           idx_max_vertical],
                                                                   self.YY[idx_max_horizontal,
                                                                           idx_max_vertical],
                                                                   init_guess[1],
                                                                   init_guess[2],
                                                                   init_guess[3],
                                                                   init_guess[4]),
                                                    **kwargs)
            if verbose:
                print("")
                print("2D Gaussian fit completed!")

            self.fwhm_x = 2.35*self.popt_fit[3]
            self.fwhm_y = 2.35*self.popt_fit[4]

            if self.fwhm_x > self.fwhm_y:
                self.major_fwhm = self.fwhm_x
                self.minor_fwhm = self.fwhm_y
            else:
                self.major_fwhm = self.fwhm_y
                self.minor_fwhm = self.fwhm_x

            self.div_major_fwhm = np.arctan(self.major_fwhm * 1e-3 / self.dist_target_screen)
            self.div_minor_fwhm = np.arctan(self.minor_fwhm * 1e-3 / self.dist_target_screen)

            self.sigma_x = self.popt_fit[3]
            self.sigma_y = self.popt_fit[4]

            if self.sigma_x > self.sigma_y:
                self.major_rms = self.sigma_x
                self.minor_rms = self.sigma_y
            else:
                self.major_rms = self.sigma_y
                self.minor_rms = self.sigma_x

            self.div_major_rms = np.arctan(self.major_rms * 1e-3 / self.dist_target_screen)
            self.div_minor_rms = np.arctan(self.minor_rms * 1e-3 / self.dist_target_screen)

        except (RuntimeError, TypeError, NameError):
            print("Error while calculating the focus parameters")
            self.popt_fit = -1
            self.pcov_fit = -1

        return self.popt_fit, self.pcov_fit

    def getDivergence(self, **kwargs):
        """
        Calculates the divergence of the ebeam at the pointin screen according
        to the 2D-Gaussian fit of the pointing shot provided in the class constructor.

        Returns:
            tuple: div_fwhm_x, div_fwhm_y, div_sigma_x, div_sigma_y
        """

        if 'verbose' not in kwargs:
            verbose = False
        else:
            verbose = kwargs['verbose']
            kwargs.pop('verbose', None)

        #self.div_fwhm_x = np.arctan(self.major_fwhm/2 * 1e-3 / self.dist_target_screen)
        #self.div_major_y = np.arctan(self.major_y/2 * 1e-3 / self.dist_target_screen)

        #self.div_sigma_x = np.arctan(self.sigma_x * 1e-3 / self.dist_target_screen)
        #self.div_sigma_y = np.arctan(self.sigma_y * 1e-3 / self.dist_target_screen)

        if verbose:
            print("")
            print("Divergence of the ebeam")
            print("RMS Divergence major_sigma = %.3f mrad" % (self.div_major_rms * 1e3))
            print("RMS Divergence minor_sigma = %.3f mrad" % (self.div_minor_rms * 1e3))

            print("Half-angle divergence major FWHM = %.3f mrad" % (self.div_major_fwhm * 1e3))
            print("Half-angle divergence minor FWHM = %.3f mrad" % (self.div_minor_fwhm * 1e3))

        return self.div_major_fwhm, self.div_minor_fwhm, self.div_major_rms, self.div_minor_rms

    def plot_fields_fit(self, save_file=None, xlim=None, ylim=None, clim=None, **kwargs):
        """
        Method to plot the cropped focus spots with a contour of the 2D-Gaussian fit
        performed a priori.

        Args:
            save_file (string, optional): Path and file name, if image is supposed to be saved.
            Defaults to None.
            xlim (tuple, optional): Defines the limits of the X-axis of the 2D-plot.
            Defaults to None.
            ylim (tuple, optional): Defines the limits of the Y-axis of the 2D-plot.
            Defaults to None.
            clim (tuple, optional): Defines the limits of the colorbar of the 2D-plot.
            Defaults to None.
        """

        # Manage *kwargs
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'coolwarm'

        fig, axes1 = plt.subplots(1, 1, figsize=(6, 4))

        axes = iter(np.ravel(axes1))
        ax = next(axes)
        im = ax.pcolormesh(self.XX, self.YY, self.image_crop, **kwargs)

        Z_FWHM = gf.twoD_Gaussian((self.popt_fit[1] + self.popt_fit[3]*1.355/2.,
                                   self.popt_fit[2] + self.popt_fit[4]*1.355/2.),
                                  *self.popt_fit)
        # print(Z_FWHM)

        data_fitted = gf.twoD_Gaussian((self.XX, self.YY), *self.popt_fit)

        ax.contour(self.XX, self.YY,
                   data_fitted.reshape(self.image_crop.shape[0], self.image_crop.shape[1]),
                   levels=[Z_FWHM], colors=['black'])

        # Text legend 1: x-parameters
        #text_legend = ("$\sigma_x$ = %.1f mm\n" +
        #               "$2\sigma_x$ = %.1f mm\n" +
        #               "FWHM$_x$ = %.1f mm\n" +
        #               "rms div_x = %.2f mrad") % (self.popt_fit[3],
        #                                            2*self.popt_fit[3],
        #                                            2.35*self.popt_fit[3],
        #                                            self.div_sigma_x*1e3)
        #_ = ax.text(0.02, 0.125, text_legend, horizontalalignment='left', color='white',
        #            fontsize=8, weight='bold', verticalalignment='center',
        #            transform=ax.transAxes)

        # Text legend 2: y-parameters
        #text_legend2 = ("$\sigma_y$ = %.1f mm\n" +
        #                "$2\sigma_y$ = %.1f mm\n" +
        #                "FWHM$_y$ = %.1f mm\n" +
        #                "rms div_y = %.2f mrad") % (self.popt_fit[4],
        #                                             2*self.popt_fit[4],
        #                                             2.35*self.popt_fit[4],
        #                                             self.div_sigma_y*1e3)
        #_ = ax.text(0.6, 0.125, text_legend2, horizontalalignment='left', color='white',
        #            fontsize=8, weight='bold', verticalalignment='center',
        #            transform=ax.transAxes)

        # Text legend 3: charge
        try:
            text_legend3 = ("%.1f pC") % (self.beam_charge_PC)
            _ = ax.text(0.75, 0.95, text_legend3, horizontalalignment='left', color='white',
                        fontsize=8, weight='bold', verticalalignment='center',
                        transform=ax.transAxes)
        except (RuntimeError, TypeError, NameError):
            print("Please calculate the beam charge using the method calculate_charge.")

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        if clim:
            im.set_clim(clim)

        ax.set_xlabel('mm')
        ax.set_ylabel('mm')

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        cb = fig.colorbar(im, cax=cax, orientation='vertical')
        cb.ax.set_ylabel('Camera counts')

        plt.tight_layout()

        if save_file is not None:
            fig.savefig(save_file, dpi=450, facecolor='white', format='png', bbox_inches='tight')
        plt.show()

    def calculate_charge(self,
                         screen_yield=8.25e9,
                         camera_calib=6.5,
                         transmission_loss=5e-3,
                         lens_focal_length=25,
                         lens_fnumber=5.6,
                         dist_cam_screen=15e-2,
                         **kwargs):
        """
        Calculates the beam charge based on the experimental parameters passed as arguments.
        The charge is given in pC.

        Args:
            screen_yield (double, optional): Calibrated screen yield. Typically in
            photons/sr/pC. Defaults to 8.25e9 (Lanex Biomax MS).
            camera_calib (double, optional): Calibration of the camera photon/counts
            conversion. Typically in photons/counts. Defaults to 6.5.
            transmission_loss (double, optional): Transmission loss of the optical
            imaging system in the experiment. Accounts for the mirror and objective
            losses for example. Defaults to 5e-3.
            lens_focal_length (double, optional): Focal lens (in mm) of the objective
            CCTV/Macro lens used. Typically in mm. Defaults to 25.
            lens_fnumber (double, optional): F/number of the objective CCTV/Macro lens
            used in the experiment. Defaults to 5.6.
            dist_cam_screen (double, optional): Distance (in meters) of the objective
            lens and the screen in the experiment. Used for calculating the imaging solid angle.
            Typically in meters. Defaults to 15e-2.

        Yields:
            double: Analyzed beam charge given in pC.
        """

        if 'verbose' not in kwargs:
            verbose = False
        else:
            verbose = kwargs['verbose']
            kwargs.pop('verbose', None)

        if verbose:
            print("")
            print("Calculating charge....")
        # Calculates the sold angle
        diameter_lens_mm = lens_focal_length / lens_fnumber
        # print(diameter_lens_mm)
        area_lens_mm2 = np.pi * (diameter_lens_mm / 2)**2
        # print(area_lens_mm2)
        solid_angle = area_lens_mm2 / (dist_cam_screen * 1e3)**2
        # print(dist_cam_screen * 1e3)
        if verbose:
            print("Solid angle = %.2e sr" % solid_angle)

        # Charge calibration photons/count
        charge_calib = camera_calib / (screen_yield * solid_angle * transmission_loss)
        if verbose:
            print("Charge calibration = %.3e photons/count" % charge_calib)

        # Calculate the sum of counts int the ROI
        sum_counts_ROI = np.sum(self.image_crop)
        if verbose:
            print("Sum of counts of the ROI image = %d" % sum_counts_ROI)

        # Calculate the total charge of the beam based on the calibration
        self.beam_charge_PC = sum_counts_ROI * charge_calib
        if verbose:
            print("Total charge of the beam = %.3f pC" % self.beam_charge_PC)
