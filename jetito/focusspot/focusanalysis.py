"""
This package provides functions to perfom a focus spot analysis from experiments
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
import cv2
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


class focusspot_analysis:
    """
    Provides functions to perform focus spot analysis and also the ability to plot the
    analyzed focus spots.
    Examples are provided on how to perform the data analysis in the test/focus_analysis folder.
    """

    def __init__(self, filename, image_calib=0.4, beam_energy=1.7, pulse_duration=30e-15,
                 **kwargs):
        """Contructor of the focusspot_analysis class.

        Args:
            filename (string): Path and file name of the near-field image to calculate
            the far-field distribution.
            image_calib (double, optional): Calibration of the near-field image in units
            of length/pixel (typical: um/pixels). Defaults to 0.4.
            beam_energy (double, optional): energy contained in the focused beam in Joules.
            Calibrated using power meter in the chamber (valid for jeti). Default = 1.7 J.
            pulse_duration (double, optional): pulse duration of the laser pulse in seconds.
        """

        if 'verbose' not in kwargs:
            verbose = False
        else:
            verbose = kwargs['verbose']
            kwargs.pop('verbose', None)

        self.file = filename
        self.image_calib = image_calib
        self.beam_energy = beam_energy
        self.pulse_duration = pulse_duration

        try:
            self.image_focus = cv2.imread(self.file, -1)

            if verbose:
                print("Image file loaded successfully!")
                print(self.image_focus.shape)

        except (RuntimeError, TypeError, NameError):
            print("Error loading the image file")

    def crop_image(self, left=580, right=830, top=600, bottom=740, save_file=None, **kwargs):
        """Crops the original focus image to an smaller part for later to perform the
        2D-Gaussian analyis of the spot.

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

        self.image_crop = self.image_focus[top:bottom, left:right]

        if verbose:
            print("Shape of the cropped image: %d x %d" % (self.image_crop.shape[0],
                                                           self.image_crop.shape[1]))

        if save_file is not None:
            cv2.imwrite(save_file, self.image_crop)
            if verbose:
                print("Saved cropped image at: " + save_file)

    def calculate_focus_parameters(self, init_guess=(100, 10, 10, 0, 0), **kwargs):
        """
        Calculates the focus spot parameters of the given image.
        A 2D-Gaussian is fit in the focus profile and the paramters:
        x0, y0, amplitude, sigma_x, sigma_y, rotation angle, fwhm_x, fwhm_y are
        calculated and stored in variables.
        If verbose = True, the fit parameters are also printed in the console.

        Args:
            init_guess (tuple, optional): Initial guess parameter for the fit
            function. (amplitude, sigma_x, sigma_y, theta, offset).
            Defaults to (100, 10, 10, 0, 0).

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

        x = np.linspace(-img_x_size/2, img_x_size/2, img_x_size) * self.image_calib
        y = np.linspace(-img_y_size/2, img_y_size/2, img_y_size) * self.image_calib
        self.XX, self.YY = np.meshgrid(x, y)

        try:
            # Get the max for initial guess
            sum_vertical = np.sum(self.image_crop, axis=0)
            idx_max_vertical = np.argmax(sum_vertical)

            sum_horizontal = np.sum(self.image_crop, axis=1)
            idx_max_horizontal = np.argmax(sum_horizontal)

            # plt.pcolormesh(self.XX, self.YY, self.image_crop)
            # plt.colorbar()
            # plt.savefig("results/focus_analysis/cropped.png", forecolor="white")

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

            self.major_sigma = self.major_fwhm / 2.35
            self.minor_sigma = self.minor_fwhm / 2.35

        except (RuntimeError, TypeError, NameError):
            print("Error while calculating the focus parameters")
            self.popt_fit = -1
            self.pcov_fit = -1

        return self.popt_fit, self.pcov_fit

    def getQfactor(self, **kwargs):
        """
        Calculates the q-factor of the far-field after a Fourier Transform is performed
        """

        if 'verbose' not in kwargs:
            verbose = False
        else:
            verbose = kwargs['verbose']
            kwargs.pop('verbose', None)

        self.image_crop = np.double(self.image_crop)
        self.image_crop -= 500
        self.image_crop[self.image_crop < 0] = 0
        idx_q_factor = np.argwhere(self.image_crop > (self.popt_fit[0]/2))
        #idx_q_factor = np.argwhere(self.image_crop > (np.max(self.image_crop)*0.37))
        q_factor_list = [self.image_crop[idx_q_factor[i][0],idx_q_factor[i][1]] for i in range(idx_q_factor.shape[0])]

        #plt.imshow(self.image_crop[idx_q_factor])
        #plt.show()

        #from PIL import Image
        #im = Image.fromarray(self.image_crop)
        #im.save("your_file.tiff")

        self.counts_within_FWHM = np.sum(q_factor_list)

        self.total_counts = np.sum(self.image_crop)

        if verbose:
            print("Counts within FWHM = %d counts" % (self.counts_within_FWHM))
            print("Total counts within ROI = %d counts" % (self.total_counts))

        self.q_factor = self.counts_within_FWHM/self.total_counts * 100

        if verbose:
            print("Q-factor = %.1f %%" % self.q_factor)

        return self.q_factor

    def getIntensity(self):
        """
        Calculates the Intensity in W/cm2 based on the FWHM and energy of the beam.
        Only possible if the FWHM and focus parameters were calculated alredy.
        """
        if self.counts_within_FWHM is None:
            print("Error in calculating the Intensity!")
            print("Please run calculate_focus_parameters and getQfactor methods before!")
            return -1
        else:
            # calculate the energy per pixel
            energy_per_pixel = self.beam_energy / self.total_counts
            energy_fwhm = energy_per_pixel * self.counts_within_FWHM
            area_cm2 = np.pi * (self.major_fwhm / 2.) * (self.minor_fwhm / 2.) * 1.e-8
            self.intensity = energy_fwhm / (self.pulse_duration * area_cm2)
            return self.intensity

        return 0

    def plot_fields_fit(self, save_file=None, xlim=None, ylim=None, clim=None,
                        show_plot=True, **kwargs):
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
            show_plot (boolean, optional): Defines if the plot will of the focus will be shown.
            Defaults to True.
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

        text_legend = ("$\sigma_x$ = %.1f \u03BCm\n" +
                       "$2\sigma_x$ = %.1f \u03BCm\n" +
                       "FWHM$_x$ = %.1f \u03BCm\n") % (self.popt_fit[3],
                                                       2*self.popt_fit[3],
                                                       2.35*self.popt_fit[3])
        _ = ax.text(0.07, 0.135, text_legend, horizontalalignment='left', color='white',
                    fontsize=10, weight='bold', verticalalignment='center',
                    transform=ax.transAxes)

        text_legend2 = ("$\sigma_y$ = %.1f \u03BCm\n" +
                        "$2\sigma_y$ = %.1f \u03BCm\n" +
                        "FWHM$_y$ = %.1f \u03BCm\n" +
                        "q-factor = %.1f %%") % (self.popt_fit[4],
                                                 2*self.popt_fit[4],
                                                 2.35*self.popt_fit[4],
                                                 self.q_factor)
        _ = ax.text(0.55, 0.135, text_legend2, horizontalalignment='left', color='white',
                    fontsize=10, weight='bold', verticalalignment='center', transform=ax.transAxes)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        if clim:
            im.set_clim(clim)

        ax.set_xlabel('\u03BCm')
        ax.set_ylabel('\u03BCm')

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        cb = fig.colorbar(im, cax=cax, orientation='vertical')
        cb.ax.set_ylabel('Camera counts')

        plt.tight_layout()

        if save_file is not None:
            fig.savefig(save_file, dpi=450, facecolor='white', format='png', bbox_inches='tight')

        if show_plot:
            plt.show()
