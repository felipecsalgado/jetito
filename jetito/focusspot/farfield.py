import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
import cv2
# Packages to calculate the far-field
from scipy.fftpack import fftshift, fft2, fftfreq

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


class farfield_calculator:
    """
    Class to calculate the far-field (2D Fourier Transform) of a near-field image.
    A sequence of commands in the correct order is required to fully compute the far-field.
    Please see the scrip test_farfield.py in the tests folder to have an example of how
    to work with this class.
    """

    def __init__(self, filename, image_calib=0.147e-3):
        """Contructor of the farfield_calculator class.

        Args:
            filename (string): Path and file name of the near-field image to calculate the
            far-field distribution.
            image_calib (double, optional): Calibration of the near-field image in units of
            length/pixel (typical: um/pixels). Defaults to 0.147e-3.
        """

        self.file = filename
        self.image_calib = image_calib

        try:
            self.img_nearfield = cv2.imread(self.file, cv2.IMREAD_GRAYSCALE)
            # cv2.imwrite("test.png", self.img_nearfield)
            print(self.img_nearfield.shape)
            print("Image file loaded successfully!")
        except (RuntimeError, TypeError, NameError):
            print("Error loading the image file")

    def add_points(self, dim=2**13, show_img=False, save_file="../tests/results/farfield/"):
        """
        Adds a border to the original near-field image. This increases the sampling points
        of the Fourier Transform afterwards.

        Args:
            dim (int, optional): Dimension of the resized image. Defaults to 2**13.
            show_img (bool, optional): Plots the resized image. Defaults to False.
            save_file (str, optional): Path and file name where to save the plot.
            Defaults to "../tests/results/farfield/".

        Returns:
            ndarray: Resized near-field image
        """
        # Add border
        top_bottom_px = (dim - self.img_nearfield.shape[0])//2
        left_right_px = (dim - self.img_nearfield.shape[1])//2
        self.near_field = cv2.copyMakeBorder(self.img_nearfield,
                                             top=top_bottom_px,
                                             bottom=top_bottom_px,
                                             left=left_right_px,
                                             right=left_right_px,
                                             borderType=cv2.BORDER_CONSTANT,
                                             value=0)

        # Generate the meshgrid of the near-field
        size = dim * self.image_calib
        x0 = np.linspace(-size/2, size/2, dim)  # 1292
        y0 = np.linspace(-size/2, size/2, dim)  # 964
        self.X, self.Y = np.meshgrid(x0, y0)

        # print('Resized image:')
        # print(np.asarray(self.im_mask_gray_resize.shape))

        # Get the field and not the intesity of the nearfield
        self.near_field = np.sqrt(self.near_field)

        if show_img:
            plt.imshow(self.near_field, cmap='magma')
            plt.tight_layout()
            plt.savefig(save_file + 'img_resized.png', dpi=450, facecolor='white', format='png')
            plt.show()

        return self.near_field

    def crop_center(self, img, cropx, cropy):
        y, x = img.shape
        startx = x//2 - cropx//2
        starty = y//2 - cropy//2
        return img[starty:starty+cropy, startx:startx+cropx]

    def calculate_far_field(self,
                            wavelength=800e-9,
                            distance=2.5,
                            norm=True):
        """
        Calculate the far field from the near field of a light beam using 2D FFT
        with interpolation.

        Args:
            wavelength (float, optional): Wavelength of the light beam in meters.
            Defaults to 800e-9.
            distance (float, optional): Distance from the near field to the far
            field in meters. Defaults to 2.5.
            norm (bool, optional): Normalize the far-field intensity. Defaults to True.

        Returns:
            ndarray: 2D array representing the far-field distribution.
        """

        try:
            print("")
            print("Calculating 2D Fourier Transform....")
            # Calculate the wave number
            k = 2 * np.pi / wavelength

            # Perform 2D FFT on the near field
            far_field = fftshift(fft2(fftshift(self.near_field)))

            # Calculate the frequencies in the frequency domain
            freq_x = fftshift(fftfreq(self.near_field.shape[0], d=1/self.image_calib))
            freq_y = fftshift(fftfreq(self.near_field.shape[1], d=1/self.image_calib))

            # Calculate the far field distribution
            far_field = np.exp(1j * k * distance) / (1j * wavelength * distance) * far_field
            self.far_field = np.abs(far_field)**2

            if norm:
                self.far_field = self.far_field/np.max(self.far_field)

            delta_fx = 1/(self.near_field.shape[0] * (self.image_calib)) * 2 * np.pi
            print("delta_fx = %.3e " % delta_fx)
            delta_fy = 1/(self.near_field.shape[1] * (self.image_calib)) * 2 * np.pi
            print("delta_fy = %.3e " % delta_fy)

            k_x = np.arange(-self.near_field.shape[0]/2, self.near_field.shape[0]/2) * delta_fx
            k_y = np.arange(-self.near_field.shape[1]/2, self.near_field.shape[1]/2) * delta_fy

            theta_k_x = np.arctan(k_x/k)
            theta_k_y = np.arctan(k_y/k)

            FF_x = distance * np.tan(theta_k_x) * 1e6
            FF_y = distance * np.tan(theta_k_y) * 1e6

            # Create the frequency meshgrid
            self.FF_x, self.FF_y = np.meshgrid(FF_x, FF_y)

            print("2D Fourier Transform calculated successfully!")
        except (RuntimeError, TypeError, NameError):
            print("Error while calculating the 2D-Fourier Transform")

        return self.far_field, self.FF_x, self.FF_y

    def crop_fields(self, NF_crop=1400, FF_crop=1000):
        # Crop the images to the center position
        # Near Field
        self.near_field_crop = self.crop_center(self.near_field, NF_crop, NF_crop)
        self.X_crop = self.crop_center(self.X, NF_crop, NF_crop)
        self.Y_crop = self.crop_center(self.Y, NF_crop, NF_crop)

        # Far-Field
        self.far_field_crop = self.crop_center(self.far_field, FF_crop, FF_crop)
        self.freq_X_crop = self.crop_center(self.FF_x, FF_crop, FF_crop)
        self.freq_Y_crop = self.crop_center(self.FF_y, FF_crop, FF_crop)

    def plot_fields(self, save_file):
        """
        Plots the near-field and far-fields after a 2D Fourier Transform is performed.

        Args:
            save_file (string): file path and name where to ve the plotted fields.

        Returns:
            int: 0 for a successfuly plot and save
        """
        # Plot and save the Intensity of the NF and FF
        fig, axes1 = plt.subplots(1, 2, figsize=(10, 5))
        axes = iter(np.ravel(axes1))
        ax = next(axes)
        ax.pcolormesh(self.X_crop*100, self.Y_crop*100, self.near_field_crop, cmap='magma')
        ax.set_title('Near Field')
        ax.set_xlabel('cm')
        ax.set_ylabel('cm')
        ax.set_xlim(-7, 7)
        ax.set_ylim(-7, 7)

        ax = next(axes)
        ax.pcolormesh(self.freq_X_crop, self.freq_Y_crop, self.far_field_crop, cmap='jet', vmax=1)
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        ax.set_title('Far Field')
        ax.set_xlabel('\u03BCm')
        ax.set_ylabel('\u03BCm')

        plt.tight_layout()
        fig.savefig(save_file, dpi=450, facecolor='white', format='png')
        plt.show()

        return 0

    def ff_2dgaussian_fit(self):
        """
        Do a 2D Gaussian fit on the far-field distribution

        Returns:
            ndarray: _description_
        """
        try:
            print("")
            print("2D Gaussian fit in the FF distribution starting....")
            self. popt_fit, self.pcov_fit = gf.fit2d(XX=self.freq_X_crop,
                                                     YY=self.freq_Y_crop,
                                                     ZZ=self.far_field_crop,
                                                     initial_guess=(1, 0, 0, 10, 10, 0, 0),
                                                     output=True)
            print("2D Gaussian fit completed!")
        except (RuntimeError, TypeError, NameError):
            print("Error in the 2D Gaussian fit.")

        return self.popt_fit, self.pcov_fit

    def getQfactor(self):
        """
        Calculates the q-factor of the far-field after a Fourier Transform is performed.
        """
        idx_q_factor = self.far_field_crop > 0.5

        counts_within_FWHM = np.sum(self.far_field_crop[idx_q_factor])
        total_counts = np.sum(self.far_field_crop)

        self.q_factor = counts_within_FWHM/total_counts * 100

        print("")
        print("q-factor = %.1f %%" % self.q_factor)

    def plot_fields_fit(self, save_file):

        fig, axes1 = plt.subplots(1, 2, figsize=(10, 5))
        axes = iter(np.ravel(axes1))
        ax = next(axes)
        ax.pcolormesh(self.X_crop*100, self.Y_crop*100, self.near_field_crop, cmap='magma')
        ax.set_title('Near Field')
        ax.set_xlabel('cm')
        ax.set_ylabel('cm')
        ax.set_xlim(-9, 9)
        ax.set_ylim(-9, 9)

        ax = next(axes)
        ax.pcolormesh(self.freq_X_crop, self.freq_Y_crop,
                      self.far_field_crop, cmap='coolwarm', vmax=1)

        countour_FWHM = gf.twoD_Gaussian((self.popt_fit[3]*2.35/2,
                                          self.popt_fit[4]*2.35/2),
                                         *self.popt_fit)

        data_fitted = gf.twoD_Gaussian((self.freq_X_crop, self.freq_Y_crop),
                                       *self.popt_fit)

        ax.contour(self.freq_X_crop, self.freq_Y_crop,
                   data_fitted.reshape(self.far_field_crop.shape[0], self.far_field_crop.shape[1]),
                   levels=[0.5], colors=['black'])

        text_legend = (r"$\sigma_x$ = %.1f \u03BCm\n" +
                       r"$2\sigma_x$ = %.1f \u03BCm\n" +
                       r"FWHM$_x$ = %.1f \u03BCm\n") % (self.popt_fit[3],
                                                        2*self.popt_fit[3],
                                                        2.35*self.popt_fit[3])
        _ = ax.text(0.1, 0.12, text_legend, horizontalalignment='left', color='white',
                    fontsize=10, weight='bold', verticalalignment='center',
                    transform=ax.transAxes)

        text_legend2 = (r"$\sigma_y$ = %.1f \u03BCm\n" +
                        r"$2\sigma_y$ = %.1f \u03BCm\n" +
                        r"FWHM$_y$ = %.1f \u03BCm\n" +
                        r"q-factor = %.1f %%") % (self.popt_fit[4],
                                                  2*self.popt_fit[4],
                                                  2.35*self.popt_fit[4],
                                                  self.q_factor)
        _ = ax.text(0.625, 0.12, text_legend2, horizontalalignment='left', color='white',
                    fontsize=10, weight='bold', verticalalignment='center',
                    transform=ax.transAxes)

        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        ax.set_title('Far Field')
        ax.set_xlabel('\u03BCm')
        ax.set_ylabel('\u03BCm')

        plt.tight_layout()
        fig.savefig(save_file, dpi=450, facecolor='white', format='png')
        plt.show()


class farfield_theory:
    """
    Class to calculate the theoretical far-field parameters (w0, FWHM, Rayleigh length)
    from experimental parameters using Gaussian beam optics.
    A sequence of commands in the correct order is required to fully compute the
    theoretical far-field parameters of a beam.
    Please see the scrip test_farfield_theory.py in the tests folder to have an example
    of how to work with this class.

    Refs.:
    *
    * https://www.newport.com/n/gaussian-beam-optics
    """

    def __init__(self, beam_diameter=12.5, focal_length=2.5, wavelength=800e-9):
        """_summary_

        Args:
            beam_diameter (float, optional): Collimated beam diameter in cm. Defaults to 12.5.
            focal_length (float, optional): Focal length of the focus optics
            (lens, parabolic mirror, spherical mirror, etc) given in meters. Defaults to 2.5.
            wavelength (_type_, optional): Wavelength of the light beam in meters.
            Defaults to 800e-9.
        """

        self.beam_diameter = beam_diameter * 1e-2  # Converts the beam diameter in meters
        self.focal_length = focal_length
        self.wavelength = wavelength

    # Getter and setters for the three required parameters.
    # Everytime a get property is called, the compute focus method is executed

    def set_beam_diameter(self, diameter):
        """
        Sets the beam diameter and compute the theoretical focus spot paramters.

        Args:
            beamdia (float): Collimated beam diameter in cm.
        """
        # print('setname() called')
        self.beam_diameter = diameter * 1e-2
        self.compute_focus_parameters()

    diameter = property(fset=set_beam_diameter)

    def set_focal_length(self, f_length):
        """
        Sets the focal length of the focus optics
        (lens, parabolic mirror, spherical mirror, etc) given in meters.

        Args:
            f_length (float): focal length in m.
        """
        # print('setname() called')
        self.focal_length = f_length
        self.compute_focus_parameters()

    f_length = property(fset=set_focal_length)

    def set_wavelength(self, wavelength_light):
        """
        Sets the wavelength of the light beam in meters.

        Args:
            wavelength_light (float): wavelength of the light beam in nm.
        """
        # print('setname() called')
        self.wavelength = wavelength_light
        self.compute_focus_parameters()

    wavelength_light = property(fset=set_wavelength)

    # Compute method
    def compute_focus_parameters(self):

        try:
            self.f_number = self.focal_length / self.beam_diameter
            print("F-number = %.1f" % self.f_number)

            self.w0 = 2 * self.f_number * self.wavelength / np.pi
            print("w0 = %.2f um" % (self.w0*1e6))

            self.rayleigh_length = np.pi * self.w0**2 / self.wavelength
            print("Rayleigh length Zr = %.2f um" % (self.rayleigh_length * 1e6))

            self.FWHM = 1.17 * self.w0
            print("Focus spot FWHM = %.2f um" % (self.FWHM * 1e6))

            return self.f_number, self.w0, self.rayleigh_length, self.FWHM

        except (RuntimeError, TypeError, NameError):
            print("Error while calculating the parameters!")
            return -1
