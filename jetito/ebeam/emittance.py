import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
import cv2

from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from peakutils import baseline
import skimage.filters as filters
from skimage.morphology import disk
from scipy.signal import savgol_filter


# Jetito imports
# Gaussian fit

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


class pp_emittance_calculator:

    def __init__(self, filename, image_calib=18.5e-3, d_target_screen=1.45, d_pp_screen=1.269):
        """
        Contructor of the pointing_analysis class.

        Args:
            filename (string): Path and file name of the pepper-pot image to be analyzed.
            image_calib (double, optional): Calibration of the near-field image in units
            of length/pixel (typical: mm/pixels). Defaults to 18.5e-3.
            d_target_screen (double, optional): Distance from the target to the screen
            where the pointing image was recorded (typical: meters). Defaults to 1.45.
            d_pp_screen (double, optional): Distance from the pepper-pot to the screen.
            (typical: meters). Defaults to 1.296.
        """

        self.file = filename
        self.image_calib = image_calib * 1e3
        self.dist_target_screen = d_target_screen
        self.dist_pp_screen = d_pp_screen

        try:
            self.image_pp = cv2.imread(self.file, -1)
            print("Image file loaded successfully!")
            print(self.image_pp.shape[0])

        except (RuntimeError, TypeError, NameError):
            print("Error loading the image file")

    def crop_image(self, left=0, right=1000, top=850, bottom=1965, save_file=None,
                   axis='xaxis', **kwargs):
        """Crops the original pointing image to an smaller part for later to perform the
        2D-Gaussian analyis of the ebeam spot.

        Args:
            left (int, optional): Pixel position for starting cropping the image from the left.
            Defaults to 0.
            right (int, optional): Pixel position for starting cropping the image from the right.
            Defaults to 1000.
            top (int, optional): Pixel position for starting cropping the image from the top.
            Defaults to 850.
            bottom (int, optional): Pixel position for starting cropping the image from the
            bottom. Defaults to 1965.
            save_file (string, optional): Path of the file to save the cropped focus.
            Please provide this path. Defaults to None.
            axis (str, optional): Axis to perform the integration of the pepper pot signal. Defaults to 'xaxis'.
        """

        if 'verbose' not in kwargs:
            verbose = False
        else:
            verbose = kwargs['verbose']
            kwargs.pop('verbose', None)

        if verbose:
            print("")
            print("Crop the image")

        self.image_crop = self.image_pp[top:bottom, left:right]

        # Integrates the signal in the axis direction set as parameter of the function
        if not type(axis) is str:
            raise TypeError("The axis parameter is only allowed to be string!")
        else:
            if axis == 'xaxis':
                self.image_crop = self.image_crop
            elif axis == 'yaxis':
                self.image_crop = cv2.rotate(self.image_crop, cv2.ROTATE_90_CLOCKWISE)
            else:
                # print("Integration axis not defined in the parameters.")
                raise Exception("Please choose correctly an integration axis.")

        self.xaxis = np.arange(self.image_crop.shape[1] - 2)
        self.yaxis = np.arange(self.image_crop.shape[0] - 2)

        if verbose:
            print("Shape of the cropped image: %d x %d" % (self.image_crop.shape[0],
                                                           self.image_crop.shape[1]))

        if save_file is not None:
            cv2.imwrite(save_file, self.image_crop)
            if verbose:
                print("Saved cropped image at: " + save_file)

    def process(self, distance=40, height=300, del_peaks=None, **kwargs):
        """
        Method for pre-processing the pepper pot signal. It process the signal in the following sequence:
        * Median filter with disk_r in the cropped image.
        * Integrates the signal in the axis direction set in the axis parameter of the method.
        * SavGol filter on the integrated with window defined in the savgol_deg parameter of the function,
        and polyorder fixed to 1.
        * Baseline calculation of the signal with degree given by baseline_deg parameter.
        * Removes the baseline from the integrated and filtered signal.
        * Find the peaks of the PP signal
        * Truncates the baseline removed signal to zero

        Args:
            distance (int, optional): Distance between peaks of the pepper pot signal. Defaults to 40.
            height (int, optional): Height threshold for finding the peaks of the pepper pot signal. Defaults to 300.
            del_peaks (tuple, optional): Tuple of indexes of the peaks to delete from the pepper pot signal.
            Defaults to None.

        Raises:
            TypeError: Raise in case the axis parameter of the function is not a string.
            Exception: Raised in case the wrong axis is chosen for the signal integration.
        """

        # Manage the input parameters of the method
        if 'verbose' not in kwargs:
            verbose = False
        else:
            verbose = kwargs['verbose']
            kwargs.pop('verbose', None)

        if 'disk_r' not in kwargs:
            median_disk_r = -1
        else:
            median_disk_r = kwargs['disk_r']
            kwargs.pop('disk_r', None)

        if 'baseline_deg' not in kwargs:
            deg = -1
        else:
            deg = kwargs['baseline_deg']
            kwargs.pop('baseline_deg', None)

        if 'savgol_deg' not in kwargs:
            savgol_deg = -1
        else:
            savgol_deg = kwargs['savgol_deg']
            kwargs.pop('savgol_deg', None)

        if verbose:
            print("")
            print("Starting signal pre-processing (integration, filtering)...")

        # Apply the filtering accoring to the chosen parameters
        if median_disk_r != -1:
            self.filtered = filters.rank.median(self.image_crop, disk(median_disk_r))
        else:
            self.filtered = self.image_crop

        # Integrate the signal
        self.xintegrated = np.sum(self.filtered[:, 1:-1], axis=0)

        # Apply the SavGol filter
        if savgol_deg != -1:
            self.xintegrated = savgol_filter(self.xintegrated, savgol_deg, 1)
        else:
            self.xintegrated = self.xintegrated

        # Calculate the baseline of the signal for removal
        if deg != -1:
            self.xbaseline = baseline(self.xintegrated, deg=deg)
        else:
            self.xbaseline = 0

        # Calculate the integrate signal for PP evaluation with baseline removed
        self.xbasereduced = self.xintegrated - self.xbaseline

        # Find the peaks of the PP signal
        self.xpeaks, self.xroi = self.peakfinder(self.xbasereduced,
                                                 distance=distance,
                                                 height=height,
                                                 del_peaks=del_peaks)

        # Truncate the baseline removal to zero
        self.xbasereduced[self.xbasereduced < 0] = 0

        if verbose:
            print("Signal was successfully pre-processed!")

    def peakfinder(self, signal, distance=40, height=300, del_peaks=None):

        peaks = find_peaks(signal, distance=distance, height=height)[0]
        peaks_diff = np.diff(peaks)
        for i in range(5):
            if (peaks[i] - peaks_diff[i] < 0):
                peaks = np.delete(peaks, i)

        if del_peaks is not None:
            peaks = np.delete(peaks, del_peaks)

        roi = np.int_(np.diff(peaks)/2) + peaks[:-1]
        roi = np.append(roi, peaks[-1] + np.diff(peaks)[-1]/2)
        roi = np.insert(roi, 0, int(2 * peaks[0] - roi[0]))
        roi = np.int_(roi)
        return peaks, roi

    def compute(self, **kwargs):

        if 'verbose' not in kwargs:
            verbose = False
        else:
            verbose = kwargs['verbose']
            kwargs.pop('verbose', None)

        if verbose:
            print("")
            print("Starting calculating the rms emittance...")

        self.xmu = np.zeros_like(self.xpeaks)
        for i in range(len(self.xpeaks)):
            self.xmu[i] = np.average(self.xaxis[int(self.xroi[i]):int(self.xroi[i+1])],
                                     weights=self.xbasereduced[int(self.xroi[i]):int(self.xroi[i+1])])

        self.xpopt = np.zeros((len(self.xpeaks), 2))

        for i in range(len(self.xpeaks)):
            def gausx(x, a, sigma):
                return a * np.exp(-(x - self.xmu[i])**2 / (2 * sigma**2))

            self.xpopt[i], yarrah = curve_fit(gausx, self.xaxis[self.xroi[i]:self.xroi[i+1]],
                                              self.xbasereduced[self.xroi[i]:self.xroi[i+1]])

        # Calculation of the emittance terms starts here
        # FIRST TERM
        self.x_N = np.sum(self.xbasereduced[self.xroi[0]:self.xroi[-1]])

        self.xs_j = np.arange(0, 120*len(self.xpeaks), 120)
        self.xn_j = np.zeros(len(self.xpeaks))

        for i in range(len(self.xpeaks)):
            self.xn_j[i] = np.sum(self.xbasereduced[int(self.xroi[i]):int(self.xroi[i+1])])

        self.xaxis_scaled = self.xaxis-self.xmu[0]
        self.xaxis_scaled = self.xaxis_scaled*120 / np.mean(np.diff(self.xmu))

        self.xs_mean = np.average(self.xaxis_scaled[self.xroi[0]:self.xroi[-1]],
                                  weights=self.xbasereduced[self.xroi[0]:self.xroi[-1]])

        self.xfirst = np.sum(self.xn_j * (self.xs_j - np.ones(len(self.xpeaks)) * self.xs_mean)**2)

        # SECOND TERM
        L = self.dist_pp_screen * 1e6  # Conversion to um
        self.xaxis_scaled_screen = (self.xaxis-self.xmu[0]) * self.image_calib

        self.sigma_xj = (np.abs(self.xpopt[:, 1]) * self.image_calib) / L
        self.x_mean_screen = self.xaxis_scaled[self.xmu]
        self.xj_dash_mean = (self.x_mean_screen-self.xs_j) / L
        self.x_dash_mean = np.sum(self.xn_j * self.xj_dash_mean) / self.x_N
        self.xsecond = np.sum(self.xn_j * self.sigma_xj**2 + self.xn_j * (self.xj_dash_mean-self.x_dash_mean)**2)

        # Third TERM
        self.xthird = (np.sum(self.xn_j * self.xs_j * self.xj_dash_mean) /
                       - self.x_N * self.xs_mean * self.x_dash_mean)**2

        # RMS emittance
        self.rms_emittance = np.sqrt((self.xfirst*self.xsecond-self.xthird) / self.x_N**2)
        #
        self.norm_emittance = np.sqrt(143**2 * (0.27 * self.xfirst*self.xsecond + self.xfirst*self.xsecond /
                                                - self.xthird) / self.x_N**2)

        if verbose:
            print("The calculated rms emittance = %.6f mm mrad" % (self.rms_emittance))
            print("The calculated normalized emittance = %.6f mm mrad" % (self.norm_emittance))
            print("first term = %.6f" % (0.27 * self.xfirst*self.xsecond/self.x_N**2))

        return self.rms_emittance

    def plot_analysis(self, save_file=None, **kwargs):
        """
        Method to plot the pepper pot analysis result. Two subplots are generated:
        * First plot: integrated signal and baseline
        * Second plot: baseline removed integrated signal and the Gaussian plots

        Args:
            save_file (string, optional): Path and file name, if image is supposed to be saved.
            Defaults to None.
        """

        fig, axes1 = plt.subplots(2, 1, figsize=(10, 10))
        axes = iter(np.ravel(axes1))

        # First plot of the integrated signal and baseline
        ax = next(axes)

        ax.plot(self.xintegrated, label="Integrated")
        ax.plot(self.xbaseline, label="Baseline")
        ax.legend(loc='upper left')
        # ax.set_xlabel('pixels')
        # ax.set_ylabel('Counts')

        # Second plot of the baseline removed integrated signal and the Gaussian plots
        ax = next(axes)
        ax.plot(self.xbasereduced, label="Baseline Subtracted")
        ax.scatter(self.xaxis[self.xpeaks], self.xbasereduced[self.xpeaks], c="r")
        ax.vlines(self.xroi, 0, 6, colors="k")
        ax.scatter(self.xaxis[np.int_(self.xmu)], self.xbasereduced[np.int_(self.xmu)], marker="X", c="k")

        for i in range(len(self.xpeaks)):
            def gausx(x, a, sigma):
                return a * np.exp(-(x - self.xmu[i])**2 / (2 * sigma**2))

            ax.plot(self.xaxis[self.xroi[i]:self.xroi[i+1]], gausx(self.xaxis[self.xroi[i]:self.xroi[i+1]],
                                                                   *self.xpopt[i]))
            ax.legend()

        # show the emittance in the plot
        ax.text(0.1, 0.8, "r.m.s. emittance\n%.5f mm mrad" % (self.rms_emittance), fontsize=12,
                horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

        if save_file is not None:
            fig.savefig(save_file, dpi=450, facecolor='white', format='png', bbox_inches='tight')

        plt.show()
