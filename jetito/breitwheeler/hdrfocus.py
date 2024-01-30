"""
Class for analysing HDR focus images from CALA.
Code implemented by the CALA team/students and included in the jetito package
by F. C. Salgado for the analysis of the HDR focus images obtained during the
Breit-Wheeler pair production campaing.

The code suffered some modifications for compliance witht he jetito package,
but it's functionalities are kept the same as the original code.

Special thanks to K. Grafenstein (CALA, LMU) for providing the code.

Reviews:
* Created on Mon Nov 29 2021 by CALA team/students
* Edited on Tue 25 Jan 2024 by F. C. Salgado
"""
import os
import cv2
import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import scipy.constants as const
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, ScalarFormatter)
from matplotlib.colors import LogNorm
import pandas as pd
import scipy.ndimage as scipy_ndimage
import matplotlib.pyplot as plt
from skimage import measure
from pathlib import Path
from skimage.feature import match_template
from skimage.feature import match_template

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

#%%
class HDR():
    def __init__(self, datafolder, outputfolder, filter = [0.496, 0.2135, 0.1285, 0.1065]):
        """Constructor of the HDR focus images analysis from the CALA
        facility.

        Args:
            datafolder (string): folder where the HDR focus data is stored.
            outputfolder (string): folder where the results of the data analysis
            is saved.
            filter (float, list): list of floats witht the filter configuration in which the
            images were taken.
        """
        self.datafolder = datafolder
        self.outputfolder = outputfolder
        self.Filter = filter

    def returnSaturatedArea(self, curImg):
        return (curImg >= (2**16*0.8)).astype(int)

    def returnNonSaturatedArea(self, curImg):
        return (curImg < (2**16*0.8)).astype(int)

    def hdr_reconstruction(self, number_of_images_per_exposure_setting, sets, sets_BG, **kwargs):
        """HDR Image reconstruction by analysing backgorund shots and focus images of different
        exposure settings. A cross correlation between the focus images is performed.

        Note that the analysis may take a while to complete.

        Args:
            number_of_images_per_exposure_setting (int): number of images per exposure setting.
            sets (list, string): list of sets of focus images for the different exposure settings. Used
            to parse the image file name in the post processing.
            sets_BG (list, string): list of sets of background focus images for the different exposure settings. Used
            to parse the image file name in the post processing.

        Raises:
            ValueError: Number_of_images_per_exposure_setting argument is not an int
            ValueError: Sets argument is not a list of strings
            ValueError: Sets_BG argument is not a list of strings
            ValueError: Data folder path is incorrect!
        """
        # self.Focus_Cam = Focus_Cam(self.basepath, self.camera_id) #Initiate the Camera
        # self.laserproxy = DeviceProxy(self.laserproxy_path) #Init the laserproxy

        if 'verbose' not in kwargs:
            verbose = False
        else:
            verbose = kwargs['verbose']
            kwargs.pop('verbose', None)

        #self.number_of_images_per_exposure_setting = 11
        #self.sets = ['_2_31_F1111_focus','_4_31_F1110_focus', '_6_31_F1011_focus', '_8_31_F1001_focus', '_10_31_F1000_focus', '_12_31_F0001_focus', '_14_31_F0000_focus']
        #self.sets_BG = ['_1_31_D_focus_', '_3_31_D_focus_', '_5_31_D_focus_', '_7_31_D_focus_', '_9_31_D_focus_', '_11_31_D_focus_', '_13_31_D_focus_', '_15_31_D_focus_']
        if not isinstance(number_of_images_per_exposure_setting, int):
            raise ValueError("Number_of_images_per_exposure_setting argument is not an int")

        if not isinstance(sets, list):
            raise ValueError("Sets argument is not a list of strings")

        if not isinstance(sets_BG, list):
            raise ValueError("Sets_BG argument is not a list of strings")

        self.number_of_images_per_exposure_setting = number_of_images_per_exposure_setting
        self.sets = sets
        self.sets_BG = sets_BG

        # Set variables for the data analysis
        self.img_list = []
        self.img_BG = []
        self.img_list_blurred = []
        self.preliminary_minval_maxval_minloc_maxloc = []
        self.minval_maxval_minloc_maxloc = []
        self.img_cropped = []
        self.blur_parameters = [20,20,30,35,45,65,120]
        self.img_BG_cropped = []
        self.hdr_image = []
        self.translation = []
        self.img_convoluted = []
        self.self_convoluted_images = []
        self.rolled_images = []
        self.img_BG_averaged = None
        self.im2_register = []
        self.img_BG_cropped_final = []
        self.match_templates = []
        self.Filter_Setting = np.array([1,
               self.Filter[0],
               self.Filter[3],
               self.Filter[0]*self.Filter[3],
               self.Filter[0]*self.Filter[1]*self.Filter[3],
               self.Filter[1]*self.Filter[2]*self.Filter[3],
               self.Filter[0]*self.Filter[1]*self.Filter[2]*self.Filter[3]
               ], dtype=np.float32)
        self.threshold = 0.8
        self.loc = []
        self.final_crop = []

        try:
            self.threshold_contour = 0.99
            # print(argin)

            # Check if the given path is correct
            if os.path.exists(self.datafolder):
                if verbose:
                    print("Looking at images at: " + str(self.datafolder))

                # Create paths for the data results
                Path(os.path.join(self.outputfolder,"analysis")).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(self.outputfolder,"debug")).mkdir(parents=True, exist_ok=True)
            else:
                raise ValueError("Data folder path is incorrect!")

            # Load images into a dictionary and substract Background
            self.img_dict = {} # dict with exposure setting as key
            self.img_list = []
            self.img_list_BG = []

            for i in range(len(self.sets)):

                if verbose:
                    print("Datasets being analyzed:")
                    print("Data:" + self.sets[i])
                    print("Background:" + self.sets_BG[i])
                    print("Background" + self.sets_BG[i+1])

                for self.filename in sorted(os.listdir(self.datafolder)):

                    # build the image path
                    image_path = os.path.join(self.datafolder, self.filename)

                    if self.sets[i] in self.filename:
                        self.img = np.array(cv2.imread(image_path, cv2.IMREAD_ANYDEPTH))
                        self.img_list.append(self.img)

                        if verbose:
                            print("Loaded image: " + image_path)

                    if self.sets_BG[i] in self.filename:
                        self.img_BG_1 = np.array(cv2.imread(image_path, cv2.IMREAD_ANYDEPTH))
                        self.img_list_BG.append(self.img_BG_1)

                        if verbose:
                            print("Loaded image: " + image_path)

                    if self.sets_BG[i+1] in self.filename:
                        self.img_BG_2 = np.array(cv2.imread(image_path, cv2.IMREAD_ANYDEPTH))
                        self.img_list_BG.append(self.img_BG_2)

                        if verbose:
                            print("Loaded image: " + image_path)

                self.img_BG_averaged = np.mean(self.img_list_BG)
                self.img_list = self.img_list - self.img_BG_averaged
                self.img_list[self.img_list < 0] = 0

                self.img_dict[i] = self.img_list
                self.img_list = []
                self.img_list_BG = []

            if verbose:
                print("")
                print("Starting image processing.......")
            # Images cropping
            if verbose:
                print("")
                print("1. Create empty variables and dictionaries for processing")
            # Define geometric center of image
            self.goalx = int(self.img_dict[0][0].shape[0]/2)
            self.goaly = int(self.img_dict[0][0].shape[1]/2)

            self.img_cropped_dict = {} # define dict for averaged images
            self.img_cropped_list = []

            self.match_templates_dict = {} # define list for the templates of the cross correlation later (first image for each expsure which is cropped more)

            # predefine dictionary with empty lists for the cropped images (makes the program faster)
            for i in range(len(self.sets)):
                for j in range(self.number_of_images_per_exposure_setting):
                    self.img_cropped_list.append([])
                self.img_cropped_dict[i] = self.img_cropped_list
                self.img_cropped_list = []
                self.match_templates_dict[i] = []
                self.maxlocsy = []
                self.maxlocsx = []

            if verbose:
                print("\t> Created empty dictionary for the cropped images and template images")

            self.crop_range = 458

            # Crop images
            if verbose:
                print("")
                print("2. Crop images roughly around center and asign the templates for cross correlation")

            for i in range(len(self.sets)):
                for j in range(self.number_of_images_per_exposure_setting):

                    # build image cropped folder
                    img_cropped_path = os.path.join(self.outputfolder, "debug", "image_cropped_" + str(i) + "_" + str(j) + ".png")
                    # build csv coordinate path
                    coordinates_path = os.path.join(self.outputfolder, "analysis", "center_of_focus.csv")

                    if i<3:
                        # blur images to get accurate maximum position
                        self.img_blurred = scipy_ndimage.median_filter(self.img_dict[i][j],
                                                                       self.blur_parameters[i])

                        self.preliminary_minval_maxval_minloc_maxloc = cv2.minMaxLoc(self.img_blurred)

                        # Crop images around center
                        #/self.number_of_images_per_exposure_setting
                        self.img_cropped_dict[i][j] = self.img_dict[i][j][
                            int(self.preliminary_minval_maxval_minloc_maxloc[3][1])-self.crop_range:
                            int(self.preliminary_minval_maxval_minloc_maxloc[3][1])+self.crop_range,
                            int(self.preliminary_minval_maxval_minloc_maxloc[3][0])-self.crop_range:
                            int(self.preliminary_minval_maxval_minloc_maxloc[3][0])+self.crop_range]

                        if verbose:
                            print("\t> Cropped image: " + img_cropped_path + " around center: " +
                                  str(self.preliminary_minval_maxval_minloc_maxloc[3][1]),
                                  str(self.preliminary_minval_maxval_minloc_maxloc[3][0]))

                        cv2.imwrite(img_cropped_path,
                                    self.img_cropped_dict[i][j].astype(np.uint16))

                        # save coordinates in csv
                        pd.DataFrame([[str(i), str(j), str(self.preliminary_minval_maxval_minloc_maxloc[3][1]),
                                       str(self.preliminary_minval_maxval_minloc_maxloc[3][0])]]).to_csv(coordinates_path,
                                                                                                         header=False,
                                                                                                         index=False,
                                                                                                         sep='\t',
                                                                                                         mode="a")

                        self.maxlocsy.append(self.preliminary_minval_maxval_minloc_maxloc[3][1])
                        self.maxlocsx.append(self.preliminary_minval_maxval_minloc_maxloc[3][0])

                        self.maxlocsy_mean = np.mean(self.maxlocsy)
                        self.maxlocsx_mean = np.mean(self.maxlocsx)
                    else:
                        if verbose:
                            print('\tMaxlocsMean:' + str(self.maxlocsy_mean), str(self.maxlocsx_mean))

                        self.img_cropped_dict[i][j] = self.img_dict[i][j][
                            int(self.maxlocsy_mean) - self.crop_range:int(self.maxlocsy_mean) + self.crop_range,
                            int(self.maxlocsx_mean) - self.crop_range:int(self.maxlocsx_mean) + self.crop_range]
                        # self.number_of_images_per_exposure_setting)
                        if verbose:
                            print("\t> Cropped image: " + img_cropped_path + " around center: " +
                                  str(self.maxlocsy_mean),
                                  str(self.maxlocsx_mean))

                        cv2.imwrite(img_cropped_path,
                                    self.img_cropped_dict[i][j].astype(np.uint16))

                        # save coordinates in csv
                        pd.DataFrame([[str(i), str(j), str(self.maxlocsy_mean),
                                        str(self.maxlocsx_mean)]]).to_csv(coordinates_path,
                                                                          header=False,
                                                                          index=False,
                                                                          sep='\t',
                                                                          mode="a")
            if verbose:
                print("\t> All immages were cropped around center")

            # Find the starting coordinates for cross-correlation
            if verbose:
                print("")
                print("3. Find starting coordinates for the cross correlation")

            self.start_coordinates_cc = []
            if verbose:
                print("\t> Start coordinates for cross-correlations:")
            for i in range(len(self.sets)):
                self.contours = measure.find_contours(self.img_cropped_dict[i][0],
                                                      np.max(self.img_cropped_dict[i][0]) * self.threshold_contour)
                self.length_of_contours = []

                for self.contour in self.contours:
                    self.length_of_contours.append(len(self.contour))

                self.index = self.length_of_contours.index(max(self.length_of_contours))
                self.start_coordinates_cc.append(
                    [int(min(self.contours[self.index][:, 0])+(max(self.contours[self.index][:, 0])-min(self.contours[self.index][:, 0]))/2),
                     int(min(self.contours[self.index][:, 1])+(max(self.contours[self.index][:, 1])-min(self.contours[self.index][:, 1]))/2)])

                if verbose:
                    print(self.start_coordinates_cc)

            print("\t> End of defining the starting coordinates for cross-correlations")

            # Cross-correlations
            # For each exposure setting: Do cross correlation and overlap images
            if verbose:
                print("")
                print("4. Cross-correlation of images with same exposure setting")

            self.final_crop_dict = {}
            self.final_crop = []

            if verbose:
                print("\t> Deviations from the center:")
            for i in range(len(self.sets)):
                for j in range(self.number_of_images_per_exposure_setting):
                    self.result = match_template(scipy_ndimage.median_filter(self.img_cropped_dict[i][j],
                                                                             self.blur_parameters[i]),
                                                 scipy_ndimage.median_filter(self.img_cropped_dict[i][0],
                                                                             self.blur_parameters[i]),
                                                 cv2.TM_CCORR_NORMED)  # TM_CCORR_NORMED,TM_CCOEFF_NORMED

                    self.ij = np.unravel_index(np.argmax(self.result),
                                               self.result.shape)

                    self.x, self.y = self.ij[::-1]

                    self.deviation_from_optimum = (self.start_coordinates_cc[i][0]-self.y,
                                                   self.start_coordinates_cc[i][1]-self.x)

                    if verbose:
                        print(self.deviation_from_optimum)

                    # Crop image around center
                    self.corrected_image = np.roll(self.img_cropped_dict[i][j],
                                                   self.deviation_from_optimum,
                                                   (0,1))

                    self.final_crop.append(self.corrected_image / self.number_of_images_per_exposure_setting)

                self.final_crop_dict[i] = self.final_crop
                self.final_crop = []

            # save images in debug folder
            for i in range(len(self.sets)):
                for j in range(self.number_of_images_per_exposure_setting):
                    img_path_save = os.path.join(self.outputfolder, "debug", "final_image_cropped_" +
                                                 str(i) + "_" + str(j) + ".png")
                    cv2.imwrite(img_path_save,
                                (self.final_crop_dict[i][j] * self.number_of_images_per_exposure_setting).astype(np.uint16))
            if verbose:
                print("\t> End of the cross correlation of images with same exposure setting")

            # Average of images with same exposure setting
            # For each exposure setting: Average over the 10 images"""
            if verbose:
                print("")
                print("5. Get the average of 10 images of each exposure setting")

            self.averaged_image = []
            for i in range(len(self.sets)):
                img_path_save = os.path.join(self.outputfolder, "debug", "averaged_image_" +
                                             str(i) + ".png")

                # Average images
                self.averaged_image.append(sum(self.final_crop_dict[i]))

                if verbose:
                    print("\t> Averaged image of exposure setting " + str(i + 1))

                cv2.imwrite(img_path_save,
                            self.averaged_image[i].astype(np.uint16))

            if verbose:
                print("\t> End of image averaging")


            # Cross correlate the averaged images
            if verbose:
                print("")
                print("6. Cross correlate the averaged images")

            self.final_averaged_cropped_images = []
            self.final_averaged_cropped_images.append(self.averaged_image[0])

            # find the start coordinates
            if verbose:
                print("\t> Find the start coordinates")

            self.contours = measure.find_contours(self.averaged_image[0],
                                                  np.max(self.averaged_image[0]) * self.threshold_contour)
            self.length_of_contours = []
            for contour in self.contours:
                self.length_of_contours.append(len(contour))

            self.index = self.length_of_contours.index(max(self.length_of_contours))
            self.start_coordinates = [int(min(self.contours[self.index][:, 0]) + (max(self.contours[self.index][:, 0]) - min(self.contours[self.index][:, 0]))/2),
                                      int(min(self.contours[self.index][:, 1]) + (max(self.contours[self.index][:, 1]) - min(self.contours[self.index][:, 1]))/2)]

            # start the cross correlation
            if verbose:
                print("\t> Start the cross correlation")
                print("\t> Deviations from the center:")

            for i in range(len(self.sets)-1):
                self.result = match_template(self.averaged_image[i+1],
                                             self.averaged_image[i],
                                             cv2.TM_CCORR_NORMED)  #TM_CCORR_NORMED,TM_CCOEFF_NORMED

                self.ij = np.unravel_index(np.argmax(self.result),
                                           self.result.shape)

                self.x, self.y = self.ij[::-1] #peaks in the output of match_template correspond to the origin (i.e. top-left corner) of the template
                self.deviation_from_optimum = (self.start_coordinates[0] - self.y,
                                               self.start_coordinates[1] - self.x)

                if verbose:
                    print(self.deviation_from_optimum)
                self.corrected_image = np.roll(self.averaged_image[i + 1],
                                               self.deviation_from_optimum,
                                               (0, 1))
                self.final_averaged_cropped_images.append(self.corrected_image)
                self.averaged_image[i+1] = self.corrected_image


            for i in range(len(self.sets)):
                img_avg_path_save = os.path.join(self.outputfolder, "debug", "final_averaged_cropped_image_" +
                                                 str(i) + "_" + ".png")
                cv2.imwrite(img_avg_path_save,
                            self.final_averaged_cropped_images[i].astype(np.uint16))


            # Replace saturated region by data of lesser exposed image
            if verbose:
                print("")
                print("7. Replace saturated region by data of lesser exposed image")

            self.img_cropped_averaged_sorted = []

            for i in range(1, len(self.sets) + 1):
                # resort list
                self.img_cropped_averaged_sorted.append(self.final_averaged_cropped_images[-i])

            self.img_cropped_averaged_sorted = []
            for i in range(1, len(self.sets) + 1):
                self.img_cropped_averaged_sorted.append(self.averaged_image[-i])

            if verbose:
                print("\t> Added final crop into list")

            # for i in range(len(self.sets)):
                # print(np.max(self.img_cropped_averaged_sorted[i]))
            self.hdr_image = self.img_cropped_averaged_sorted[0]
            self.hdr_image_step = [self.img_cropped_averaged_sorted[0]]

            if verbose:
                print("\t> First step of hdr image defined as most exposed image")

            for i in range(len(self.sets)):
                img_saturated_path_save = os.path.join(self.outputfolder, "debug", "staturated_area_image" +
                                                       str(i + 1) + ".png")
                cv2.imwrite(img_saturated_path_save,
                            self.returnSaturatedArea(self.img_cropped_averaged_sorted[i]))



            for i in range(len(self.sets)-1):
                self.hdr_image_step.append(self.hdr_image_step[i]*self.returnNonSaturatedArea(self.img_cropped_averaged_sorted[i])+
                  self.returnSaturatedArea(self.img_cropped_averaged_sorted[i])*self.img_cropped_averaged_sorted[i+1]/self.Filter_Setting[i+1])

                if verbose:
                    print("\t> maximum: ",np.max(self.hdr_image_step[i]))
                    print("\t> minimum: ",np.min(self.hdr_image_step[i]))
                    print("\t> Saturated image of most exposed image replaced by data from next lesser exposed image\n")


            self.hdr_image = self.hdr_image_step[-1]

            output_path_hdr_csv = os.path.join(self.outputfolder, "analysis", "hdr_image.csv")
            np.savetxt(output_path_hdr_csv,
                       self.hdr_image,
                       delimiter = ",")

            for i in range(len(self.final_crop)):
                img_final_crop_path_save = os.path.join(self.outputfolder, "Atlas_finalcrop_" +
                                                        str(i + 1) + ".png")
                cv2.imwrite(img_final_crop_path_save,
                            self.final_crop[i].astype(np.uint16))

                if verbose:
                    print("")
                    print("END: Calculation done successfully!!")

        except Exception as ex:
            self.hdr_image = None
            print(str(ex))

    def load_hdr_image(self, file, **kwargs):

        if not isinstance(file, str):
            raise ValueError("File argument is not a valid string.")

        if 'verbose' not in kwargs:
            verbose = False
        else:
            verbose = kwargs['verbose']
            kwargs.pop('verbose', None)

        try:
            self.hdr_image = np.genfromtxt(file, delimiter=',')
            if verbose:
                print("Reminder: Next step, convert the counts units from the HDR image to W/cm2.")
        except:
            self.hdr_image = None
            raise ValueError("Error in loading the file")

    def convert_counts2physics(self,
                               diameterBeam = 27,
                               focalLengthOap = 1000,
                               E_laser=18,
                               t_laser=30e-15,
                               lambda0=0.8, #micron
                               aperture=0.1,
                               **kwargs):
        """Convert the units from the HDR image, originally in counts, to w/cm^2 (physical units).

        For this conversion, the HDR image should have been calculated via the hdr_reconstruction method
        or loaded via method load_hdr_image.

        Args:
            diameterBeam (int, optional): diameter of the laser beam (centimeters). Defaults to 27.
            focalLengthOap (int, optional): focal length of the focusing optics (spherical mirror or parabolic mirror (centimeters)). Defaults to 1000.
            E_laser (int, optional): energy contained in the laser beam (Joules). Defaults to 18.
            t_laser (_type_, optional): pulse duration of the laser beam (seconds). Defaults to 30e-15.
            lambda0 (float, optional): wavelength of the laser beam (micrometers). Defaults to 0.8.

        Raises:
            ValueError: Please calculate or load the HDR image first.
        """

        if 'verbose' not in kwargs:
            verbose = False
        else:
            verbose = kwargs['verbose']
            kwargs.pop('verbose', None)

        if self.hdr_image is None:
            raise ValueError("Please calculate or load the HDR image first.")

        self.diameter = diameterBeam
        self.focalLengthOap = focalLengthOap
        self.E_laser = E_laser
        self.t_laser = t_laser
        self.lambda0 = lambda0
        self.aperture = aperture

        # Get max of the hdr image
        maxMyCutOut = np.max(self.hdr_image)

        OverThreshold = np.where(self.hdr_image > maxMyCutOut * 0.5, 1, 0)
        anzahlPixel = np.sum(OverThreshold)
        myFiltered = self.hdr_image#scipy_ndimage.median_filter(myCutOut,4)
        # myPic = np.log10(myFiltered)

        sumPx = np.sum(myFiltered)

        FnumberOap = focalLengthOap / diameterBeam

        P0 = 2 * np.sqrt(2 * np.log(2)) * E_laser / np.sqrt(2 * np.pi) / t_laser

        # magnification measurement: diffraction from 'grating' (graph paper) with 10mm slit distance: 
        # measured distance on focus cam between 0th and 1st order diffraction peaks: 727px
        slit_dist = 10e3 # micron
        self.dx = focalLengthOap * np.tan(np.arcsin(lambda0 / slit_dist)) * 1e4 / 727 # pixel size (micron)
        # similar result from measuring 1/e width of 10micron pinhole image on focus cam --> 9 pixel diameter --> 10micron/9=1.1micron); 
        # IDS camera UI 5244-M: 5.3micron pixel size: magnification therefore 5.3micron/1.1micron = 4.8 (close to nominal magnification of 5x Mitutoyo)
        self.dy = self.dx
        Flaeche = anzahlPixel * self.dx * self.dx
        radius = np.sqrt(Flaeche / np.pi)
        B2 = myFiltered / sumPx * P0 / (self.dx * self.dy * 1e-8)
        I_max = np.max(B2)
        self.I_distr = (self.hdr_image / np.max(self.hdr_image) * I_max)

        self.a0_max = np.sqrt((I_max * lambda0**2) / 1.37e18)

        if verbose:
            print("a0 peak = %.2f" % (self.a0_max))

        lambda0_2 = lambda0 * 1e-4 # Convert um in cm
        I_max_ideal_Stefan = (np.pi * P0) / (4 * lambda0_2**2 * FnumberOap**2)

        Strehl = I_max / I_max_ideal_Stefan

        if verbose:
            print("Strehl = %.2f" % (Strehl))
            print("Intensity = %.2f W/cm^2" % (I_max))

    def plot_focus(self, xlim=(-400, 400), ylim=(-400, 400), color_map='PuRd', **kwargs):
        """Plot the high definition resolution (HDR) focus images post processed from
        focus images during the experiment.

        Raises:
            ValueError: Please calculate or load the HDR intensity first.
        """

        if 'verbose' not in kwargs:
            verbose = False
        else:
            verbose = kwargs['verbose']
            kwargs.pop('verbose', None)

        if self.I_distr is None:
            raise ValueError("Please calculate or load the HDR intensity first.")

        x_mm = np.arange(-self.hdr_image.shape[1]/2, self.hdr_image.shape[1]/2, 1) * self.dx
        y_mm = np.arange(-self.hdr_image.shape[0]/2, self.hdr_image.shape[0]/2, 1) * self.dy
        x, y = np.meshgrid(x_mm,y_mm)

        # Plot
        fig, axes1 = plt.subplots(1, 1, figsize=(5, 5))
        axes = iter(np.ravel(axes1))
        ax = next(axes)

        # Create map
        FS = 20
        lev = np.logspace(15, 20, 1000)
        im = ax.contourf(x, y, self.I_distr,
                         cmap = color_map, norm = LogNorm(), levels = lev)#,zorder=-9)
        #plt.gca().set_rasterization_zorder(-1)
        ax.set_aspect(1)
        ax.set_xlabel('x [\u03BCm]')
        ax.set_ylabel('y [\u03BCm]')
        #ax2.tick_params(labelsize=FS)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])

        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(25))
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        ax.yaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_minor_locator(MultipleLocator(25))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        cbar_ax = fig.add_axes([0.92, 0.11, 0.03, 0.77])
        cbar = fig.colorbar(im, cax=cbar_ax)

        #cbar = fig.colorbar(cf, cax=ax)
        #cbar.ax.set_yticklabels(['16', '17', '18', '19','20'])
        cbar.set_label(r"Intensity [W/cm$^2$]")
        d = np.arange(1, 10)
        D = np.append(d * np.logspace(15, 15, num = 9), d * np.logspace(16, 16, num = 9))
        D = np.append(D, d * np.logspace(17, 17, num = 9))
        D = np.append(D, d * np.logspace(18, 18, num = 9))
        D = np.append(D, d * np.logspace(19, 19, num = 9))
        D = np.append(D, 1e20)
        cbar.set_ticks(D)
        cbar.ax.tick_params(labelsize = FS)
        #for ax in {ax1, ax2}:
        #    ax.set_xlabel(r'$x$')
        #    ax.set_ylabel(r'$y$')
        #    ax.set_aspect('auto')
        #fig.tight_layout()
        #plt.savefig(os.path.join(argin,"analysis","20231018_focus_plot.pdf"), facecolor='white', transparent=False, dpi=300)
        #plt.savefig(os.path.join(argin,"analysis","20231018_focus_plot.png"), facecolor='white', transparent=False, dpi=300)
        #plt.savefig("Comparisons/SG6_beam_Triangules_Mask_a=50_b=150_R=300.png", dpi=400, facecolor='w', edgecolor='w', bbox_inches='tight')
        plt.show()