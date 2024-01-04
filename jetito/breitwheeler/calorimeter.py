import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from matplotlib.ticker import (MultipleLocator)
from pathlib import Path
import os.path
import pandas as pd
#
from scipy import constants as const
from scipy import integrate
import glob
import os
import cv2
from pathlib import Path

import pymc as pm  # for the Bayes estimation
import arviz as az

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


class bwset:

    def __init__(self, setnumber='025', ref_setnumber='20',
                 roi_csi_bunker=np.s_[680:845, 365:535],
                 toi_picoscope=np.s_[270:350],
                 data_parent_folder='./'):
        """
        Constructor of the Breit-Wheeler class for data analysis of the calorimeter datasets

        Args:
            setnumber (str, optional): Set number of the dataset with bkg+signal to be analyzed. Defaults to '025'.
            ref_setnumber (str, optional): Backgroud dataset where no signal is present. Defaults to '20'.
            roi_csi_bunker (_type_, optional): Region-of-interest (roi) of the CSI image in the bunker to help on
            the signal filtering. Defaults to np.s_[680:845, 365:535].
            toi_picoscope (_type_, optional): Time-of-interest (toi) where the signal of the calorimenter will be
            integrated. Defaults to np.s_[270:350].
            data_parent_folder (str, optional): Parent folder where data is stored. Defaults to './'.
        """

        self.setnumber = setnumber
        self.parent_folder = data_parent_folder
        self.ref_setnumber = ref_setnumber
        self.roi_csi_bunker = roi_csi_bunker
        self.toi_picoscope = toi_picoscope

    def get_reference_csi(self, folder="BunkerLanex (21768927)/", **kwargs):
        """
        Get the CSI signal of the background signal of the CSI bunker. This is used to
        select the useful shotnumebrs in the signal dataset that will be used in the
        analysis of the calorimeter signals.

        Args:
            folder (str, optional): Folder where the CSI images are stored. Defaults to "BunkerLanex (21768927)/".
            verbose (bool, optional): Print outputs. Defaults to False.
        """

        if 'verbose' not in kwargs:
            verbose = False
        else:
            verbose = kwargs['verbose']
            kwargs.pop('verbose', None)

        if verbose:
            print("Start getting the reference for filtering...")

        # list all reference images in the set
        list_files_ref = sorted(glob.glob(self.parent_folder + folder + "*_" + str(self.ref_setnumber) + "_*.png"))

        if verbose:
            print("Number of reference shots found: %d" % (len(list_files_ref)))

        # Get mean and standard deviation of counts from the sum of counts in the ROI
        sum_ref = []

        for f in list_files_ref:
            img_dummy = cv2.imread(f, -1)[self.roi_csi_bunker]
            sum_aux = np.sum(img_dummy)
            #
            sum_ref.append(sum_aux)
        #
        self.sum_ref = np.asarray(sum_ref)
        self.mean_of_sum_ref = np.mean(sum_ref)
        self.std_of_sum_ref = np.std(sum_ref)

        if verbose:
            print("Mean counts of the sum of the counts in the reference images = %.1e +- %.1e counts" %
                  (self.mean_of_sum_ref, self.std_of_sum_ref))

    def filter_useful_data(self, threshold_ratio=1.1, folder="BunkerLanex (21768927)/", **kwargs):
        """
        Filter the data from the calorimeter channels based on the shot numbers selected
        by analyzing the shots that are going through the bunker using the CSI images.

        Args:
            threshold_ratio (float, optional): Threshold ratio to consider a shot useful or not. Defaults to 1.1.
            folder (str, optional): Folder where the CSI images are stored. Defaults to "BunkerLanex (21768927)/".
            verbose (bool, optional): _description_. Defaults to True.
        """

        if 'verbose' not in kwargs:
            verbose = False
        else:
            verbose = kwargs['verbose']
            kwargs.pop('verbose', None)

        if verbose:
            print("Start filtering data...")

        # Manage the input parameters of the method
        if 'save_folder' not in kwargs:
            save_filtered = False
            if verbose:
                print("NOT saving filtered ROI of the filtered data.")
        else:
            save_filtered = True
            save_path = self.parent_folder + folder + kwargs['save_folder']
            kwargs.pop('save_folder', None)
            # check if path exists, otherwise, create
            Path(save_path).mkdir(parents=True, exist_ok=True)
            if verbose:
                print("Saving filtered ROI of the filtered data in the folder:")
                print(save_path)

        list_files_set = sorted(glob.glob(self.parent_folder + folder + "*_" + str(self.setnumber) + "_*.png"))

        if verbose:
            print("Number of files found for dataset %s = %d" % (self.setnumber, len(list_files_set)))

        # Get mean and standard deviation of counts from the sum of counts in the ROI
        self.sum_set = []
        self.useful_shotnumbers = []

        for f in list_files_set:
            img_dummy = cv2.imread(f, -1)[self.roi_csi_bunker]
            sum_aux = np.sum(img_dummy)

            # Check if the current shotnumber file is larger than the reference shot
            if (sum_aux >= self.mean_of_sum_ref * threshold_ratio):
                # get shotnumber of current file
                aux_shotnumber = int(f.split('_')[2])
                # sum the counts in the current file calculated outside of the if-statement
                self.sum_set.append(sum_aux)
                # store the shotnumber of current file
                self.useful_shotnumbers.append(aux_shotnumber)

                if save_filtered:
                    cv2.imwrite(save_path + f.split('_')[1] + '_' + f.split('_')[2] + '.png', img_dummy)

        self.mean_of_sum_set = np.mean(np.asarray(self.sum_set))
        self.std_of_sum_set = np.std(np.asarray(self.sum_set))

        if verbose:
            print("Mean counts of the sum of the counts in the set images = %.1e" % (self.mean_of_sum_set))
            print("Total number of useful data in the set = %d" % (len(self.useful_shotnumbers)))

    def getNpe(self, list_files_pico, R, gain_pmt, QE):
        """
        Get the integrated calorimenter signal for each channel within the time-of-interest (toi)
        region.

        Args:
            list_files_pico (list): List of picoscope files to have the signal integrated.
            R (double): Resistance of the PMTs.
            gain_pmt (list): List of gains of the PMTs connected on each channel.
            QE (double): Quantum efficiency of the PMTs.

        Returns:
            _type_: _description_
            Npe_dict (list): List with the integrated signals for each channel.
            mean_Npe_dict (numpy.array): Array with the mean Npe of each channel signal.
            std_Npe_dict (numpy.array): Array with the standard deviation of each channel signal.
        """
        Npe_A = []
        Npe_B = []
        Npe_C = []
        Npe_D = []

        for f in list_files_pico:
            if os.path.isfile(f):
                # Reads the csv file of PMTs
                data_csv_single = pd.read_csv(f, delimiter=',', decimal='.')

                # Re-level the data from the picoscopes
                # invert the data
                data_A = data_csv_single['A'].values * -1
                data_B = data_csv_single['B'].values * -1
                data_C = data_csv_single['C'].values * -1
                data_D = data_csv_single['D'].values * -1

                # Get the level at end of picoscope data 150 points after start
                length_arrays = len(data_csv_single['t'].values)  # get the length of the data arrays
                idx_mean_relevel = 100
                level_A = np.mean(data_A[:idx_mean_relevel])
                level_B = np.mean(data_B[:idx_mean_relevel])
                level_C = np.mean(data_C[:idx_mean_relevel])
                level_D = np.mean(data_D[:idx_mean_relevel])
                # print(level_A, level_B, level_C, level_D)
                # Re-level the data to the average min
                data_A = data_A - level_A
                data_B = data_B - level_B
                data_C = data_C - level_C
                data_D = data_D - level_D

                # Calculate the number of photoelectrons at the output of the PMTs
                int_A = integrate.trapz(y=data_A[self.toi_picoscope],
                                        x=data_csv_single['t'].values[self.toi_picoscope])
                int_B = integrate.trapz(y=data_B[self.toi_picoscope],
                                        x=data_csv_single['t'].values[self.toi_picoscope])
                int_C = integrate.trapz(y=data_C[self.toi_picoscope],
                                        x=data_csv_single['t'].values[self.toi_picoscope])
                int_D = integrate.trapz(y=data_D[self.toi_picoscope],
                                        x=data_csv_single['t'].values[self.toi_picoscope])

                # Calculate the number of photons hitting the PMTs
                Npe_A.append(int_A / R / gain_pmt[0] / const.e / QE)
                Npe_B.append(int_B / R / gain_pmt[1] / const.e / QE)
                Npe_C.append(int_C / R / gain_pmt[2] / const.e / QE)
                Npe_D.append(int_D / R / gain_pmt[3] / const.e / QE)

        # Get mean and std values
        mean_Npe_A = np.mean(Npe_A)
        std_Npe_A = np.std(Npe_A)

        mean_Npe_B = np.mean(Npe_B)
        std_Npe_B = np.std(Npe_B)

        mean_Npe_C = np.mean(Npe_C)
        std_Npe_C = np.std(Npe_C)

        mean_Npe_D = np.mean(Npe_D)
        std_Npe_D = np.std(Npe_D)

        mean_Npe_dict = {'A': mean_Npe_A, 'B': mean_Npe_B, 'C': mean_Npe_C, 'D': mean_Npe_D}
        std_Npe_dict = {'A': std_Npe_A, 'B': std_Npe_B, 'C': std_Npe_C, 'D': std_Npe_D}
        Npe_dict = {'A': Npe_A, 'B': Npe_B, 'C': Npe_C, 'D': Npe_D}

        return Npe_dict, mean_Npe_dict, std_Npe_dict

    def get_reference_picoscope(self, folder="picoscope/",
                                gain_pmt_pico17=np.asarray([3.6e3, 3.6e3, 3.6e3, 1.8e3]),
                                gain_pmt_pico22=np.asarray([3.6e3, 3.6e3, 3.6e3, 1.8e3]),
                                gain_pmt_pico15=np.asarray([3.6e3, 3.6e3, 3.6e3, 1.8e3]),
                                R=50, QE=0.27, verbose=True, **kwargs):

        if 'verbose' not in kwargs:
            verbose = False
        else:
            verbose = kwargs['verbose']
            kwargs.pop('verbose', None)

        folder_scope_pico17 = 'ho234_017/'
        folder_scope_pico22 = 'ho234_022/'

        # get files in the specific folder
        list_files_pico_17 = []
        list_files_pico_22 = []

        # pico 17
        list_files_pico_17 = sorted(glob.glob(folder + folder_scope_pico17 + "*_" +
                                              str(int(self.ref_setnumber)) + "__ho234_017*.csv"))

        # get list of useful_shotnumbers per pico 17
        self.useful_shotnumbers_ref_pico17 = []
        for f in list_files_pico_17:
            aux_pico17 = f.split('_')[3]
            if aux_pico17:
                self.useful_shotnumbers_ref_pico17.append(int(aux_pico17))

        # Pico 17
        if verbose:
            print("")
            print("Processing reference data for %s" % folder_scope_pico17)
            print("Found %d csv files" % (len(list_files_pico_17)))

        self.pico17_Npe_dict_ref, pico17_mean_Npe_dict, pico17_std_Npe_dict = self.getNpe(list_files_pico_17,
                                                                                          R=R,
                                                                                          gain_pmt=gain_pmt_pico17,
                                                                                          QE=QE)
        if verbose:
            print("Finished pico17")

        # Pico 22
        list_files_pico_22 = sorted(glob.glob(folder + folder_scope_pico22 + "*_" +
                                              str(int(self.ref_setnumber)) + "__ho234_022*.csv"))

        # get list of useful_shotnumbers per pico 22
        self.useful_shotnumbers_ref_pico22 = []
        for f in list_files_pico_22:
            aux_pico22 = f.split('_')[3]
            if aux_pico22:
                self.useful_shotnumbers_ref_pico22.append(int(aux_pico22))

        if verbose:
            print("")
            print("Processing reference data for %s" % folder_scope_pico22)
            print("Found %d csv files" % (len(list_files_pico_22)))

        self.pico22_Npe_dict_ref, pico22_mean_Npe_dict, pico22_std_Npe_dict = self.getNpe(list_files_pico_22,
                                                                                          R=R,
                                                                                          gain_pmt=gain_pmt_pico22,
                                                                                          QE=QE)
        if verbose:
            print("Finished pico22")

        # Pico 15
        """
        Need to copy the same methods from pico17 or pico22 and change the variable names.
        Still to do!
        """

        # Dummy pico 15 results
        # remove after loading pico15 data
        pico15_mean_Npe_dict = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        pico15_std_Npe_dict = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        pico15_Npe_dict = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

        # Rearrange of data
        # positrons
        self.positrons_mean_Npe_ref = (pico15_mean_Npe_dict['B'],
                                       pico15_mean_Npe_dict['A'],
                                       pico22_mean_Npe_dict['D'],
                                       pico22_mean_Npe_dict['C'],
                                       pico22_mean_Npe_dict['B'])
        self.positrons_mean_Npe_ref = np.asarray(self.positrons_mean_Npe_ref)

        self.positrons_std_Npe_ref = (pico15_std_Npe_dict['B'],
                                      pico15_std_Npe_dict['A'],
                                      pico22_std_Npe_dict['D'],
                                      pico22_std_Npe_dict['C'],
                                      pico22_std_Npe_dict['B'])
        self.positrons_std_Npe_ref = np.asarray(self.positrons_std_Npe_ref)

        # electrons
        self.electrons_mean_Npe_ref = (pico22_mean_Npe_dict['A'],
                                       pico17_mean_Npe_dict['D'],
                                       pico17_mean_Npe_dict['C'],
                                       pico17_mean_Npe_dict['B'],
                                       pico17_mean_Npe_dict['A'])
        self.electrons_mean_Npe_ref = np.asarray(self.electrons_mean_Npe_ref)

        self.electrons_std_Npe_ref = (pico22_std_Npe_dict['A'],
                                      pico17_std_Npe_dict['D'],
                                      pico17_std_Npe_dict['C'],
                                      pico17_std_Npe_dict['B'],
                                      pico17_std_Npe_dict['A'])
        self.electrons_std_Npe_ref = np.asarray(self.electrons_std_Npe_ref)

    def get_filtered_picoscope(self, folder="picoscope/",
                               gain_pmt_pico17=np.asarray([3.6e3, 3.6e3, 3.6e3, 1.8e3]),
                               gain_pmt_pico22=np.asarray([3.6e3, 3.6e3, 3.6e3, 1.8e3]),
                               gain_pmt_pico15=np.asarray([3.6e3, 3.6e3, 3.6e3, 1.8e3]),
                               R=50, QE=0.27, verbose=True):

        folder_scope_pico17 = 'ho234_017/'
        folder_scope_pico22 = 'ho234_022/'
        # get files in the specific folder
        list_files_pico_17 = []
        list_files_pico_22 = []

        # get list of useful_shotnumbers per pico
        self.useful_shotnumbers_set_pico17 = []
        self.useful_shotnumbers_set_pico22 = []
        self.useful_shotnumbers_set_pico15 = []

        for nshot in self.useful_shotnumbers:
            # pico 17
            aux_pico17 = sorted(glob.glob(folder + folder_scope_pico17 + "*_" +
                                          str(nshot) + "_*.csv"))
            if aux_pico17:
                list_files_pico_17.append(aux_pico17[0])
                self.useful_shotnumbers_set_pico17.append(nshot)

            # pico 22
            aux_pico22 = sorted(glob.glob(folder + folder_scope_pico22 + "*_" +
                                          str(nshot) + "_*.csv"))
            if aux_pico22:
                list_files_pico_22.append(aux_pico22[0])
                self.useful_shotnumbers_set_pico22.append(nshot)

        # Pico 17
        if verbose:
            print("")
            print("Processing reference data for %s" % folder_scope_pico17)
            print("Found %d csv files" % (len(list_files_pico_17)))

        self.pico17_Npe_dict_set, pico17_mean_Npe_dict, pico17_std_Npe_dict = self.getNpe(list_files_pico_17,
                                                                                          R=R,
                                                                                          gain_pmt=gain_pmt_pico17,
                                                                                          QE=QE)
        if verbose:
            print("Finished pico17!")

        # Pico 22
        if verbose:
            print("")
            print("Processing reference data for %s" % folder_scope_pico22)
            print("Found %d csv files" % (len(list_files_pico_22)))

        self.pico22_Npe_dict_set, pico22_mean_Npe_dict, pico22_std_Npe_dict = self.getNpe(list_files_pico_22,
                                                                                          R=R,
                                                                                          gain_pmt=gain_pmt_pico22,
                                                                                          QE=QE)
        if verbose:
            print("Finished pico22!")

        # Pico 15
        """
        Create the pico 15 based on the pico17 and pico22
        """
        # Dummy pico 15 results
        # remove after loading pico15 data
        pico15_mean_Npe_dict = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        pico15_std_Npe_dict = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        pico15_Npe_dict = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

        # Rearrange of data
        # positrons
        self.positrons_mean_Npe_set = (pico15_mean_Npe_dict['B'],
                                       pico15_mean_Npe_dict['A'],
                                       pico22_mean_Npe_dict['D'],
                                       pico22_mean_Npe_dict['C'],
                                       pico22_mean_Npe_dict['B'])
        self.positrons_mean_Npe_set = np.asarray(self.positrons_mean_Npe_set)

        self.positrons_std_Npe_set = (pico15_std_Npe_dict['B'],
                                      pico15_std_Npe_dict['A'],
                                      pico22_std_Npe_dict['D'],
                                      pico22_std_Npe_dict['C'],
                                      pico22_std_Npe_dict['B'])
        self.positrons_std_Npe_set = np.asarray(self.positrons_std_Npe_set)

        # electrons
        self.electrons_mean_Npe_set = (pico22_mean_Npe_dict['A'],
                                       pico17_mean_Npe_dict['D'],
                                       pico17_mean_Npe_dict['C'],
                                       pico17_mean_Npe_dict['B'],
                                       pico17_mean_Npe_dict['A'])
        self.electrons_mean_Npe_set = np.asarray(self.electrons_mean_Npe_set)

        self.electrons_std_Npe_set = (pico22_std_Npe_dict['A'],
                                      pico17_std_Npe_dict['D'],
                                      pico17_std_Npe_dict['C'],
                                      pico17_std_Npe_dict['B'],
                                      pico17_std_Npe_dict['A'])
        self.electrons_std_Npe_set = np.asarray(self.electrons_std_Npe_set)

    def getChargeArray(self, charge_table, useful_shots):
        """
        Get a list with the charge for the selected shotnumbers for an specific dataset.

        Args:
            charge_table (pandas.table): Pandas table with the charge for each shotnumber of the given dataset.
            useful_shots (list): List of shotnumbers to retrieve their charge from the table.

        Returns:
            charge_set_useful_shotnumbers (list): list with thecharge of the given shotnumbers.
        """
        charge_set_useful_shotnumbers = []
        for shot in useful_shots:
            # Set shots
            # print(charge_table)
            idx_set_aux = np.where(charge_table.values[:, 0] == int(shot))[0][0]
            # print(idx_set_aux)
            charge = np.abs(charge_table.values[idx_set_aux, 1])
            charge_set_useful_shotnumbers.append(charge)
        return charge_set_useful_shotnumbers

    def chargeNormalization(self, file_reference, file_set, delimiter=' ', decimal='.', verbose=True):
        #
        if verbose:
            print("Normalize Npe by charge....")

        charge_reference_table = pd.read_csv(file_reference, delimiter=delimiter, decimal=decimal)
        charge_set_table = pd.read_csv(file_set, delimiter=delimiter, decimal=decimal)

        # Charge for the reference background
        self.useful_shotnumbers_charge_ref_pico17 = self.getChargeArray(charge_reference_table,
                                                                        self.useful_shotnumbers_ref_pico17)
        if verbose:
            print("Number of charge analyzed for background of pico 17: %d" %
                  (len(self.useful_shotnumbers_charge_ref_pico17)))

        self.useful_shotnumbers_charge_ref_pico22 = self.getChargeArray(charge_reference_table,
                                                                        self.useful_shotnumbers_ref_pico22)
        if verbose:
            print("Number of charge analyzed for background of pico 22: %d" %
                  (len(self.useful_shotnumbers_charge_ref_pico22)))

        # Charge for the Set background
        self.useful_shotnumbers_charge_set_pico17 = self.getChargeArray(charge_set_table,
                                                                        self.useful_shotnumbers_set_pico17)
        if verbose:
            print("Number of charge analyzed for set of pico 17: %d" %
                  (len(self.useful_shotnumbers_charge_set_pico17)))

        self.useful_shotnumbers_charge_set_pico22 = self.getChargeArray(charge_set_table,
                                                                        self.useful_shotnumbers_set_pico22)
        if verbose:
            print("Number of charge analyzed for set of pico 22: %d" %
                  (len(self.useful_shotnumbers_charge_set_pico22)))

        # Calculate the charge normalized Npe
        # Copy variables
        self.pico17_Npe_dict_set_norm = self.pico17_Npe_dict_set
        self.pico17_Npe_dict_ref_norm = self.pico17_Npe_dict_ref

        self.pico22_Npe_dict_set_norm = self.pico22_Npe_dict_set
        self.pico22_Npe_dict_ref_norm = self.pico22_Npe_dict_ref

        # Normalization
        for key in self.pico17_Npe_dict_set.keys():
            # pico17
            self.pico17_Npe_dict_set_norm[key] = [i / j for i, j in zip(self.pico17_Npe_dict_set[key],
                                                                        self.useful_shotnumbers_charge_set_pico17)]
            # Remove the negative elements from the list
            self.pico17_Npe_dict_set_norm[key] = [ele for ele in self.pico17_Npe_dict_set_norm[key] if ele > 0]

            self.pico17_Npe_dict_ref_norm[key] = [i / j for i, j in zip(self.pico17_Npe_dict_ref[key],
                                                                        self.useful_shotnumbers_charge_ref_pico17)]
            # Remove the negative elements from the list
            self.pico17_Npe_dict_ref_norm[key] = [ele for ele in self.pico17_Npe_dict_ref_norm[key] if ele > 0]

            # pico22
            self.pico22_Npe_dict_set_norm[key] = [i / j for i, j in zip(self.pico22_Npe_dict_set[key],
                                                                        self.useful_shotnumbers_charge_set_pico22)]
            # Remove the negative elements from the list
            self.pico22_Npe_dict_set[key] = [ele for ele in self.pico22_Npe_dict_set[key] if ele > 0]

            self.pico22_Npe_dict_ref_norm[key] = [i / j for i, j in zip(self.pico22_Npe_dict_ref[key],
                                                                        self.useful_shotnumbers_charge_ref_pico22)]
            # Remove the negative elements from the list
            self.pico22_Npe_dict_ref[key] = [ele for ele in self.pico22_Npe_dict_ref[key] if ele > 0]

        if verbose:
            print("Normalized Npe of the reference and data were calculated")

    def getBayesMeanStd(self, observedSignal, observedBkg, signal_min_Npe=10e3, signal_max_Npe=100e3,
                        bkg_min_Npe=1e3, bkg_max_Npe=80e3, verbose=True, **kwargs):

        # Manage the input parameters of the method
        if 'save_folder' not in kwargs:
            save_filtered = False
            if verbose:
                print("NOT saving traces results.")
        else:
            save_filtered = True
            save_path = self.parent_folder + kwargs['save_folder']
            kwargs.pop('save_folder', None)
            # check if path exists, otherwise, create
            Path(save_path).mkdir(parents=True, exist_ok=True)
            if verbose:
                print("Saving traces results in the folder:")
                print(save_path)

        if 'label' not in kwargs:
            label = 'single_scope'
        else:
            label = kwargs['label']
            kwargs.pop('label', None)

        with pm.Model() as model:
            signal_mean = pm.Uniform('signal_mean', signal_min_Npe, signal_max_Npe)
            bg_mean = pm.Uniform('bg_mean', bkg_min_Npe, bkg_max_Npe)

            observations = pm.Poisson('counts', bg_mean+signal_mean, observed=observedSignal)
            bg_obs = pm.Poisson('bg_counts', bg_mean, observed=observedBkg)

            step = pm.Metropolis(vars=[signal_mean, bg_mean])
            trace = pm.sample(15000, step=[step], tune=5000, chains=1)

            trace2 = pm.compute_log_likelihood(trace)

        result_trace = az.summary(trace, kind='stats')

        if save_filtered:
            signal_mean_post = trace.posterior.signal_mean.data[0]
            bg_mean_post = trace.posterior.bg_mean.data[0]

            # Background
            fig, axes1 = plt.subplots(1, 2, figsize=(8, 3), layout='constrained')
            axes = iter(np.ravel(axes1))
            #
            ax = next(axes)
            ax.plot(bg_mean_post)
            ax.set_xlabel('trial')
            ax.set_ylabel('bg mean')

            ax = next(axes)
            n, bin, _ = ax.hist(bg_mean_post, bins=20, density=True)
            ax.set_xlabel('bg mean')
            ax.set_ylabel('PDF')

            plt.savefig(save_path + label + "_bkg.png", dpi=400, facecolor='white',
                        format='png', bbox_inches='tight')
            plt.show()

            # Signal
            # Background
            fig, axes1 = plt.subplots(1, 2, figsize=(8, 3), layout='constrained')
            axes = iter(np.ravel(axes1))

            ax = next(axes)
            ax.plot(signal_mean_post)
            ax.set_xlabel('trial')
            ax.set_ylabel('signal mean')

            ax = next(axes)
            n, bin, _ = ax.hist(signal_mean_post, bins=20, density=True)
            ax.set_xlabel('signal mean')
            ax.set_ylabel('PDF')

            plt.savefig(save_path + label + "_signal.png", dpi=400, facecolor='white',
                        format='png', bbox_inches='tight')
            plt.show()

            if verbose:
                print("Signal = %.1f +/- %.1f" % (result_trace['mean']['signal_mean'],
                                                  result_trace['sd']['signal_mean']))
                print("Background = %.1f +/- %.1f" % (result_trace['mean']['bg_mean'],
                                                      result_trace['sd']['bg_mean']))

        return (result_trace['mean']['signal_mean'], result_trace['sd']['signal_mean'],
                result_trace['mean']['bg_mean'], result_trace['sd']['bg_mean'])
