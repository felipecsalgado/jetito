import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, ScalarFormatter)
from matplotlib.colors import LogNorm

from matplotlib import cm
from PIL import Image
import copy as copy
from scipy.interpolate import interp1d
import os as os
import glob as glob

# Configures the plots
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)            # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
ml = MultipleLocator(2)


class espec:

    def __init__(self, calib_espec_experimental=3e-7, parent_folder='e-spec', **kwargs):
        self.calib_espec_experimental = calib_espec_experimental
        self.parent_folder = parent_folder

        if 'set_folder' not in kwargs:
            self.set_folder = None
        else:
            self.set_folder = kwargs['set_folder']
            kwargs.pop('set_folder', None)

    def getEspec(self, shotnumber, calib_file="../ebeam/espec_calib_files/", 
                 **kwargs):

        # savename = date+set_pic+'_'+picturenumber

        if 'verbose' not in kwargs:
            verbose = False
        else:
            verbose = kwargs['verbose']
            kwargs.pop('verbose', None)

        # Manage if the save folder is given. If so, save the plots
        if 'save_folder' not in kwargs:
            save = False
        else:
            save = True
            save_folder = kwargs['save_folder']
            kwargs.pop('save_folder')

        # Check if a set folder is specified
        if self.set_folder != None:  # There is a set folder
            image1_name = os.path.join(self.parent_folder,"e-spec_1",self.set_folder,str(shotnumber))
            image2_name = os.path.join(self.parent_folder,"e-spec_2",self.set_folder,str(shotnumber))
            image3_name = os.path.join(self.parent_folder,"e-spec_3",self.set_folder,str(shotnumber))
        else:
            image1_name = os.path.join(self.parent_folder,"e-spec_1",str(shotnumber))
            image2_name = os.path.join(self.parent_folder,"e-spec_2",str(shotnumber))
            image3_name = os.path.join(self.parent_folder,"e-spec_3",str(shotnumber))

        # Get the images path and load them
        image1_name = glob.glob(image1_name + "*")[0]
        image1 = Image.open(image1_name)
        if verbose:
            print("Image e-spec 1:" + image1_name)

        image2_name = glob.glob(image2_name + "*")[0]
        image2 = Image.open(image2_name)
        if verbose:
            print("Image e-spec 2:" + image2_name)

        image3_name = glob.glob(image3_name + "*")[0]
        image3 = Image.open(image3_name)
        if verbose:
            print("Image e-spec 3:" + image3_name)

        # Post-processing low energy image
        im1 = copy.copy(image1)
        im1 = im1.rotate(-0.05,resample=Image.Resampling.BICUBIC)
        im1_crop = im1.crop((200, 800, 2200, 1040))
        img_array = np.array(im1_crop)

        if save:
            fig = plt.figure(figsize=(10, 3))
            plt.imshow(img_array)
            save_low_energy = save_folder + str(shotnumber) +  "_low_energy.png"
            fig.savefig(save_low_energy, dpi=450, facecolor='white', format='png', bbox_inches='tight')
            plt.show()

        # Post-processing mid-energy image (center energy)
        im2 = copy.copy(image2)
        im2 = im2.rotate(0.1,resample=Image.Resampling.BICUBIC)
        im2_crop = im2.crop((180, 700, 2200, 990))
        scale2 = 5.74/5.75
        oldsize_2 = im2_crop.size
        im2_crop = im2_crop.resize((int(oldsize_2[0]*scale2),int(oldsize_2[1]*scale2)))
        img_array = np.array(im2_crop)

        if save:
            fig = plt.figure(figsize=(10, 3))
            plt.imshow(img_array)
            save_mid_energy = save_folder + str(shotnumber) + "_mid_energy.png"
            fig.savefig(save_mid_energy, dpi=450, facecolor='white', format='png', bbox_inches='tight')
            plt.show()

        # Post-processing high-energy image
        im3 = copy.copy(image3)
        im3_crop = im3.crop((180, 700, 2250, 990))
        scale3 = 5.74/5.7
        oldsize_3 = im3_crop.size
        im3_crop = im3_crop.resize((int(oldsize_3[0]*scale3),int(oldsize_3[1]*scale3)))
        img_array = np.array(im3_crop)

        if save:
            fig = plt.figure(figsize=(10, 3))
            plt.imshow(img_array)
            save_high_energy = save_folder + str(shotnumber) + "_high_energy.png"
            fig.savefig(save_high_energy, dpi=450, facecolor='white', format='png', bbox_inches='tight')
            plt.show()

        # Crop images to make them fit together when joined
        im1_crop = im1_crop.crop((132, 0, 2000, 240))
        im2_crop = im2_crop.crop((180, 23, 1967, 263))
        im3_crop = im3_crop.crop((0, 31, 2045, 271))

        # Join the images
        new_all = Image.new('I',(im1_crop.size[0]+im2_crop.size[0]+im3_crop.size[0],im1_crop.size[1]))
        new_all.paste(im3_crop,(0,0))
        new_all.paste(im2_crop,(im3_crop.size[0],0))
        new_all.paste(im1_crop,(im2_crop.size[0]+im3_crop.size[0],0))

        # Display and save joined image
        img_array = np.array(new_all)

        #distance_sorce_spec = 1.500
        calib_px_per_mm = 5.74
        distance_sorce_spec = 20*calib_px_per_mm
        background = 375

        if save:
            fig, axes1 = plt.subplots(1, 1, figsize=(10, 5))
            axes = iter(np.ravel(axes1))
            ax = next(axes)
            ax.imshow(img_array,extent=((np.shape(img_array)[1]+distance_sorce_spec)/calib_px_per_mm,+distance_sorce_spec/calib_px_per_mm,-im3_crop.size[1]/2/calib_px_per_mm,im3_crop.size[1]/2/calib_px_per_mm))
            ax.set_xlabel('Deflection (mm)', fontsize=12)
            ax.set_ylabel('y (mm)', fontsize=12)
            ax.tick_params(axis='y', labelsize=8)
            ax.set_xlim(1000, 20)
            save_espec_energy = save_folder + str(shotnumber) + "_e-spec_energy.png"
            fig.savefig(save_espec_energy, dpi=450, facecolor='white', format='png', bbox_inches='tight')
            plt.show()

        # Rescale the deflection axis to energy according to the calibraiton
        # of the spectrometer. The calibration file is located in the same folder
        # of the module.
        lin_spec = np.sum(img_array, axis=0)

        # Load calibration file fo the spectrometer
        file_calibration = calib_file + "e_calib_long_espec.txt"
        Energie_kalib = np.genfromtxt(file_calibration).T
        pxlatEnergy = Energie_kalib[0]
        Energy = Energie_kalib[1]

        max_px = 74
        min_px = 5600
        Energyatpxl_inter = interp1d(pxlatEnergy, Energy, kind='cubic')
        ablenkung_in_pxl = np.linspace(max_px, min_px, min_px - max_px)

        x_achse = Energyatpxl_inter(ablenkung_in_pxl)

        dE_per_dpxl = np.diff(x_achse) / np.diff(ablenkung_in_pxl)
        dE_per_dpxl = np.append(dE_per_dpxl, dE_per_dpxl[-1])

        # Plot the linear spectrum
        fig, axes1 = plt.subplots(1, 1, figsize=(4, 3))
        axes = iter(np.ravel(axes1))
        ax = next(axes)

        ax.plot(x_achse,
                -1 * lin_spec[max_px:min_px] / dE_per_dpxl * self.calib_espec_experimental)

        ax.set_xlim(10, 1000)

        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel(r'$\frac{\mathrm{d}Q}{\mathrm{d}E}$ $\left(\frac{\mathrm{pC}}{\mathrm{MeV}}\right)$')

        ax.yaxis.set_major_locator(MultipleLocator(2.5))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))

        ax.xaxis.set_major_locator(MultipleLocator(250))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(50))

        ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.grid(which='minor', axis='y', color='gray', linestyle='-', linewidth=0.25, alpha=0.2)
        ax.grid(which='major', axis='x', color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.grid(which='minor', axis='x', color='gray', linestyle='-', linewidth=0.25, alpha=0.2)

        if save:
            save_linspec_energy = save_folder + str(shotnumber) + "_e-spec_linear.png"
            fig.savefig(save_linspec_energy, dpi=450, facecolor='white', format='png', bbox_inches='tight')

        plt.show()
