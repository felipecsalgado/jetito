import numpy as np
from scipy.optimize import curve_fit
#
#

## Packages to calculate the far-field

def twoD_Gaussian(x_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = x_tuple
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    #h = 0
    return g.ravel()


def fit2d(XX, YY, ZZ, initial_guess, output=False):
    """Calculate a 2d-Gaussian fit over a dataset and returns the fit parameters.

    Args:
        XX (ndarray): 2D array representing the X-positions.
        YY (ndarray): 2D array representing the Y-positions.
        ZZ (ndarray): 2D array representing the amplitude of the field at XX and YY positions.
        initial_guess (tuple): initial guess for the fit function (amplitude, xo, yo, sigma_x, sigma_y, theta, offset).
        output (bool): prints the fit parameters into the screen.

    Returns:
        popt (ndarray): fit parameters.
        pcov (ndarray): fit covariance matrix.
    """

    popt, pcov = curve_fit(twoD_Gaussian, (XX, YY), ZZ.ravel(), p0=initial_guess)
    #print("amplitude, xo, yo, sigma_x, sigma_y, theta, offset")
    #print(popt)
    #data_fitted = twoD_Gaussian((freq_X_crop, freq_Y_crop), *popt)
    #
    if output == True:
        print("pcov")
        print(pcov)
        print("")
        print("Amplitude = %.3f"%popt[0])
        print("x0 = %.3f"%popt[1])
        print("y0 = %.3f"%popt[2])
        print("")
        
        print("Sigma_x = %.3f"%popt[3])
        print("2 x Sigma_x = %.3f"%(2*popt[3]))
        print("FWHM_x = %.3f"%(2.35*popt[3]))
        print("")
        print("Sigma_y = %.3f"%popt[4])
        print("2 x Sigma_y = %.3f"%(2*popt[4]))
        print("FWHM_y = %.3f"%(2.35*popt[4]))
    
    return popt, pcov