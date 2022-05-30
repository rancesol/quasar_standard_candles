
import numpy as np
import scipy
from numpy import sin, pi
from scipy.optimize import curve_fit
from center_data import center_data
import numpy.polynomial.polynomial as poly



def func(x, a, b, c) :
    return a*x**2 + b*x + c

def find_quadratic(a, minww) :
    #a = center_data(a)
    overlap = True
    if overlap == True :
        windows = [[a[(a[:,0] >= np.min(a[:,0]) + 0.5*i*np.ptp(a[:,0])/(j+1))
            & (a[:,0] <= np.max(a[:,0]) - (j-0.5*i)*np.ptp(a[:,0])/(j+1))]
            for i in range(2*(j+1)-1)]
            for j in range(int(np.ptp(a[:,0])/minww))]
    else :
        windows = [[a[(a[:,0] >= np.min(a[:,0]) + i*np.ptp(a[:,0])/(j+1))
            & (a[:,0] <= np.max(a[:,0]) - (j-i)*np.ptp(a[:,0])/(j+1))] for i in range(j+1)]
            for j in range(int(2*np.ptp(a[:,0])/minww))]



    ## some windows may have less data points than the number of fitting parameters (3 for quadratic fit)
    ## this will throw an error when fitting so we discard these regions
    ## we require 10 for security
    mask = [[(len(windows[j][i]) > 10) for i in range(len(windows[j]))] for j in range(len(windows))]
    windows = [[windows[j][i] for i in range(len(windows[j])) if mask[j][i]] for j in range(len(windows))]



    ## gaps in the data larger than the minimum window size will return an empty window array
    ## this will throw errors so discard the empty windows
    mask = [np.size(w) > 2 for w in windows[:]]
    windows = [windows[j] for j in range(len(windows)) if mask[j]]


    ## windows with sparse data bias the preferred slope towards lower values
    ## we require a sampling every 15 days on average
    mask = [[(len(windows[j][i]) / np.ptp(windows[j][i][:,0]) > 1/15) for i in range(len(windows[j]))]
            for j in range(len(windows))]
    windows = [[windows[j][i] for i in range(len(windows[j])) if mask[j][i]] for j in range(len(windows))]


    mask = [(len(windows[j]) > 2) for j in range(len(windows))]
    windows = [windows[j] for j in range(len(windows)) if mask[j]]


    if len(windows) > 0 :
        ## make best fits of all the windows and calculate the cooresponding chi^2/DoF
        try :
            fit_params, _ = zip(*[zip(*[curve_fit(func, windows[j][i][:,0], windows[j][i][:,1],
                sigma=windows[j][i][:,2])
                for i in range(len(windows[j])) if (len(windows[j][i]) > 3)])
                for j in range(len(windows)) if (len(windows[j]) > 2)])
   

            chi2dof = [[np.sum(((windows[j][i][:,1] - func(windows[j][i][:,0], fit_params[j][i][0],
                fit_params[j][i][1], fit_params[j][i][2]))/windows[j][i][:,2])**2/(len(windows[j][i])-3))
                for i in range(len(windows[j])) if (len(windows[j]) > 2)]
                for j in range(len(windows)) if (len(windows[j]) > 2)]
    
    
            ## we no longer need the list of list of arrays so we can flatten windows
            ## we also collect the window widths to correspond slopes to their time-scales
            windows = [window for sublist in windows for window in sublist]
            window_width = [np.ptp(windows[i][:,0]) for i in range(len(windows))]
            window_midpoints = [(windows[i][np.argmax(windows[i][:,1]),0]
                + windows[i][np.argmin(windows[i][:,1]),0])/2 for i in range(len(windows))]
            fit_params = np.vstack([fit_params[i] for i in range(len(fit_params))])
            chi2dof   = np.hstack(chi2dof)
    
            ## for analysis purposes we collect the N lines with the lowest (or highest) chi^2/dof
            ## note: this is not being used right now. all lines are collected.
            N = 100
            min_i = np.argsort(chi2dof)
            best_fit = [np.vstack((windows[i][:,0], fit_params[i,0]*windows[i][:,0]**2 + fit_params[i,1]*windows[i][:,0] + fit_params[i,2])).T
                    for i in min_i[:]]
    
            return window_width, window_midpoints, fit_params, chi2dof, best_fit
        except RuntimeError :
            return 0, 0, 0, 0, 0

    else :
        return 0, 0, 0, 0, 0


