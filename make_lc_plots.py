
import numpy as np


import matplotlib.pyplot as plt
cm = 1/2.54
plt.style.use(['science', 'no-latex'])
plt.rc('font', size=8)
plt.rcParams['font.family'] = 'monospace'
width_def = 2*15.466*cm
height_def = width_def/2.55/2


def plot_lc(lc, xlabel='Days', ylabel='Flux', title='', width=width_def, height=height_def) :
    fig, axs = plt.subplots(1,1, figsize=(width,height))
    markers, caps, bars = axs.errorbar(lc[:,0], lc[:,1], yerr=lc[:,2], xerr=None, ls='none', c='C3',
            capsize=2, elinewidth=0.7, markeredgewidth=1)
    axs.set_ylim(1.5*min(lc[:,1]), 1.5*max(lc[:,1]))
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_title(title)



def plot_lc_with_quad_fits(lc, best_fit, xlabel='Days', ylabel='Flux', title='', width=width_def, height=height_def) :
    fig, axs = plt.subplots(1,1, figsize=(width,height))
    markers, caps, bars = axs.errorbar(lc[:,0], lc[:,1], yerr=lc[:,2], xerr=None, ls='none', c='C3',
            capsize=2, elinewidth=0.7, markeredgewidth=1)
    [axs.plot(best_fit[i][:,0], best_fit[i][:,1], c='k', lw=0.5, alpha=0.3) for i in range(len(best_fit))]
    axs.set_ylim(1.5*min(lc[:,1]), 1.5*max(lc[:,1]))
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_title(title)


def plot_lc_with_quad_slopes(lc, fit_params, window_midpoints, window_width, xlabel='Days', ylabel='Flux', title='', width=width_def, height=height_def) :
    fig, axs = plt.subplots(1,1, figsize=(width,height))
    axs.scatter(lc[:,0], lc[:,1], c='C3', s=1)
    markers, caps, bars = axs.errorbar(lc[:,0], lc[:,1], yerr=lc[:,2], xerr=None, ls='none', c='C3',
            capsize=2, elinewidth=0.7, markeredgewidth=1)
    [axs.plot(np.linspace(window_midpoints[i]-0.5*window_width[i],
        window_midpoints[i]+0.5*window_width[i], 5),
        [(2*fit_params[i,0]*window_midpoints[i] + fit_params[i,1])*(ti-window_midpoints[i])
            + (fit_params[i,0]*window_midpoints[i]**2 + fit_params[i,1]*window_midpoints[i]
                + fit_params[i,2]) for ti in np.linspace(window_midpoints[i]-0.5*window_width[i],
                    window_midpoints[i]+0.5*window_width[i],5)], c='k', linewidth=0.5, alpha=0.5)
                for i in range(len(window_midpoints))]
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_title(title)


from scipy.optimize import curve_fit
from scipy.stats import norm, skewnorm

def skewnormfit(x, a, loc, scale) :
    return skewnorm.pdf(x, a=a, loc=loc, scale=scale)

def one_slope_histogram(data, weights, slopes, xlabel='$\log$|slope|', title='', width=width_def, height=height_def) :
    fig, axs = plt.subplots(1,1, figsize=(width,height))
    slope_range = np.linspace(np.min(np.log10(data))-1, np.max(np.log10(data))+1, 100)
    h_measured = axs.hist(np.log10(data), bins=slope_range, density=True, weights=weights, histtype='step', label='measured')
    h_expected = axs.hist(np.log10(abs(np.array(slopes))), bins=slope_range, density=True, histtype='step', fill=True, alpha=0.5, label='expected')

    # fit the histogram to a skewnorm distribution
    bounds = ((-50,-50,0),(50,50,100))
    p = curve_fit(skewnormfit, h_measured[1][:-1], h_measured[0], bounds=bounds)
    lower_upper = skewnorm.interval(0.68, p[0][0], loc=p[0][1], scale=p[0][2])
    axs.axvspan(lower_upper[0], lower_upper[1], alpha=0.3, color='C0')
    fit = skewnorm.pdf(slope_range, a=p[0][0], loc=p[0][1], scale=p[0][2])

    mean = skewnorm.mean(a=p[0][0], loc=p[0][1], scale=p[0][2])
    print('logsf = {}^(+{})_(-{})'.format(mean, lower_upper[1]-mean, mean-lower_upper[0]))

    axs.plot(slope_range, skewnorm.pdf(slope_range, a=p[0][0], loc=p[0][1], scale=p[0][2]), c='k', lw=1.5)
    axs.set_xlabel('$\log|s_F|$')
    axs.legend()
