
import numpy as np
from astroML.time_series import generate_damped_RW as gen_drw
from center_data import center_data
from find_quadratics import find_quadratic
from data_parse import K
from tqdm import tqdm
import matplotlib.pyplot as plt
cm = 1/2.54
plt.style.use(['science', 'no-latex'])
plt.rc('font', size=8)
plt.rcParams['font.family'] = 'monospace'
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import matplotlib.colors as mclolors
width = 1.5*8.6*cm
height = width/2.55



RF = 'qso'
method = 'quadratics'
minww = 40

N = 50
M = [-2.5*(9 + 3.2*np.random.random_sample()) for _ in range(N)]
zs = [0*7*np.random.random_sample() for _ in range(N)]

tau = [10**(2.4 + 0.03*(M[i]+23)) for i in range(N)]
SFinf = [10**(-0.51 + 0.13*(M[i]+23)) for i in range(N)]
t = [np.linspace(0,3*365, 1000) for i in range(N)]
flux = [np.vstack((t[i], (1+gen_drw(t[i],tau=tau[i],z=0,SFinf=SFinf[i]))*10**(-M[i]/2.5),
    [1 for _ in range(len(t[i]))])).T for i in tqdm(range(N), desc='Light Curves')]

M = [-2.5*np.log10(np.mean(flux[n])) for n in range(N)]

flux = [center_data(flux[i]) for i in range(N)]


window_width,window_midpoints,fit_params,chi2dof,best_fit = zip(*[find_quadratic(flux[n],minww)
    for n in tqdm(range(len(flux)), desc='Finding')
    if (type(flux) is not None) and (type(flux[n]) is not None)])

index = 0
## plot the light curve for a single quasar
flux_centered = center_data(flux[index])
fig, axs = plt.subplots(1,1, figsize=(width, height))
axs.scatter(flux_centered[:,0], flux_centered[:,1], c='k', s=0.2)
markers, caps, bars = axs.errorbar(flux_centered[:,0], flux_centered[:,1],
        yerr=flux_centered[:,2], xerr=None, ls='none', c='k', capsize=2, elinewidth=0.7, markeredgewidth=1)
if method == 'quadratics' :
    [axs.plot(np.linspace(window_midpoints[index][i]-0.5*window_width[index][i], window_midpoints[index][i]+0.5*window_width[index][i], 5),
        [(2*fit_params[index][i,0]*window_midpoints[index][i] + fit_params[index][i,1])*(ti-window_midpoints[index][i]) + (fit_params[index][i,0]*window_midpoints[index][i]**2 + fit_params[index][i,1]*window_midpoints[index][i] + fit_params[index][i,2]) for ti in np.linspace(window_midpoints[index][i]-0.5*window_width[index][i], window_midpoints[index][i]+0.5*window_width[index][i],5)], c='k', linewidth=0.5, alpha=0.5)
        for i in tqdm(range(len(window_midpoints[index])), desc='Plotting lc')]
else :
    [axs.plot(best_fit[index][i][:,0], best_fit[index][i][:,1], c='k', linewidth=0.5, alpha=0.5)
            for i in tqdm(range(len(best_fit[index])), desc='Plotting lc')]
axs.set_xlabel('Days')
axs.set_ylabel('Flux')
[bar.set_alpha(0.1) for bar in bars]
[cap.set_alpha(0.1) for cap in caps]
plt.tight_layout(pad=0.1)


from scipy.optimize import curve_fit
from scipy.stats import norm, skewnorm
def skewnormfit(x, a, loc, scale) :
    return skewnorm.pdf(x, a=a, loc=loc, scale=scale)

slope_range = np.linspace(3, 10, 100)       # F and s_F and t_qso
bounds = ((-50,-50,0),(50,50,100))


## make histogram of plots for one quasar
fps = fit_params[index]
order = np.argsort(chi2dof[index])[::-1]
weights = np.exp(-(chi2dof[index]-1)**2/(2*1**2))
if method == 'quadratics' :
    wmpts = window_midpoints[index]
    data = [abs(fps[i,1] + 2*fps[i,0]*wmpts[i]) for i in order]
else :
    data = [abs(fps[i,0]) for i in order]

fig, axs = plt.subplots(1,1, figsize=(width, height))
h = axs.hist([np.log10(d) for d in data], bins=slope_range, density=True)#, weights=weights)#, fill=False)

## fit the histogram to a skewnorm distribution
p = curve_fit(skewnormfit, h[1][:-1], h[0], bounds=bounds)
fit = skewnorm.pdf(slope_range, a=p[0][0], loc=p[0][1], scale=p[0][2])
print('logsf = ' + str(slope_range[np.where(fit==np.max(fit))[0]]))

axs.plot(slope_range, skewnorm.pdf(slope_range, a=p[0][0], loc=p[0][1], scale=p[0][2]), c='k', lw=1.5)
axs.set_xlabel('$\log|s_F|$')
plt.show()

## repeat the histogram fitting for each quasar
logsf = []
logsf_errs = []
ns = []
fig, axs = plt.subplots(4,4, figsize=(2*8.5*cm, 2*8.6*cm), sharex=True, sharey=True)
plt.tight_layout(pad=0.1)
for index in tqdm(range(len(zs)), desc='Getting peaks') :
    order = np.argsort(chi2dof[index])[::-1]
    weights = np.exp(-(chi2dof[index]-1)**2/(2*1**2))

    if method == 'quadratics' :
        data = [abs(fit_params[index][i,1]+2*fit_params[index][i,0]*window_midpoints[index][i])
                for i in order]
    else :
        data = [abs(fit_params[index][i,0]) for i in order]

    h = np.histogram([np.log10(d) for d in data], bins=(len(slope_range)-1),
            range=(np.min(slope_range), np.max(slope_range)), density=True)#, weights=weights)

    try :
        p = curve_fit(skewnormfit, h[1][:-1], h[0], bounds=bounds)
        fit = skewnorm.pdf(slope_range, a=p[0][0], loc=p[0][1], scale=p[0][2])
        peak = slope_range[np.where(fit==np.max(fit))[0]]
        if RF == 'qso' :
            logsf.append(peak - np.log10(1+zs[index]))
        else :
            logsf.append(peak)
        logsf_errs.append(skewnorm.std(a=p[0][0], loc=p[0][1], scale=p[0][2])/np.sqrt(len(data)))
        ns.append(index)
        #plt.plot(slope_range, fit)
        if index<16 :
            axs[index//4,index-int((index//4)*4)].hist([np.log10(d) for d in data], bins=slope_range,
                    density=True, fill=False, histtype='step')
            axs[index//4,index-int((index//4)*4)].plot(slope_range, fit, c='k', lw=1.5)
            if index//4==3 :
                axs[index//4,index-int((index//4)*4)].set_xlabel('$\log|s_F|$')
    except :
        continue

## Some quasars may not have histograms that can reasonably be fitted with a skewnorm.
## Most likely bad statistics or the slopes don't fit in the expected bounds.
if len(ns) != len(zs) :
    M = [M[n] for n in ns]
    zs = [zs[n] for n in ns]

## Fit the s_F - <F> relation.
coeffs, cov = np.polyfit(M, logsf, 1, cov=True)
perrs = np.sqrt(np.diag(np.hstack(cov)))
fit = np.vstack((M, np.polyval(coeffs, M))).T



## Calculate the residuals of the relation.
residuals = [logsf[i] - np.polyval(coeffs, M[i]) for i in range(len(logsf))]



fig, axs = plt.subplots(2,1, figsize=(0.8*width, 0.8*width/1.66), gridspec_kw={'height_ratios': [4,1]}, sharex=True)
plt.subplots_adjust(wspace=0, hspace=0)
axs[0].scatter(M, logsf, s=2, c='k')
markers, caps, bars = axs[0].errorbar(M, logsf,
        yerr=logsf_errs, xerr=None, linestyle='none', c='k', capsize=1, elinewidth=0.7, markeredgewidth=1)
[bar.set_alpha(0.6) for bar in bars]
[cap.set_alpha(0.6) for cap in caps]
axs[0].plot(fit[:,0],fit[:,1], linewidth=1, c='k')
axs[0].set_xlim(np.min(fit[:,0])-1, np.max(fit[:,0])+1)
if RF == 'qso' :
    axs[0].set_ylabel(r'$\log|s_F|$')
else :
    axs[0].set_ylabel(r'$\log|s_f|$')

axs[1].scatter(M, residuals, s=2, c='k')
axs[1].set_xlim(max(M)+0.1*np.ptp(M), min(M)-0.1*np.ptp(M))
axs[1].set_ylim(-0.55, 0.55)
axs[1].axhline(y=0, c='k', lw='1')
axs[1].set_ylabel(r'Residuals')
if RF == 'qso' :
    axs[1].set_xlabel(r'$\langle M \rangle$')
else :
    axs[1].set_xlabel(r'$\langle m \rangle$')
plt.tight_layout(pad=0.1)

plt.show()



## Calculate the dispersion over the relation.
dispersion = np.sqrt(sum([(fit[i,1] - logsf[i])**2 for i in range(len(logsf))])/(len(logsf)-1))
print('Dispersion = ' + str(dispersion))


#fig, axs = plt.subplots(1,1, figsize=(8.6*cm, 4.3*cm))
#axs.scatter(zs, [logsf[i]-coeffs[0]*logF[i]-coeffs[1] for i in range(len(logsf))], s=1, c='k')
#axs.axhline(y=0, lw=1, c='k')
#axs.set_xlabel(r'z')
#axs.set_ylabel(r'Residuals')
#plt.tight_layout()
#plt.show()


print('m = ' + str(coeffs[0]) + '+-' + str(perrs[0]))
print('b = ' + str(coeffs[1]) + '+-' + str(perrs[1]))

from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(M, np.hstack(logsf))      # for V-band
print('inter = ' + str(intercept))
print('R2    = ' + str(r_value**2))
print('P     = ' + str(p_value))
print('std_err = ' + str(std_err))
