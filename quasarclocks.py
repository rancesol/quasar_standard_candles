

import numpy as np
from data_parse import parse_data_macho, parse_data_kepler, parse_data_cosmograil, parse_data_sdss, clean_data, d_lum, K
#from data_parse_QSOS82 import parse_data, clean_data, d_lum, K
from center_data import center_data
from progress.bar import IncrementalBar
from time import sleep
from tqdm import tqdm


RF = 'qso'                  # choose the reference frame ('qso'=quasar, 'obs'=observer)
method = 'quadratics'       # finding method to use ('quadratics' or 'straights')
minww  = 40                 # choose minimum window size
dataset = 'macho'          # choose dataset ('kepler' or 'macho' or 'cosmograil' or 'sdss')


if  (RF != 'qso') & (RF != 'obs') :
    print('You have chosen an undefined reference frame.')
    exit()
if  (method != 'quadratics') & (method != 'straights') :
    print('You have chosen an undefined finding method.')
    exit()



## get flux and redshifts of quasars
if dataset == 'macho' :
    qlist, quasar = parse_data_macho(RF)
elif dataset == 'kepler' :
    qlist, quasar = parse_data_kepler(RF)
elif dataset == 'cosmograil' :
    qlist, quasar = parse_data_cosmograil(RF)
elif dataset == 'sdss' :
    qlist, quasar = parse_data_sdss(RF)
zs = [quasar[qlist[n]]['z'] for n in range(len(qlist))]
flux = [np.vstack((quasar[qlist[n]]['time'], quasar[qlist[n]]['flux'], quasar[qlist[n]]['fluxerr'])).T
        for n in range(len(qlist))]


## if you want to use only low z quasars
zcutoff = False
if zcutoff :
    mask = [zs[i] <= .2 for i in range(len(zs))]
    flux = [flux[i] for i in range(len(flux)) if mask[i]]
    qlist = [qlist[i] for i in range(len(qlist)) if mask[i]]
    zs = [zs[i] for i in range(len(zs)) if mask[i]]


## if you want to cutoff poorly sampled quasars
Ncutoff = True
if Ncutoff :
    mask = [len(flux[i][:,0]) >= 5e2 for i in range(len(flux))]
    flux = [flux[i] for i in range(len(flux)) if mask[i]]
    qlist = [qlist[i] for i in range(len(qlist)) if mask[i]]
    zs = [zs[i] for i in range(len(zs)) if mask[i]]


## if you want to cutoff quasars sampled for only short durations
Lcutoff = True
if Lcutoff :
    mask = [np.ptp(flux[i][:,0]) >= 2e2 for i in range(len(flux))]
    flux = [flux[i] for i in range(len(flux)) if mask[i]]
    qlist = [qlist[i] for i in range(len(qlist)) if mask[i]]
    zs = [zs[i] for i in range(len(zs)) if mask[i]]


## get rid of any faulty data points
flux = [clean_data(flux[n]) for n in range(len(flux))]
m = [-2.5*np.log10(np.mean(flux[n][:,1])) for n in range(len(flux))]
flux = [center_data(flux[n]) for n in range(len(flux))]


## get rid of any faulty light curves
if not np.any([flux[i].size > 2 for i in range(len(flux))]):
    mask = [flux[i].size > 2 for i in range(len(flux))]
    flux = [flux[i] for i in range(len(flux)) if mask[i]]
    qlist = [qlist[i] for i in range(len(qlist)) if mask[i]]
    zs = [zs[i] for i in range(len(zs)) if mask[i]]



## get rid of any quasars sampled for times shorter than minww (unlikely)
if not np.any([np.ptp(flux[i][:,0]) > minww for i in range(len(flux))]) :
    mask = [np.ptp(flux[i][:,0]) > minww for i in range(len(flux))]
    flux = [flux[i] for i in range(len(flux)) if mask[i]]
    qlist = [qlist[i] for i in range(len(qlist)) if mask[i]]
    zs = [zs[i] for i in range(len(zs)) if mask[i]]

print('zeff = {}'.format(np.mean(zs)))

## apply finding method ('quadratics' or 'straights') on all of the quasars
if method == 'quadratics' :
    from find_quadratics import find_quadratic
    if RF == 'qso' :
        window_width,window_midpoints,fit_params,chi2dof,best_fit=zip(*[find_quadratic(flux[n],minww)
            for n in tqdm(range(len(flux))) if (type(flux) is not None) and (type(flux[n]) is not None)])
    else :
        window_width,window_midpoints,fit_params,chi2dof,best_fit=zip(*[
            find_quadratic(flux[n],minww*(1+zs[n]))
            for n in tqdm(range(len(flux))) if (type(flux) is not None) and (type(flux[n]) is not None)])
elif method == 'straights' :
    from find_straights_v3 import find_straights
    if RF == 'qso' :
        window_width,fit_params,chi2dof,best_fit = zip(*[find_straights(flux[n],minww)
            for n in tqdm(range(len(flux))) if (type(flux) is not None) and (type(flux[n]) is not None)])
    else :
        window_width,fit_params,chi2dof,best_fit = zip(*[find_straights(flux[n],minww*(1+zs[n]))
            for n in tqdm(range(len(flux))) if (type(flux) is not None) and (type(flux[n]) is not None)])



## some quasars may not meet some of the criteria imposed in the finding method
## resulting in having no windows, so get rid of them
if not np.any([type(window_width[n]) is list for n in range(len(window_width))]) :
    mask = [type(window_width[n]) is list for n in range(len(window_width))]
    flux = [flux[n] for n in range(len(flux)) if mask[n]]
    qlist = [qlist[n] for n in range(len(qlist)) if mask[n]]
    window_width = [window_width[n] for n in range(len(window_width)) if mask[n]]
    if method == 'quadratics' :
        window_midpoints = [window_midpoints[n] for n in range(len(window_midpoints)) if mask[n]]
    fit_params = [fit_params[n] for n in range(len(fit_params)) if mask[n]]
    chi2dof = [chi2dof[n] for n in range(len(chi2dof)) if mask[n]]
    best_fit = [best_fit[n] for n in range(len(best_fit)) if mask[n]]
    zs = [zs[n] for n in range(len(zs)) if mask[n]]



print('N = ' + str(len(flux)))


## now we are going to plot some stuff
import matplotlib.pyplot as plt
cm = 1/2.54
plt.style.use(['science', 'no-latex'])
plt.rc('font', size=8)
plt.rcParams['font.family'] = 'monospace'
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import matplotlib.colors as mclolors
width = 2*8.6*cm
height = width/2.55


#index = np.where(np.asarray(qlist) == '9.5484.258')[0][0]
#index = np.where(np.asarray(qlist) == '8946433')[0][0]
#index = np.where(np.asarray([len(flux[n][:,0]) for n in range(len(flux))]) == max([len(flux[n][:,0]) for n in range(len(flux))]))[0][0]
index = np.where(np.asarray([len(window_midpoints[n]) for n in range(len(window_midpoints))]) == max([len(window_midpoints[n]) for n in range(len(window_midpoints))]))[0][0]
print(qlist[index])
NDX   = index



## plot the light curve for a single quasar
flux_centered = center_data(flux[index])
fig, axs = plt.subplots(1,1, figsize=(width,height))
#fig, axs = plt.subplots(1,1, figsize=(7.1*cm,7.1*cm))
axs.scatter(flux_centered[:,0], flux_centered[:,1], c='k', s=0.2)
markers, caps, bars = axs.errorbar(flux_centered[:,0], flux_centered[:,1],
        yerr=flux_centered[:,2], xerr=None, ls='none', c='k', capsize=2, elinewidth=0.7, markeredgewidth=1,
        label=qlist[index]+',  z = '+str(quasar[qlist[index]]['z']))
if method == 'quadratics' :
    [axs.plot(np.linspace(
        window_midpoints[index][i]-0.5*window_width[index][i],
        window_midpoints[index][i]+0.5*window_width[index][i], 5),
        [(2*fit_params[index][i,0]*window_midpoints[index][i] + fit_params[index][i,1])
            *(ti-window_midpoints[index][i]) + (fit_params[index][i,0]*window_midpoints[index][i]**2
                + fit_params[index][i,1]*window_midpoints[index][i] + fit_params[index][i,2])
            for ti in np.linspace(
                window_midpoints[index][i]-0.5*window_width[index][i],
                window_midpoints[index][i]+0.5*window_width[index][i],5)],
            c='k', linewidth=0.5, alpha=0.5)
        for i in range(len(window_midpoints[index]))]
else :
    [axs.plot(best_fit[index][i][:,0], best_fit[index][i][:,1], c='k', linewidth=0.5, alpha=0.5)
            for i in range(len(best_fit[index]))]
axs.set_xlabel('Days')
axs.set_ylabel('Flux')
[bar.set_alpha(0.05) for bar in bars]
[cap.set_alpha(0.05) for cap in caps]
plt.tight_layout(pad=0.1)



from scipy.optimize import curve_fit
from scipy.stats import norm, skewnorm
def skewnormfit(x, a, loc, scale) :
    return skewnorm.pdf(x, a=a, loc=loc, scale=scale)



## define the range of expected slopes and bounds for fitting the histogram
if RF == 'qso' :
    slope_range = np.linspace(0, 10, 100)
    bounds = ((-50,-50,0),(50,50,100))
elif RF == 'obs' :
    slope_range = np.linspace(-14, -9, 100)
    bounds = ((-50,-50,0),(50,50,100))



## make histogram of plots for one quasar
fps = fit_params[index]
order = np.argsort(chi2dof[index])[::-1]
sigma = 1
weights = 1/chi2dof[index]
if method == 'quadratics' :
    wmpts = window_midpoints[index]
    data = [abs(fps[i,1] + 2*fps[i,0]*wmpts[i]) for i in order]
else :
    data = [abs(fps[i,0]) for i in order]


fig, axs = plt.subplots(1,1, figsize=(width,height))
#fig, axs = plt.subplots(1,1, figsize=(7.1*cm,7.1*cm))
h = axs.hist([np.log10(d) for d in data], bins=slope_range, density=True, weights=weights, fill=False, histtype='step')


# fit the histogram to a skewnorm distribution
p = curve_fit(skewnormfit, h[1][:-1], h[0], bounds=bounds)
fit = skewnorm.pdf(slope_range, a=p[0][0], loc=p[0][1], scale=p[0][2])
print('logsf = ' + str(slope_range[np.where(fit==np.max(fit))[0]]))

axs.plot(slope_range, skewnorm.pdf(slope_range, a=p[0][0], loc=p[0][1], scale=p[0][2]), c='k', lw=1.5)
if RF == 'qso' :
    axs.set_xlabel('$\log|s_F|$')
else :
    axs.set_xlabel('$\log|s_f|$')
plt.tight_layout(pad=0.1)
plt.show()


## repeat the histogram fitting for each quasar
logsf = []
logsf_errs = []
ns = []
bar = IncrementalBar('Getting Peaks', max=len(zs))
fig, axs = plt.subplots(4,4, figsize=(2*8.6*cm,3*8.6*(height/width)*cm), sharex=True, sharey=True)
plt.tight_layout(pad=0.1)
for n in range(len(zs)) :
    order = np.argsort(chi2dof[n])[::-1]
    weights = 1/chi2dof[n]
    #weights = np.exp(-(chi2dof[n]-1)**2/(2*sigma**2))
    
    if method == 'quadratics' :
        data = [abs(fit_params[n][i,1]+2*fit_params[n][i,0]*window_midpoints[n][i])
                for i in order]
    else :
        data = [abs(fit_params[n][i,0]) for i in order]

    h = np.histogram([np.log10(d) for d in data], bins=(len(slope_range)-1),
            range=(np.min(slope_range), np.max(slope_range)), density=True, weights=weights)
    
    try :
        p = curve_fit(skewnormfit, h[1][:-1], h[0], bounds=bounds)
        fit = skewnorm.pdf(slope_range, a=p[0][0], loc=p[0][1], scale=p[0][2])
        peak = slope_range[np.where(fit==np.max(fit))[0]]
        if RF == 'qso' :
            logsf.append(peak - np.log10(1+zs[n]))
        else :
            logsf.append(peak)
        logsf_errs.append(skewnorm.std(a=p[0][0], loc=p[0][1], scale=p[0][2])/np.sqrt(len(data)))
        ns.append(n)
        if n<16 :
            axs[n//4,n-int((n//4)*4)].hist([np.log10(d) for d in data], bins=slope_range,
                    density=True, weights=weights, fill=False, histtype='step')
            axs[n//4,n-int((n//4)*4)].plot(slope_range, fit, c='k', lw=1.5)
            axs[n//4,n-int((n//4)*4)].text(min(slope_range)+0.05*np.ptp(slope_range),1.1,
                    qlist[n]+'\nz = '+str(quasar[qlist[n]]['z'])+'\nN = '+str(len(flux[n][:,0])))
            if n//4==3:
                if RF == 'qso' :
                    axs[n//4,n-int((n//4)*4)].set_xlabel('$\log|s_F|$')
                else :
                    axs[n//4,n-int((n//4)*4)].set_xlabel('$\log|s_f|$')
        bar.next()
    except :
        bar.next()
        continue
bar.finish()


## Some quasars may not have histograms that cannot reasonably be fitted with a skewnorm.
## Most likely bad statistics or the slopes don't fit in the expected bounds.
if len(ns) != len(zs) :
    m = [m[n] for n in ns]
    zs = [zs[n] for n in ns]


## Fit the s_F - <m> relation.
coeffs, cov = np.polyfit(m, logsf, 1, cov=True)
perrs = np.sqrt(np.diag(np.hstack(cov)))
fit = np.vstack((m, np.polyval(coeffs, m))).T



## Calculate the residuals of the relation.
residuals = [logsf[i] - np.polyval(coeffs, m[i]) for i in range(len(logsf))]



#/////////////////////////////////////////A\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V/////////////////////////////////////////

from scipy.optimize import minimize

def MLE(B,x,y,dy) :
    yPred = B[0]*x + B[1]
    s2 = dy**2 + B[2]**2
    negLL = -np.sum(-(y-yPred)**2/2/s2 - np.log(np.sqrt(2*np.pi*s2)))
    return negLL

results = minimize(MLE, [0,-0.5, 0.5], args=(np.array(m), np.array(logsf), np.array(logsf_errs)), method='Nelder-Mead', bounds=((-1,1),(-5,5),(0,1)))
print(results.x)


#/////////////////////////////////////////A\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\V/////////////////////////////////////////



#fig, axs = plt.subplots(2,1, figsize=(8.6*cm, 10*cm), gridspec_kw={'height_ratios': [4,1]}, sharex=True)
#fig, axs = plt.subplots(2,1, figsize=(width, 10*cm), gridspec_kw={'height_ratios': [4,1]}, sharex=True)
fig, axs = plt.subplots(2,1, figsize=(0.8*width, 0.8*width/1.66), gridspec_kw={'height_ratios': [4,1]}, sharex=True)
plt.subplots_adjust(wspace=0, hspace=0)
a = -0.2878#-0.29263324#-0.29353524#coeffs[0]
b = -0.3053#-0.39721584#-0.4315298#coeffs[1]
axs[0].scatter(m, logsf, s=5, c='k')
axs[0].scatter([m[i] for i in range(len(m)) if (len(flux[i][:,0])>=5e2)],
        [logsf[i] for i in range(len(logsf)) if (len(flux[i][:,0])>=5e2)], s=5, c='r')
markers, caps, bars = axs[0].errorbar(m, logsf,
        yerr=logsf_errs, xerr=None, linestyle='none', c='k', capsize=1, elinewidth=0.7, markeredgewidth=1)
[bar.set_alpha(0.6) for bar in bars]
[cap.set_alpha(0.6) for cap in caps]
axs[0].plot(fit[:,0],fit[:,1], linewidth=1, c='k')
axs[0].set_xlim(np.min(fit[:,0])-1, np.max(fit[:,0])+1)
if RF == 'qso' :
    axs[0].set_ylabel(r'$\log|s_F| - \log(1+z)$')
else :
    axs[0].set_ylabel(r'$\log|s_f|$')
axs[0].plot(fit[:,0], [a*fit[i,0]+b for i in range(len(fit[:,0]))], lw=1, c='b')


axs[1].scatter(m, residuals, s=5, c='k')
axs[1].scatter([m[i] for i in range(len(m)) if (len(flux[i][:,0])>=5e2)],
        [residuals[i] for i in range(len(residuals)) if (len(flux[i][:,0])>=5e2)], s=5, c='r')
axs[1].set_ylim(-0.55, 0.55)
axs[1].axhline(y=0, c='k', lw='1')
axs[1].set_xlim(max(m)+0.1*np.ptp(m), min(m)-0.1*np.ptp(m))
axs[1].set_ylabel(r'Res.')
if RF == 'qso' :
    axs[1].set_xlabel(r'$\langle M \rangle$')
else :
    axs[1].set_xlabel(r'$\langle m \rangle$')
plt.tight_layout(pad=0.1)

plt.show()



## Calculate the dispersion over the relation.
dispersion = np.sqrt(sum([(fit[i,1] - logsf[i])**2 for i in range(len(logsf))])/(len(logsf)-1))
print('Dispersion = ' + str(dispersion))


#fig, axs = plt.subplots(1,1, figsize=(1.5*8.6*cm, 4.3*cm))
fig, axs = plt.subplots(1,1, figsize=(width, height))
axs.axhline(y=0, lw=1, c='k')
axs.scatter(zs, [logsf[i]-coeffs[0]*m[i]-coeffs[1] for i in range(len(logsf))], s=1, c='k')
axs.scatter([zs[i] for i in range(len(zs)) if (len(flux[i][:,0])>=5e2)],
        [logsf[i]-coeffs[0]*m[i]-coeffs[1] for i in range(len(logsf)) if (len(flux[i][:,0])>=5e2)], s=5, c='r')
axs.set_xlabel(r'z')
axs.set_ylabel(r'Residuals')
plt.tight_layout(pad=0.1)
plt.show()



print('m = ' + str(coeffs[0]) + '+-' + str(perrs[0]))
print('b = ' + str(coeffs[1]) + '+-' + str(perrs[1]))

from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(m, np.hstack(logsf))      # for V-band
print('inter = ' + str(intercept))
print('R2    = ' + str(r_value**2))
print('P     = ' + str(p_value))
print('std_err = ' + str(std_err))



## Calculate DM = 5(log(dl) - 1) and its uncertainties.
## Only useful when RF == 'obs'.
a = -0.275#coeffs[0]
b =  0.013#coeffs[1]
da = 0.0111#perrs[0]
db = 0.2709#perrs[1]
DM_QSO = np.hstack([5*(a*m[i] + b - logsf[i])/(2+5*a) - K(zs[i])
    for i in range(len(m))])
dDM_QSO = np.hstack([5*(2+5*a)**(-1)*np.sqrt((da*(m[i]-5*(a*m[i]+b-logsf[i])/(2+5*a)))**2
    + db**2 + logsf_errs[i]**2) for i in range(len(m))])


def DMzfit(z, k, a2, a3, a4, a5, a0) :
    return 5*(np.log10(k*(np.log10(1+z) + a2*np.log10(1+z)**2 + a3*np.log10(1+z)**3 + a4*np.log10(1+z)**4 + a5*np.log10(1+z)**5))-1) + a0



order_QSO = np.argsort(zs)[::-1]
p = curve_fit(DMzfit, [zs[n] for n in order_QSO], [DM_QSO[n] for n in order_QSO],
        sigma=[dDM_QSO[n] for n in order_QSO])
fit = [DMzfit(zs[n], p[0][0], p[0][1], p[0][2], p[0][3], p[0][4], p[0][5]) for n in order_QSO]



from data_parse_SNe import parse_SNe
zs_SNe, DM_SNe, dDM_SNe = parse_SNe()
order_SNe = np.argsort(zs_SNe)[::-1]

p_SNe = curve_fit(DMzfit, [zs_SNe[n] for n in order_SNe], [DM_SNe[n] for n in order_SNe],
        sigma=[dDM_SNe[n] for n in order_SNe])
fit_SNe = [DMzfit(zs_SNe[n], p_SNe[0][0], p_SNe[0][1], p_SNe[0][2], p_SNe[0][3], p_SNe[0][4], p_SNe[0][5])
        for n in order_SNe]

d = 0.5
bounds = ((p_SNe[0][0]-d*abs(p_SNe[0][0]),p_SNe[0][1]-d*abs(p_SNe[0][1]),p_SNe[0][2]-d*abs(p_SNe[0][2]),
    p_SNe[0][3]-d*abs(p_SNe[0][3]),p_SNe[0][4]-d*abs(p_SNe[0][4]),p_SNe[0][5]-d*abs(p_SNe[0][5])),
    (p_SNe[0][0]+d*abs(p_SNe[0][0]),p_SNe[0][1]+d*abs(p_SNe[0][1]),p_SNe[0][2]+d*abs(p_SNe[0][2]),
        p_SNe[0][3]+d*abs(p_SNe[0][3]),p_SNe[0][4]+d*abs(p_SNe[0][4]),p_SNe[0][5]+d*abs(p_SNe[0][5])))

#z_bins = np.linspace(min(zs)-0.1,max(zs)+0.1,10, endpoint=True)
z_bins = np.geomspace(min(zs)-0.01,max(zs)+0.01,10, endpoint=True)
zs_binned = [np.mean([zs[j] for j in order_QSO if (zs[j]>=z_bins[i-1]) & (zs[j]<z_bins[i])])
        for i in range(1,len(z_bins))]
DM_binned = [np.mean([DM_QSO[j] for j in order_QSO if (zs[j]>=z_bins[i-1]) & (zs[j]<z_bins[i])])
        for i in range(1,len(z_bins))]
dDM_binned = [np.sqrt(sum([dDM_QSO[j]**2 for j in range(len(DM_QSO))
    if (zs[j]>=z_bins[i-1]) & (zs[j]<z_bins[i])]))/sum(map(lambda x : (x>=z_bins[i-1]) & (x<z_bins[i]), zs))
    for i in range(1,len(z_bins))]


order_QSO_binned = np.argsort(zs_binned)[::-1]
#try :
#    p = curve_fit(DMzfit,[zs_binned[n] for n in order_QSO_binned],[DM_binned[n] for n in order_QSO_binned],
#            sigma=[dDM_binned[n] for n in order_QSO_binned], bounds=bounds)
#    fit = [DMzfit(zs_binned[n], p[0][0], p[0][1], p[0][2], p[0][3], p[0][4], p[0][5])
#            for n in order_QSO_binned]
#except ValueError :
#    print('Could not fit to binned data.')

zs_tot  = np.hstack((zs_binned, zs_SNe))
DM_tot  = np.hstack((DM_binned, DM_SNe))
dDM_tot = np.hstack((dDM_binned, dDM_SNe))
order_tot = np.argsort(zs_tot)[::-1]

#p_tot = curve_fit(DMzfit, [zs_tot[n] for n in order_tot], [DM_tot[n] for n in order_tot],
#        sigma=[dDM_tot[n] for n in order_tot],
#        bounds=bounds)
#fit_tot = [DMzfit(zs_tot[n], p_tot[0][0], p_tot[0][1], p_tot[0][2], p_tot[0][3], p_tot[0][4], p_tot[0][5])
#        for n in order_tot]



import os, os.path
cwd = os.getcwd()
with open(cwd + '/../qso_redshift_evolution/DL_all_short.txt') as datafile :
    data = np.loadtxt(datafile, skiprows=1)
    zs_RL  = np.array(data[::1,0])
    mu_RL  = np.array(data[::1,1]) #+ 17.5 - 5*np.log10(70) + 5
    dmu_RL = np.array(data[::1,2])



#fig, axs = plt.subplots(1,1, figsize=(1.5*8.6*cm,5.73*cm))
fig, axs = plt.subplots(1,1, figsize=(width,height))
#axs.scatter(zs_RL, mu_RL, s=2, c='y')
#markers1, caps1, bars1 =axs.errorbar(zs_RL, mu_RL, xerr=None, yerr=dmu_RL,
#        linestyle='none', c='y', capsize=1, elinewidth=0.7, markeredgewidth=1)
axs.scatter(zs_SNe, DM_SNe, s=1.5, c='b')
axs.errorbar(zs_SNe, DM_SNe, xerr=None, yerr=dDM_SNe, ls='none', c='b', capsize=1, elinewidth=0.7, markeredgewidth=1)
axs.scatter(zs, DM_QSO, s=2, c='k', zorder=32)
markers, caps, bars =axs.errorbar(zs, DM_QSO, xerr=None, yerr=dDM_QSO,
        linestyle='none', c='k', capsize=1, elinewidth=0.7, markeredgewidth=1, zorder=32)
axs.scatter(zs_binned, DM_binned, s=1, c='r', zorder=33)
try :
    axs.plot([zs_binned[n] for n in order_QSO_binned], fit, lw=1, c='r')
except ValueError :
    print('')
axs.errorbar(zs_binned, DM_binned, xerr=None, yerr=dDM_binned,
        ls='none', c='r', capsize=1, elinewidth=0.7, markeredgewidth=1, zorder=33)
#[bar.set_alpha(0.3) for bar in bars]
#[cap.set_alpha(0.3) for cap in caps]
#axs.plot([zs_tot[n] for n in order_tot], fit_tot, lw=1, c='k')
axs.plot(np.linspace(0,4.2,100), [5*(np.log10(d_lum(z)) - 1) for z in np.linspace(0,4.2,100)],
        lw=0.8, ls='--', c='k')
axs.set_xlabel(r'z')
axs.set_ylabel(r'Distance Modulus')
axs.set_xlim(-0.1, 4.2)
if RF == 'qso' :
    axs.set_ylim(-10,10)
else :
    axs.set_ylim(32.5, 55)
plt.tight_layout(pad=0.1)
plt.show()




import scipy.integrate as integrate

def E(z, OmM) :
    OmR = 9e-5
    OmL = 1 - OmM
    return np.sqrt(OmR*(1+z)**4 + OmM*(1+z)**3 + OmL)

def D_lum_constH(zs, h) :
    dH = 3e9/h
    return np.array([(1+z)*dH*integrate.quad(lambda x: 1, 0, z)[0] for z in zs])

def D_lum(zs, h, OmM) :
    dH = 3e9/h
    return np.array([(1+z)*dH*integrate.quad(lambda x: 1/E(x, OmM), 0, z)[0] for z in zs])






DL_QSO = 10**(np.array(DM_QSO)/5+1)
DL_SNe = 10**(np.array(DM_SNe)/5+1)
DL_RL  = 10**(np.array(mu_RL)/5+1)
#dDL_QSO = np.log(10)*DL_QSO*np.array(dDM_QSO)/5
#dDL_SNe = np.log(10)*DL_SNe*np.array(dDM_SNe)/5

fig, axs = plt.subplots(1,1, figsize=(width, height))
axs.scatter(zs_RL, DL_RL,   s=0.7, c='y', label='RL QSOs')
axs.scatter(zs_SNe, DL_SNe, s=0.7, c='b', label='SNe')
axs.scatter(zs, DL_QSO, s=10, label='SS QSOs', marker='o', facecolors='none', edgecolors='k')
axs.plot(zs_SNe[np.argsort(zs_SNe)[::1]], D_lum(zs_SNe[np.argsort(zs_SNe)[::1]],0.7,0.321), lw=1, c='b', label='$H(z)$')
axs.fill_between(zs_SNe[np.argsort(zs_SNe)[::1]], D_lum(zs_SNe[np.argsort(zs_SNe)[::1]],0.7,0), D_lum(zs_SNe[np.argsort(zs_SNe)[::1]],0.7,1), alpha=0.3, color='b', ec='None')
axs.plot(zs_SNe[np.argsort(zs_SNe)[::1]], D_lum_constH(zs_SNe[np.argsort(zs_SNe)[::1]], 0.7), lw=1, c='r', label='$H_0$')
axs.set_xscale('log')
axs.set_yscale('log')
axs.set_xlim(min(zs_SNe)/2, 7.5)
axs.set_ylim(d_lum(min(zs_SNe))/2, d_lum(7.5))
axs.set_xlabel('$z$')
axs.set_ylabel('$D_L (pc)$')
axs.legend()
plt.tight_layout(pad=0.1)
plt.show()
