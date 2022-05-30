
import numpy as np
from find_quadratics import find_quadratic
from scipy.optimize import curve_fit
from center_data import center_data
import numpy.polynomial.polynomial as poly


mu = 1e-4#6.47e-11
sigma = 5e-5

def curve_gen() :
    T = 0#1e-7*np.random.exponential(scale=0.1)
    n_segm = np.random.randint(70, 80)
    lengths = [int(10**(np.random.randint(0,2) + np.random.rand())) + 10 for i in range(n_segm)]
    ndata = [int(2*lengths[i] / np.random.randint(1,lengths[i] - 2)) for i in range(n_segm)]
    
    #xs = [np.random.choice(np.linspace(0, lengths[i], lengths[i]), ndata[i]) for i in range(n_segm)]
    xs = [np.linspace(0, lengths[i], lengths[i]) for i in range(n_segm)]
    x = np.sort(np.hstack([xs[i] + np.sum([np.max(xs[j])+5 for j in range(i)]) if i > 0 else xs[i]
        for i in range(n_segm)]))
    
    noise = [T*(2*np.random.rand(1,len(xs[i])) - 1) for i in range(n_segm)]
    amp = -(28 + 2*(2*np.random.random_sample() - 1))
    #m = [1e-3*np.random.choice([-1,1]) * (1 + 1e0*np.random.random_sample()) for i in range(n_segm)]
    m = [np.random.choice([-1,1])*np.random.normal(mu, sigma) for i in range(n_segm)]
    mag = np.hstack([m[i]*xs[i] + noise[i] + amp + np.sum([m[j]*np.max(xs[j])
        for j in range(i)]) if i > 0 else m[i]*xs[i] + noise[i] + amp for i in range(n_segm)])
    magerr = 3e-9*np.random.exponential(scale=0.5, size=np.size(x))

    #mean = np.mean(mag)
    #mag = mag / mean
    #magerr = magerr / mean
    
    #flux = 10**(-mag/2.5)
    #fluxerr = magerr/2.5*np.log(10)*10**(-mag/2.5)
    
    #return np.vstack((x, flux, fluxerr)).T, m
    return np.vstack((x, mag, magerr)).T, m


def func(x, a, b, c) :     # fitting function for curve_fit
    return a*x**2 + b*x + c


#def find_quadratic(a, minww) :
#    overlap = False
#    if overlap == True :
#        windows = [[a[(a[:,0] >= np.min(a[:,0]) + 0.5*i*np.ptp(a[:,0])/(j+1))
#            & (a[:,0] <= np.max(a[:,0]) - (j-0.5*i)*np.ptp(a[:,0])/(j+1))]
#            for i in range(2*(j+1)-1)]
#            for j in range(int(np.ptp(a[:,0])/minww))]
#    else :
#        windows = [[a[(a[:,0] >= np.min(a[:,0]) + i*np.ptp(a[:,0])/(j+1))
#            & (a[:,0] <= np.max(a[:,0]) - (j-i)*np.ptp(a[:,0])/(j+1))] for i in range(j+1)]
#            for j in range(int(2*np.ptp(a[:,0])/minww))]
#
#    mask = [[(len(windows[j][i]) > 10) for i in range(len(windows[j]))] for j in range(len(windows))]
#    windows = [[windows[j][i] for i in range(len(windows[j])) if mask[j][i]] for j in range(len(windows))]
#
#    mask = [np.size(w) > 2 for w in windows[:]]
#    windows = [windows[j] for j in range(len(windows)) if mask[j]]
#
#
#    mask = [[(len(windows[j][i]) / np.ptp(windows[j][i][:,0]) > 1/10) for i in range(len(windows[j]))]
#            for j in range(len(windows))]
#    windows = [[windows[j][i] for i in range(len(windows[j])) if mask[j][i]] for j in range(len(windows))]
#
#    mask = [(len(windows[j]) > 2) for j in range(len(windows))]
#    windows = [windows[j] for j in range(len(windows)) if mask[j]]
#
#
#    if len(windows) > 0 :
#        fit_params, _ = zip(*[zip(*[curve_fit(func, windows[j][i][:,0], windows[j][i][:,1],
#            sigma=windows[j][i][:,2])
#            for i in range(len(windows[j]))])
#            for j in range(len(windows)) if (len(windows[j]) > 2)])
#
#
#        chi2dof = [[np.sum(((windows[j][i][:,1] - func(windows[j][i][:,0], fit_params[j][i][0],
#            fit_params[j][i][1], fit_params[j][i][2]))/windows[j][i][:,2])**2/(len(windows[j][i])-1))
#            for i in range(len(windows[j])) if (len(windows[j]) > 2)]
#            for j in range(len(windows)) if (np.size(windows[j]) > 2)]
#
#
#        windows = [window for sublist in windows for window in sublist]
#        window_width = [np.ptp(windows[i][:,0]) for i in range(len(windows))]
#        window_midpoints = [np.median(windows[i][:,0]) for i in range(len(windows))]
#        fit_params = np.vstack([fit_params[i] for i in range(len(fit_params))])
#        chi2dof   = np.hstack(chi2dof)
#
#
#
#        N = 20
#        min_i = np.argsort(chi2dof)
#        best_fit = [np.vstack((windows[i][:,0], fit_params[i,0]*windows[i][:,0]**2 + fit_params[i,1]*windows[i][:,0] + fit_params[i,2])).T
#                for i in min_i[:]]
#
#        return window_width, window_midpoints, fit_params, chi2dof, best_fit
#
#    else :
#        return 0, 0, 0, 0, 0


#def norm_list(lst) :
#    if np.max(lst) > 0 :
#        return lst/np.max(lst)
#    else :
#        return lst
#
#a, slopes = zip(*[curve_gen() for i in range(20)])
#mags = [np.mean(a[n][:,1]) for n in range(len(a))]
#a = [center_data(a[n]) for n in range(len(a))]
#
#minww = 50
#window_width, window_midpoints, fit_params, chi2dof, best_fit = zip(*[find_quadratic(a[n], minww)
#    for n in range(len(a))])
#
#
#import matplotlib.pyplot as plt
#cm = 1/2.54
#plt.style.use(['science','no-latex'])
#import matplotlib.gridspec as gridspec
#from matplotlib.colors import LogNorm
#import matplotlib.colors as mcolors
#
#index = 0
#
#
### to plot the light curve of a single quasar with the fits
#plt.rc('font', size=8)
#fig, axs = plt.subplots(1,1, figsize=(8.6*cm,7*cm))
#axs.errorbar(a[index][:,0], a[index][:,1], yerr=a[index][:,2], xerr=None, ls='none', c='C3')
#[axs.plot(best_fit[index][i][:,0], best_fit[index][i][:,1], c='k', linewidth=0.5, alpha=0.3)
#        for i in range(len(best_fit[index]))]
#axs.set_xlabel('Days')
#axs.set_ylabel('Flux')
#axs.set_title('Synthetic Linear Segments')
#plt.tight_layout()
##plt.show()
#
#
#from scipy.optimize import curve_fit
#from scipy.stats import norm, skewnorm
#def skewnormfit(x, a, loc, scale) :
#    return skewnorm.pdf(x, a=a, loc=loc, scale=scale)
#
#
##slope_range = np.linspace(5, 9, 500)
#slope_range = np.linspace(-15, 0, 500)
#
#fps = np.vstack([fit_params[i] for i in range(len(fit_params))])
#wmpts = np.hstack(window_midpoints)
#order = np.argsort(np.hstack(chi2dof))[::-1]
#weights = np.exp(-(np.hstack(chi2dof)-1)**2/(2*0.5**2))
#data = [abs(fps[i,1] + 2*fps[i,0]*wmpts[i]) for i in order]
#
#
#fig, axs = plt.subplots(1,1, figsize=(8.6*cm,6*cm))
#h = axs.hist(np.log10(data), bins=(len(slope_range)-1), density=True)#, weights=weights)
#
#hfitparams = curve_fit(skewnormfit, h[1][:-1], h[0], bounds=((-np.inf,-np.inf,0),(0,0,10)))
#fit = skewnorm.pdf(slope_range, a=hfitparams[0][0], loc=hfitparams[0][1], scale=hfitparams[0][2])
#peak = slope_range[np.where(fit==np.max(fit))[0]]
#print('peak = ' + str(peak))
#
#axs.plot(slope_range, skewnorm.pdf(slope_range, a=hfitparams[0][0], loc=hfitparams[0][1], scale=hfitparams[0][2]), c='k', linestyle='--', linewidth=0.8)
#axs.set_ylim(bottom=0, top=1.2*np.max(h[0]))
#axs.set_xlabel('$\log_{10}|s|$')
#plt.tight_layout()
#plt.show()
#
#peaks = []
#peak_errs = []
#ns = []
#for index in range(len(a)) :
#    order = np.argsort(chi2dof[index])[::-1]
#    weights = np.exp(-(chi2dof[index]-1)**2/(2*0.5**2))
#
#    data = [abs(fit_params[index][i,1]+2*fit_params[index][i,0]*window_midpoints[index][i]) for i in order]
#    #data = [abs(fit_params[index][i,0]) for i in order]
#
#    h = np.histogram([np.log10(d) for d in data], bins=(len(slope_range)-1),
#            range=(np.min(slope_range), np.max(slope_range)), density=True)#, weights=weights)
#
#    try :
#        hfitparams = curve_fit(skewnormfit, h[1][:-1], h[0], bounds=((-np.inf,-np.inf,0),(0,0,10)))
#        fit = skewnorm.pdf(slope_range, a=hfitparams[0][0], loc=hfitparams[0][1], scale=hfitparams[0][2])
#        peak = slope_range[np.where(fit==np.max(fit))[0]]
#        peaks.append(peak)
#        peak_errs.append(hfitparams[0][2])
#        ns.append(index)
#        plt.plot(slope_range, fit)
#    except :
#        continue
#
#
#mags = [mags[n] for n in ns]
##coeffs, res = poly.polyfit(mags, np.hstack(peaks), 1, full=True)
##fit = np.vstack((mags, poly.polyval(mags, coeffs))).T
#fig, axs = plt.subplots(1,1, figsize=(8.6*cm, 8.6*cm))
#axs.scatter(mags, peaks, s=0.9, c='k')
#axs.errorbar(mags, peaks, yerr=peak_errs, xerr=None, linestyle='none', c='k', capsize=5, elinewidth=0.7, markeredgewidth=1)
##axs.plot(fit[:,0],fit[:,1], linewidth=1, c='k')
##axs.set_xlim(np.min(fit[:,0])-1, np.max(fit[:,0])+1)
#axs.set_xlabel('Absolute Magnitude (V)')
#axs.set_ylabel('$\log_{10}|s|$ (V)')
#plt.tight_layout()
#plt.show()
#
#
#
#print('m = ' + str(coeffs[1]))
#print('b = ' + str(coeffs[0]))
#print('res = ' + str(res))
#
#from scipy import stats
#slope, intercept, r_value, p_value, std_err = stats.linregress(mags, np.hstack(peaks))      # for V-band
#print('slope = ' + str(slope))
#print('inter = ' + str(intercept))
#print('R2    = ' + str(r_value**2))
#print('P     = ' + str(p_value))
#print('std_err = ' + str(std_err))


#fig = plt.figure(tight_layout=True)
#gs = gridspec.GridSpec(2,2)
#
#ax = fig.add_subplot(gs[0,:])
#ax.errorbar(a[index][:,0], a[index][:,1], yerr=a[index][:,2], xerr=None, ls='none', c='C0')
#[ax.plot(best_fit[index][i][:,0], best_fit[index][i][:,1], c='k', linewidth=0.5)
#        for i in range(len(best_fit[index]))]
#ax.set_xlabel('Days')
#ax.set_ylabel('Flux')
#ax.set_title('Mock Data')
#
#fit_params = np.vstack([fit_params[i] for i in range(len(fit_params))])
#chi2dof = np.hstack(chi2dof)
#window_width = np.hstack(window_width)
#window_midpoints = np.hstack(window_midpoints)
#order = np.argsort(chi2dof)[::-1]
#
#
#ax = fig.add_subplot(gs[1,0])
#cs1 = ax.scatter([abs(2*fit_params[i,0]*window_midpoints[i]) for i in order
#    if (window_width[i]<=1e2) and (abs(fit_params[i,1]/(2*fit_params[i,0]*window_midpoints[i]))>=1)],
#    [abs(fit_params[i,1]) for i in order
#        if (window_width[i]<=1e2) and (abs(fit_params[i,1]/(2*fit_params[i,0]*window_midpoints[i]))>=1)],
#    s=0.3, c=[window_width[i] for i in order
#        if (window_width[i]<=1e2) and (abs(fit_params[i,1]/(2*fit_params[i,0]*window_midpoints[i]))>=1)],
#    norm=LogNorm(), cmap=plt.cm.get_cmap('gist_rainbow'))
#cs2 = ax.scatter([abs(fit_params[i,1]) for i in order
#    if (window_width[i]<=1e2) and (abs(fit_params[i,1]/(2*fit_params[i,0]*window_midpoints[i]))<1)],
#    [abs(2*fit_params[i,0]*window_midpoints[i]) for i in order
#        if (window_width[i]<=1e2) and (abs(fit_params[i,1]/(2*fit_params[i,0]*window_midpoints[i]))<1)],
#    s=0.3, c=[window_width[i] for i in order
#        if (window_width[i]<=1e2) and (abs(fit_params[i,1]/(2*fit_params[i,0]*window_midpoints[i]))<1)],
#    norm=LogNorm(), cmap=plt.cm.get_cmap('gist_rainbow'))
#ax.set_xscale('log')
#ax.set_yscale('log')
#ax.set_xlabel('$a$')
#ax.set_ylabel('$b$')
#fig.colorbar(cs1, ax=ax).set_label('window width')
#
#
#
#x_space = np.logspace(np.log10(1e-10), np.log10(1e-1), 100)
#y_space = np.logspace(np.log10(1e-7), np.log10(1e-3), 100)
#weights = np.exp(-(chi2dof-1)**2/(2*0.5**2))
#
#hx = [[abs(2*fit_params[i,0]*window_midpoints[i]) for i in order
#    if (window_width[i]<1e2) and (abs(fit_params[i,1]/(2*fit_params[i,0]*window_midpoints[i]))>=1)],
#    [abs(fit_params[i,1]) for i in order
#        if (window_width[i]<1e2) and (abs(fit_params[i,1]/(2*fit_params[i,0]*window_midpoints[i]))<1)]]
#hy = [[abs(fit_params[i,1]) for i in order
#    if (window_width[i]<1e2) and (abs(fit_params[i,1]/(2*fit_params[i,0]*window_midpoints[i]))>=1)],
#    [abs(2*fit_params[i,0]*window_midpoints[i]) for i in order
#        if (window_width[i]<1e2) and (abs(fit_params[i,1]/(2*fit_params[i,0]*window_midpoints[i]))<1)]]
#
#hx = [inner for outer in hx for inner in outer]
#hy = [inner for outer in hy for inner in outer]
#
#ax = fig.add_subplot(gs[1,1])
##h = ax.hist2d([abs(fit_params[i,0]) for i in order],
##        [abs(fit_params[i,1] + 2*fit_params[i,0]*window_midpoints[i]) for i in order],
##        bins=(x_space, y_space), weights=weights)
#h = ax.hist2d(hx,hy,bins=(x_space,y_space))
#ax.set_xlabel('$a$')
#ax.set_ylabel('Slopes [units/day]')
#ax.set_xscale('log')
#ax.set_yscale('log')
#fig.colorbar(h[3], ax=ax)
#plt.show()
#
#
#h[0][:] = [norm_list([h[0][i][j] for j in range(len(h[0][0]))]) for i in range(len(h[0]))]
#fig, axs = plt.subplots(1,1, figsize=(10,8))
#axs.imshow(h[0].T, origin='lower', interpolation='none', extent=[-10,-1,-7,-3], aspect=9/4)
#plt.show()
#
#
#from scipy.stats import norm, skewnorm
#def skewnormfit(x, a, loc, scale) :
#    return skewnorm.pdf(x, a=a, loc=loc, scale=scale)
#
#data = [abs(fit_params[i,1] + 2*fit_params[i,0]*window_midpoints[i]) for i in order
#        if (window_width[i]<=1e2)]
#
#xs = np.linspace(-14, -6, 200)
#
#
#fig, axs = plt.subplots(1,1, figsize=(10,8))
#h = axs.hist(np.log10(data), bins=xs, density=True)
#
#hfitps = curve_fit(skewnormfit, h[1][:-1], h[0], bounds=((-np.inf,-np.inf,0),(0,0,10)))
#print(hfitps[0])
#
#axs.axvline(x=np.log10(6.47e-11), c='k', linestyle='--', linewidth=0.5)
#axs.plot(xs, skewnorm.pdf(xs, a=hfitps[0][0], loc=hfitps[0][1], scale=hfitps[0][2]))
#axs.set_ylim(bottom=0, top=1.5*np.max(h[0]))
#axs.set_xlabel('Slopes [units/day]')
#plt.show()
