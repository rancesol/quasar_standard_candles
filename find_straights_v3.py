

import numpy as np
from numpy import sin, pi
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from data_parse_prime import parse_data, clean_data



def func(x, m, b) :     # fitting function for curve_fit
    return m*x + b

def norm_list(lst) :
    if np.max(lst) > 0 :
        return lst/np.max(lst)
    else :
        return lst


def find_straights(a, minww) :
    ## define the windows to fit to
    ## starts with full data set and cuts in half until it reaches half the minimum window width (minww)
    ## setting overlap==True allows for more sampling though it disfavors the edges
    ## setting overlap==False samples evenly from end to end but comes with less sampling
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



    ## some windows may have less data points than the number of fitting parameters (2 for linear fit)
    ## this will throw an error when fitting so we discard these regions
    ## we arbitrarily take a minimum of 10 points cause I don't want to be caught drawing conclusions -
    ## from a line defined with just two data points
    mask = [[(len(windows[j][i]) > 10) for i in range(len(windows[j]))] for j in range(len(windows))]
    windows = [[windows[j][i] for i in range(len(windows[j])) if mask[j][i]] for j in range(len(windows))]
   

    
    ## gaps in the data larger than the minimum window size will return an empty window array
    ## this will throw errors so discard the empty windows
    mask = [np.size(w) > 2 for w in windows[:]]
    windows = [windows[j] for j in range(len(windows)) if mask[j]]

    ## windows with sparse data bias the preferred slope towards lower values
    ## we arbitrarily require a sampling every 10 days on average
    mask = [[(len(windows[j][i]) / np.ptp(windows[j][i][:,0]) > 1/250) for i in range(len(windows[j]))]
            for j in range(len(windows))]
    windows = [[windows[j][i] for i in range(len(windows[j])) if mask[j][i]] for j in range(len(windows))]


    mask = [(len(windows[j]) > 2) for j in range(len(windows))]
    windows = [windows[j] for j in range(len(windows)) if mask[j]]



    if len(windows) > 0 :
        ## make best fits of all the windows and calculate the cooresponding chi^2/dof
        try :
            fit_params, _ = zip(*[zip(*[curve_fit(func, windows[j][i][:,0], windows[j][i][:,1],
                sigma=windows[j][i][:,2])
                for i in range(len(windows[j]))])
                for j in range(len(windows)) if (np.size(windows[j]) > 2)])
            
            chi2dof = [[np.sum([((windows[j][i][k,1] - func(windows[j][i][k,0], fit_params[j][i][0],
                fit_params[j][i][1]))/windows[j][i][k,2])**2/(len(windows[j][i])-2)
                for k in range(len(windows[j][i]))])
                for i in range(len(windows[j]))]
                for j in range(len(windows))]



            ## we no longer need the list of list of arrays so we can flatten windows
            ## we also collect the window widths to correspond slopes to their time-scales
            windows = [window for sublist in windows for window in sublist]
            window_width = [np.ptp(windows[i][:,0]) for i in range(len(windows))]
            fit_params = np.vstack([fit_params[i] for i in range(len(fit_params))])
            chi2dof   = np.hstack(chi2dof)



            ## for analysis purposes we collect the N lines with the lowest (or highest) chi^2/dof
            N = 20
            min_i = np.argsort(chi2dof)
            best_fit = [np.vstack((windows[i][:,0], fit_params[i,0]*windows[i][:,0] + fit_params[i,1])).T
                    for i in min_i[:]]

            return window_width, fit_params, chi2dof, best_fit
        except RuntimeError :
            return 0, 0, 0, 0
    else :
        return 0, 0, 0, 0



### read in data
#qlist, quasar = parse_data()
#
##n = 39
##Vflux = np.vstack((quasar[qlist[n]]['time'], quasar[qlist[n]]['Vflux'], quasar[qlist[n]]['Vfluxerr'])).T
##Vflux = clean_data(Vflux)
#Vflux = [np.vstack((quasar[qlist[n]]['time'], quasar[qlist[n]]['Rflux'], quasar[qlist[n]]['Rfluxerr'])).T
#        for n in range(len(qlist)) if qlist[n] != 'quasar_info']
#Vflux = [clean_data(Vflux[n]) for n in range(len(Vflux))]
#
##window_width, fit_params, chi2dof, best_fit = find_straights(Vflux)
#minww = 10
#mask = [np.ptp(Vflux[i][:,0]) > minww for i in range(len(Vflux))]
#Vflux = [Vflux[i] for i in range(len(Vflux)) if mask[i]]
#window_width, fit_params, chi2dof, best_fit = zip(*[find_straights(Vflux[n],minww) for n in range(len(Vflux))])
#
#
##index = 0   #np.where(np.asarray(qlist) == 'J1515.ECAM.0')[0][0]
##import matplotlib.pyplot as plt
##fig, axs = plt.subplots(2,1, figsize=(10,8))
##axs[0].errorbar(Vflux[index][:,0], Vflux[index][:,1], yerr=Vflux[index][:,2], xerr=None, ls='none', c='C0')
##[axs[0].plot(best_fit[index][i][:,0], best_fit[index][i][:,1], c='k', linewidth=0.5) for i in range(len(best_fit[index]))]
##axs[0].set_xlabel('Days (in quasar RF)')
##axs[0].set_ylabel('Flux of V-baind')
##axs[0].set_title('Quasar: ' + qlist[index] + ',   z = ' + str(quasar[qlist[index]]['z']))
##
##fit_params = np.vstack([fit_params[i] for i in range(len(fit_params))])
##chi2dof = np.hstack(chi2dof)
##window_width = np.hstack(window_width)
##
##order = np.argsort(chi2dof)[::-1]
##cs = axs[1].scatter([abs(fit_params[i,0]) for i in order], [window_width[i] for i in order], s=0.5, c=[chi2dof[i] for i in order], norm=LogNorm(vmin=7e-2, vmax=2e2),
##        cmap=plt.cm.get_cmap('gist_rainbow'))
##axs[1].set_yscale('log')
##axs[1].set_xscale('log')
###axs[1].set_xlim(left=0)    
###axs[1].set_ylim([1e-2, 2e2])
##fig.colorbar(cs, ax=axs[1]).set_label('$\chi^2$/DOF')
##
##k = 6.47e-11
###axs[1].axvspan(-k-0.07e-11, -k+0.07e-11, alpha=0.5, color='k')
##axs[1].axvspan( k-0.07e-11,  k+0.07e-11, alpha=0.5, color='k')
##axs[1].set_xlabel('Slopes [units/day]')
##axs[1].set_ylabel('Time Scale')
###axs[1].set_facecolor('gray')
##
##x_space = np.logspace(np.log10(1e-6), np.log10(2e-3), 100)
##y_space = np.logspace(np.log10(1e1), np.log10(1e3), 100)
##
##axs[1].hist2d([abs(fit_params[i,0]) for i in order], [window_width[i] for i in order],
##        bins=(x_space,y_space), alpha=.7)
##plt.tight_layout()
##plt.show()
#
#
#index = 5   #np.where(np.asarray(qlist) == 'J1515.ECAM.0')[0][0]
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#
#fig, axs = plt.subplots(1,1, figsize=(10,4))
#axs.errorbar(Vflux[index][:,0], Vflux[index][:,1], yerr=Vflux[index][:,2], xerr=None, ls='none', c='C0')
#axs.set_xlabel('Days (in QSO frame)')
#axs.set_ylabel('Flux of R-band')
#axs.set_title('Quasar: ' + qlist[index] + ',   z = ' + str(quasar[qlist[index]]['z']))
#plt.tight_layout()
#plt.show()
#        
#
#
#fig = plt.figure(tight_layout=True)
#gs  = gridspec.GridSpec(2,2)
#
#ax = fig.add_subplot(gs[0,:])
#ax.errorbar(Vflux[index][:,0], Vflux[index][:,1], yerr=Vflux[index][:,2], xerr=None, ls='none', c='C0')
#[ax.plot(best_fit[index][i][:,0], best_fit[index][i][:,1], c='k', linewidth=0.5) for i in range(len(best_fit[index]))]
#ax.set_xlabel('Days (in QSO frame)')
#ax.set_ylabel('Flux of R-band')
#ax.set_title('Quasar: ' + qlist[index] + ',   z = ' + str(quasar[qlist[index]]['z']))
#
#overall_slopes = np.vstack([fit_params[i][0][0] for i in range(len(fit_params))])
#fit_params = np.vstack([fit_params[i] - overall_slopes[i] for i in range(len(fit_params))])
#chi2dof = np.hstack(chi2dof)
#window_width = np.hstack(window_width)
#order = np.argsort(chi2dof)[::-1]
#
#ax = fig.add_subplot(gs[1,0])
#cs = ax.scatter([abs(fit_params[i,0]) for i in order], [window_width[i] for i in order], s=0.5, c=[chi2dof[i] for i in order], norm=LogNorm(vmin=7e-2, vmax=2e2),
#        cmap=plt.cm.get_cmap('gist_rainbow'))
#ax.set_yscale('log')
#ax.set_xscale('log')
#ax.set_xlim([1e-6, 2e-3])
#ax.set_ylim([1e0, 1e3])
#fig.colorbar(cs, ax=ax).set_label('$\chi^2$/DOF')
#
#k = 6.47e-11
##ax.axvspan(-k-0.07e-11, -k+0.07e-11, alpha=0.5, color='k')
##ax.axvspan( k-0.07e-11,  k+0.07e-11, alpha=0.5, color='k')
#ax.set_xlabel('Slopes [units/day]')
#ax.set_ylabel('Time Scale')
##ax.set_facecolor('gray')
#
#x_space = np.logspace(np.log10(1e-6), np.log10(2e-3), 50)
#y_space = np.logspace(np.log10(1e0), np.log10(1e3), 50)
#weights = np.exp(-(chi2dof-1)**2/(2*0.125**2))   #(1+(1-chi2dof)**2)**-1
#
#import matplotlib.colors as mcolors
#ax = fig.add_subplot(gs[1,1])
#h = ax.hist2d([abs(fit_params[i,0]) for i in order],
#        [window_width[i] for i in order], bins=(x_space,y_space), weights=weights)
#ax.set_xlabel('Slopes [units/day]')
#ax.set_ylabel('Time Scale')
#ax.set_yscale('log')
#ax.set_xscale('log')
#fig.colorbar(h[3], ax=ax)
#plt.show()
#
#
#def fake_log(x, pos) :
#    return r'$10^{%d}$' % (x)
#
#h[0][:] = [norm_list([h[0][i][j] for i in range(len(h[0]))]) for j in range(len(h[0][0]))]
#fig, axs = plt.subplots(1,1, figsize=(10,8))
#axs.imshow(h[0], origin='lower', interpolation='none', extent=[-6,-3,0,3])
#axs.xaxis.set_major_formatter(fake_log)
#axs.yaxis.set_major_formatter(fake_log)
##plt.xscale('log')
##plt.yscale('log')
#plt.show()
