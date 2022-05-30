

import numpy as np
from find_quadratics import find_quadratic
from scipy.optimize import curve_fit
import scipy.integrate as integrate
from data_parse import parse_data, clean_data
from center_data import center_data
from signal_filters import sgfilter
import numpy.polynomial.polynomial as poly
from scipy import signal


qlist, quasar = parse_data()
#lcs = [np.vstack((quasar[qlist[n]]['time'], quasar[qlist[n]]['Vflux'], quasar[qlist[n]]['Vfluxerr'])).T for n in range(len(qlist))]
lcs = [np.vstack((quasar[Q]['time'], quasar[Q]['V'], quasar[Q]['Verr'])).T for Q in ['25.3712.72','206.17052.388']]
#lcs = [np.vstack((np.linspace(0,i*50,i*50), (-1)**i*signal.gaussian(i*50,std=7*i))).T for i in range(1,3)]
#lcs = [np.vstack((np.linspace(0,50,50), (-1)**i*signal.gaussian(50,std=7*i))).T for i in range(1,3)]
#lcs[1] = np.delete(lcs[1], np.s_[:8], 0)
#lcs[1][:,0] -= min(lcs[1][:,0])
zs  = [quasar[Q]['z'] for Q in ['25.3712.72','206.17052.388']]

lcs = [clean_data(lcs[n]) for n in range(len(lcs))]
#lcs = [center_data(lcs[n]) for n in range(len(lcs))]
lcs = [sgfilter(lcs[n]) for n in range(len(lcs))]

for n in range(len(lcs)) :
    for i in range(len(lcs[n][:,0])-1) :
        if abs(lcs[n][i+1,0] - lcs[n][i,0]) <= 1 :
            mean = (lcs[n][i+1,1] + lcs[n][i,1]) / 2
            lcs[n][i+1,1] = mean
            lcs[n][i,1]   = mean
            lcs[n][i+1,0] += 0.001

from scipy.interpolate import CubicSpline
cs = [CubicSpline(lcs[n][:,0], lcs[n][:,1]) for n in range(len(lcs))]
xs = [np.arange(np.min(lcs[n][:,0]),np.max(lcs[n][:,0]), 1) for n in range(len(lcs))]

s = [integrate.cumtrapz(np.sqrt(1+cs[n](xs[n],1)**2),xs[n]) for n in range(len(cs))]
#d2f_ds2 = -cs(xs,2)*(cs(xs,1)/(1+cs(xs,1)**2))**2 + cs(xs,2)/(1+cs(xs,1)**2)



import matplotlib.pyplot as plt
cm = 1/2.54
plt.style.use(['science','no-latex'])
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors


plt.rc('font', size=8)
fig, axs = plt.subplots(1,1, figsize=(8.6*cm,7*cm))
[axs.scatter(lcs[n][:,0], lcs[n][:,1], s=0.8, c='k') for n in range(len(lcs))]
axs.plot(xs[0], cs[0](xs[0]))
axs.plot(lcs[0][:,0], lcs[0][:,1])
axs.set_xlabel('Days')
axs.set_ylabel('Flux')
plt.tight_layout()



k = [integrate.cumtrapz(abs(cs[n](xs[n],2)), xs[n]) for n in range(len(cs))]
fig, axs = plt.subplots(1,1, figsize=(8.6*cm,7*cm))
[axs.plot(xs[n][:-1], k[n], c='k') for n in range(len(k))]
axs.set_xlabel('Days')
axs.set_ylabel('$\int|\kappa|$')
plt.tight_layout()

kappa_k_spline = [CubicSpline(abs(k[n]), abs(cs[n](xs[n],2)[:-1])) for n in range(len(cs))]

fig, axs = plt.subplots(1,1, figsize=(8.6*cm,7*cm))
[axs.plot(k[n], kappa_k_spline[n](k[n],0)) for n in range(len(k))]
axs.set_xlabel('$\int|\kappa|$')
axs.set_ylabel('$\kappa$')
plt.tight_layout()


fbar = np.mean(kappa_k_spline[0](k[0],0))
tbar = np.mean(kappa_k_spline[1](k[1],0))

cutoff = 500
offsets = np.linspace(0, np.max(k[1]), len(xs[1]))
v = [sum([(kappa_k_spline[0](k[0][i],0) - fbar)*(kappa_k_spline[1](k[1][i]-offsets[j],0) - tbar) for i in range(min(len(k[0]),(len(k[1])-j)))]) / np.sqrt(sum([(kappa_k_spline[0](k[0][i],0)-fbar)**2 for i in range(min(len(k[0]),(len(k[1])-j)))])*sum([(kappa_k_spline[1](k[1][i]-offsets[j],0)-tbar)**2 for i in range(min(len(k[0]),(len(k[1])-j)))])) for j in range(len(offsets)-cutoff)]

print(offsets[np.where(v == max(v))])
print(kappa_k_spline[1](offsets[np.where(v == max(v))],0))
print(xs[1][np.where(kappa_k_spline[1](offsets,0) == kappa_k_spline[1](offsets[np.where(v == max(v))],0))])
print(np.max(abs(cs[1](xs[1],2))/np.max(abs(cs[0](xs[0],2)))))

fig, axs = plt.subplots(1,1, figsize=(8.6*cm,7*cm))
axs.plot(offsets[:-cutoff],v)
axs.set_xlabel('offsets')
axs.set_ylabel('v')
plt.tight_layout()

fig, axs = plt.subplots(1,1, figsize=(8.6*cm,7*cm))
axs.plot(xs[1][:-cutoff],v)
axs.set_xlabel('Days offset')
axs.set_ylabel('v')
plt.tight_layout()
plt.show()


