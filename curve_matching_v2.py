
import numpy as np
from scipy import signal, spatial
import matplotlib.pyplot as plt
from synthetic_random_data_v2 import curve_gen
from center_data import center_data
from signal_filters import sgfilter
import numpy.polynomial.polynomial as poly
from scipy.interpolate import CubicSpline
import scipy.integrate as integrate


def brk_into_windows(a, minww) :
    overlap = False
    if overlap == True :
        windows = [[a[(a[:,0] >= np.min(a[:,0]) + 0.5*i*np.ptp(a[:,0])/(j+1))
            & (a[:,0] <= np.max(a[:,0]) - (j-0.5*i)*np.ptp(a[:,0])/(j+1))]
            for i in range(2*(j+1)-1)]
            for j in range(int(np.ptp(a[:,0])/minww))]
    else :
        windows = [[a[(a[:,0] >= np.min(a[:,0]) + i*np.ptp(a[:,0])/(j+1))
            & (a[:,0] <= np.max(a[:,0]) - (j-i)*np.ptp(a[:,0])/(j+1))]
            for i in range(j+1)]
            for j in range(int(2*np.ptp(a[:,0])/minww))]


    ### some windows may have very few data points
    ### we arbitrarily take a minimum of 10 points as a lower cutoff
    #mask = [[(len(windows[j][i]) > 10) for i in range(len(windows[j]))] for j in range(len(windows))]
    #windows = [[windows[j][i] for i in range(len(windows[j])) if mask[j][i]] for j in range(len(windows))]

    ### gaps in the data larger than the minimum window size will return an empty window array
    ### this will throw errors so discard the empty windows
    #mask = [np.size(w) > 2 for w in windows[:]]
    #windows = [windows[j] for j in range(len(windows)) if mask[j]]

    return windows


def correlation1(A, B) :
    if len(A) > len(B) :
        a = center_data(A[:-1])
        b = center_data(B)
    elif len(B) > len(A) :
        a = center_data(A)
        b = center_data(B[:-1])
    else :
        a = center_data(A)
        b = center_data(B)

    #a[:,1] = np.sqrt((a[:,1]-np.mean(a[:,1]))**2)
    #b[:,1] = np.sqrt((b[:,1]-np.mean(b[:,1]))**2)
    corr = signal.correlate(a[:,1]-np.mean(a[:,1]), b[:,1]-np.mean(b[:,1]))

    norm = np.sqrt(max(signal.correlate(a[:,1]-np.mean(a[:,1]), a[:,1]-np.mean(a[:,1])))
            *max(signal.correlate(b[:,1]-np.mean(b[:,1]), b[:,1]-np.mean(b[:,1]))))
    return round(max(corr)/norm,6)


def correlation2(A,B) :
    if len(A) > len(B) :
        a = center_data(A[:-1])
        b = center_data(B)
    elif len(B) > len(A) :
        a = center_data(A)
        b = center_data(B[:-1])
    else :
        a = center_data(A)
        b = center_data(B)
    csa = CubicSpline(a[:,0], a[:,1])
    csb = CubicSpline(b[:,0], b[:,1])
    xsa = np.arange(np.min(a[:,0]),np.max(a[:,0]), 1)
    xsb = np.arange(np.min(b[:,0]),np.max(b[:,0]), 1)
    
    
    ka = integrate.cumtrapz(abs(csa(xsa,2)), xsa)
    kb = integrate.cumtrapz(abs(csb(xsb,2)), xsb)
    
    kappa_k_spline_a = CubicSpline(abs(ka), abs(csa(xsa,2)[:-1]))
    kappa_k_spline_b = CubicSpline(abs(kb), abs(csb(xsb,2)[:-1]))
    
    fbar = np.mean(kappa_k_spline_a(ka,0))
    tbar = np.mean(kappa_k_spline_b(kb,0))
    
    v = sum([(kappa_k_spline_a(ka[i],0) - fbar)*(kappa_k_spline_b(kb[i],0) - tbar)
        for i in range(min(len(ka),(len(kb))))]) / np.sqrt(sum([(kappa_k_spline_a(ka[i],0)-fbar)**2
            for i in range(min(len(ka),(len(kb))))])*sum([(kappa_k_spline_b(kb[i],0)-tbar)**2
                for i in range(min(len(ka),(len(kb))))]))

    return round(abs(v),6)

def return_matches(a,b,wa,wb) :
    Ia = 0
    Ja = 0
    Ib = 0
    Jb = 0

    overlap = False
    if overlap == True :
        corr = [[[correlation1(wa[min(i,len(wa)-1)][j],wb[min(i,len(wb)-1)][k])
            *correlation2(wa[min(i,len(wa)-1)][j],wb[min(i,len(wb)-1)][k])
            for k in range(min(int(2*i+1),int(2*(len(wb)-1)+1)))]   # cycle through wb
            for j in range(min(int(2*i+1),int(2*(len(wa)-1)+1)))]   # cycle through wa
            for i in range(max(len(wa),len(wb)))]
    else :
        corr = [[[correlation1(wa[min(i,len(wa)-1)][j],wb[min(i,len(wb)-1)][k])
            *correlation2(wa[min(i,len(wa)-1)][j],wb[min(i,len(wb)-1)][k])
            for k in range(min(int(i+1),int((len(wb)-1)+1)))]
            for j in range(min(int(i+1),int((len(wa)-1)+1)))]
            for i in range(max(len(wa),len(wb)))]
    
    weights = [[sum([sum([corr[n][j][k]
        for k in range(len(corr[n][j]))])
        for j in range(len(corr[n])) if a[i,0]>min(wa[n][j][:,0]) and a[i,0]<=max(wa[n][j][:,0])])
        for i in range(len(a[:,0]))] for n in range(len(wa))]

    weights = np.sum(weights, axis=0)
    
    return corr, weights

#//////////////////////////////////

## t = template curve (straight line for now)
## f = curve to be compared to template (obtained from synthetic_data_v2.py)
## w = windows or subregions of t and f

C1, slopes = curve_gen()
#C2, slopes = curve_gen()
C2 = np.copy(C1)
#noise = 1e-5*(2*np.random.rand(1,len(C2)) - 1)
#C2[:,1] = C2[:,1] + noise
#C2[:,1] *= -1
#C2[:,1] = C2[::-1,1]
C2 = C2[100:1100,:]
#C2[:,0] *= 2

if len(C1) >= len(C2) :
    t = C1
    f = C2
else :
    t = C2
    f = C1

t = sgfilter(center_data(t))
f = sgfilter(center_data(f))

minww = 1000

t_remainder = np.mod(len(t[:,0]),minww)
f_remainder = np.mod(len(f[:,0]),minww)

if np.mod(t_remainder,2) != 0 :
    t = t[t_remainder//2:len(t[:,0])-t_remainder//2+1,:]
else :
    t = t[t_remainder//2:len(t[:,0])-t_remainder//2,:]
if np.mod(f_remainder,2) != 0 :
    f = f[f_remainder//2:len(f[:,0])-f_remainder//2+1,:]
else :
    f = f[f_remainder//2:len(f[:,0])-f_remainder//2,:]



if not min(len(t[:,0]),len(f[:,0]))//minww > 1 :
    minww = min(len(t[:,0]), len(f[:,0]))
wt = brk_into_windows(t, minww)
wf = brk_into_windows(f, minww)

corr, weights= return_matches(t,f,wt,wf)


C1 = center_data(C1)
C2 = center_data(C2)
cm = 1/2.54
fig, axs = plt.subplots(2,1, figsize=(12*cm, 10*cm))
axs[0].scatter(C1[:,0], C1[:,1], s=0.8, c='k')
axs[0].scatter(C2[:,0], C2[:,1]+0.1, s=0.8, c='k')

axs[1].hist(t[:,0], bins=len(t[::50,0]), weights=weights)
plt.show()
