
from bregman.suite import *
from scipy.signal import lfilter
from pylab import zeros,where,diff,sqrt

default_params = Features.default_params()
default_params['feature']='cqft'
default_params['nbpo']=1
default_params['nhop']=441

def beat_track(x, p=default_params, coef_fun=lambda t: 0.5**(2.0/t)):
    """
    Scheirer beat tracker
    """
    frame_rate = p['sample_rate'] / float(p['nhop'])
    F = Features(x,p)
    D = diff(F.X,axis=1)
    D[where(D<0)] = 0
    D = (D.T/sqrt((D**2).sum(1))).T # unit norm
    tempos = range(20,400,4)    
    z = zeros((len(tempos), D.shape[0]))    
    for i, bpm in enumerate(tempos): # loop over tempos to test
        t = int(round(frame_rate * 60. / bpm)) # num frames per beat
        alpha = coef_fun(t)
        b = [1 - alpha]
        a = zeros(t)
        a[0] = 1.0
        a[-1] = alpha
        z[i,:] = lfilter(b, a, D).sum(1) # filter and sum sub-band onsets    
    return (z,tempos,D)


