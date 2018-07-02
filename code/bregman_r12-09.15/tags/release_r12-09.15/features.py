# features.py - feature extraction and plotting
# Bregman - music information retrieval toolkit
"""
Overview
========
To get features from audio signals, or any signal, use a feature extractor class.
Feature extractors are derived from the base Features class in module **features_base**. 

Instantiate any feature extractor class with a signal vector, or audio file name, and
feature parameter keywrod arguments. E.g.::
::
   myFeatures = featureExtractor(fileName, param1=value1, param2=value2, ...)

The global default feature-extractor parameters are defined in a parameter dictionary: 
::
    default_feature_params()
    {
        'sample_rate': 44100, # The audio sample rate
        'feature':'cqft',     # Which feature to extract (automatic for Features derived classes)
        'nbpo': 12,           # Number of Bands Per Octave for front-end filterbank
        'ncoef' : 10,         # Number of cepstral coefficients to use for cepstral features
        'lcoef' : 1,          # Starting cepstral coefficient
        'lo': 62.5,           # Lowest band edge frequency of filterbank
        'hi': 16000,          # Highest band edge frequency of filterbank
        'nfft': 16384,        # FFT length for filterbank
        'wfft': 8192,         # FFT signal window length
        'nhop': 4410,         # FFT hop size
        'window' : 'hamm',    # FFT window type 
        'log10': False,       # Whether to use log output
        'magnitude': True,    # Whether to use magnitude (False=power)
        'power_ext': ".power",# File extension for power files
        'intensify' : False,  # Whether to use critical band masking in chroma extraction
        'onsets' : False,     # Whether to use onset-synchronus features
        'verbosity' : 1       # How much to tell the user about extraction
    }

Parameter keywords can be passed explicitly as formal arguments or as a keyword argument parameter dict:, e.g.:
::
   myFeatures = featureExtractor(fileName, nbpo=24, nhop=2205 )
   myFeatures = featureExtractor(fileName, **{'nbpo':24, 'nhop':2205} )

To make a new feature extractor, just derive your new class from the Features class.
New feature extractor classes might override default parameters and override the *extract* method:
::
   class MyExtractor(Features):
       def __init__(self, arg, **feature_params):
           feature_params['feature'] = 'hcqft'
           Features.__init__(self, arg, feature_params)

       def extract(self):
           Features.extract(self)
           self.X = do_some_extra_stuff(self.X) # further process the features
           self.__setattr__('new_parameter', new_value) # set up some new feature attributes

Features instance members
=========================
Any instance of the Features class (including all the feature extractors below) contain the feature parameters as class members.
Additionally, a number of other useful class members are provided as follows:
::
   F = LogFrequencySpectrum(x) # An example instance of the Features class (or F=LogFrequencyCepstrum(x), etc...)

   F.any_feature_parameter # any keyword parameter: sample_rate, feature, nbpo, ncoef, lcoef, lo, hi, etc...
   F.X # instance features expressed as a N x T column-wise observation matrix for N-dimensional features
   F.STFT # complex-valued half spectrum of the STFT
   F.POWER # total power per frame
   F.Q # if log spectrum invoked, contains the constant-Q transform matrix for STFT->CQFT, else None
   F.CQFT # if log spectrum invoked, contains the CQFT feature matrix
   F.MFCC # if log cepstrum invoked, contains the MFCC feature matrix
   
   # Private (hidden) members that may be useful
   F._outN # size of the front-end output (F.fftN/2+1)
   F._cqtN # size of the log spectrum output
   F._fftfrqs # The center frequencies, up to F.hi, of the front-end filterbank (STFT)
   F._logfrqs # The center frequences, up to F.hi, of the log frequency transform (if extracted)
   F._logfbws # The bandwidths of the log frequency transform (if extracted)

Feature Extractors
==================
"""

__version__ = '1.0'
__author__ = 'Michael A. Casey'
__copyright__ = "Copyright (C) 2010  Michael Casey, Dartmouth College, All Rights Reserved"
__license__ = "gpl 2.0 or higher"
__email__ = 'mcasey@dartmouth.edu'

import pylab as P
import error
import glob
import plca
from sound import *
from audiodb import *
import pdb
from features_base import Features, feature_plot, feature_scale

# All features exposed as separate classes

# Frequency Domain
class LinearFrequencySpectrum(Features):
    """
    Linear-frequency spectrum, the short-time Fourier transform.
    ::
        feature = 'stft' # The underlying algorithm
    For the STFT implementation, the following parameters control the trade-off between information in time and information in frequency:
    ::
        nfft = 16384 # default fft size
        wfft = 8192  # default window size
        nhop = 4410  # default hop size
    """
    def __init__(self, arg=None, **feature_params):
        feature_params['feature']='stft'
        Features.__init__(self, arg, feature_params)

class LogFrequencySpectrum(Features):
    """
    Log-frequency constant-Q spectrum
    ::
        feature_params['feature']='cqft'
    """
    def __init__(self, arg=None, **feature_params):
        feature_params['feature']='cqft'
        Features.__init__(self, arg, feature_params)

class MelFrequencySpectrum(Features):
    """
    Mel-frequency constant-Q spectrum (same as log-frequency constant-Q spectrum)
    ::
        feature_params['feature']='cqft'
    """
    def __init__(self, arg=None, **feature_params):
        feature_params['feature']='cqft'    
        Features.__init__(self, arg, feature_params)

class Chromagram(Features):
    """"
    Chromagram
    ::
        feature_params['feature']='chroma'    
    """
    def __init__(self, arg=None, **feature_params):
        feature_params['feature']='chroma'    
        Features.__init__(self, arg, feature_params)

class HighQuefrencyChromagram(Features):
    """"
    HighQuefrenyChromagram (High-Pass Liftered with MFCCs)
    ::
        feature_params['feature']='hchroma'    
    """
    def __init__(self, arg=None, **feature_params):
        feature_params['feature']='hchroma'    
        Features.__init__(self, arg, feature_params)

# Cepstral Domain
#class LinearFrequencyCepstrum(Features):
#    """
#    Linear-frequency cepstrum
#    ::
#        feature_params['feature']='lcqft'    
#    """
#    def __init__(self, arg=None, **feature_params):
#        feature_params['feature']='lcqft'    
#        Features.__init__(self, arg, feature_params)

class LogFrequencyCepstrum(Features):
    """
    Log-frequency cepstrum (same as mel-frequency cepstrum)
    ::
        feature_params['feature']='mfcc'    
    """
    def __init__(self, arg=None, **feature_params):
        feature_params['feature']='mfcc'    
        Features.__init__(self, arg, feature_params)

class MelFrequencyCepstrum(Features):
    """
    Log-frequency cepstrum (approximates MFCC, same as log-frequency cepstrum)
    ::
        feature_params['feature']='mfcc'    
    """
    def __init__(self, arg=None, **feature_params):
        feature_params['feature']='mfcc'    
        Features.__init__(self, arg, feature_params)

# Quefrency-Domain Liftered
# class LowQuefrencyLinearSpectrum(Features):
#     def __init__(self, arg=None, **feature_params):
#         Features.__init__(self, arg, feature_params)

# class HighQuefrencyLinearSpectrum(Features):
#     def __init__(self, arg=None, **feature_params):
#         Features.__init__(self, arg, feature_params)

class LowQuefrencyLogFrequencySpectrum(Features):
    """
    Low-Quefrency Log Frequency Spectrum
    ::
        feature_params['feature']='lcqft'            
    """
    def __init__(self, arg=None, **feature_params):
        feature_params['feature']='lcqft'            
        Features.__init__(self, arg, feature_params)

class HighQuefrencyLogFrequencySpectrum(Features):
    """
    High-Quefrency Log-Frequency Spectrum:
    ::
        feature_params['feature']='hcqft'                    
    """
    def __init__(self, arg=None, **feature_params):
        feature_params['feature']='hcqft'            
        Features.__init__(self, arg, feature_params)

class LowQuefrencyMelSpectrum(Features):
    """
    Low-Quefrency Mel-Frequency Spectrum
    ::
        feature_params['feature']='lcqft'            
    """
    def __init__(self, arg=None, **feature_params):
        feature_params['feature']='lcqft'            
        Features.__init__(self, arg, feature_params)

class HighQuefrencyMelSpectrum(Features):
    """
    High-Quefrency Mel-Frequency Spectrum:
    ::
        feature_params['feature']='hcqft'                    
    """
    def __init__(self, arg=None, **feature_params):
        feature_params['feature']='hcqft'            
        Features.__init__(self, arg, feature_params)

# Time Domain
class RMS(Features):
    """
    Root mean square (RMS)
    ::
        feature_params['feature']='power'
        feature_params['mantitude']=True
        feature_params['log10']=False    
    """
    def __init__(self, arg=None, **feature_params):        
        feature_params['feature']='power'
        feature_params['mantitude']=True
        feature_params['log10']=False
        Features.__init__(self, arg, feature_params)

class LinearPower(Features):
    """
    Linear power
    ::
        feature_params['feature']='power'
        feature_params['mantitude']=False
        feature_params['log10']=False
    """
    def __init__(self, arg=None, **feature_params):
        feature_params['feature']='power'
        feature_params['mantitude']=False
        feature_params['log10']=False
        Features.__init__(self, arg, feature_params)

class dBPower(Features):
    """
    deci-Bel power (dB power)
    ::
        feature_params['feature']='power'
        feature_params['mantitude']=False
        feature_params['log10']=True    
    """
    def __init__(self, arg=None, **feature_params):
        feature_params['feature']='power'
        feature_params['mantitude']=False
        feature_params['log10']=True
        Features.__init__(self, arg, feature_params)

# Statistics and Derivatives
class LinearFrequencySpectrumCentroid(Features):
    """
    Linear-Frequency Spectrum Centroid
    """
    def __init__(self, arg, **kwargs):
        kwargs['feature']='stft'
        Features.__init__(self, arg, kwargs)
    def extract(self):
        Features.extract(self)
        self.X = (self.X.T * self._fftfrqs).sum(1) / self.X.T.sum(1)

class LogFrequencySpectrumCentroid(Features):
    """
    Log-Frequency Spectrum Centroid
    """
    def __init__(self, arg, **kwargs):
        kwargs['feature']='cqft'
        Features.__init__(self, arg, kwargs)
    def extract(self):
        Features.extract(self)
        self.X = (self.X.T * self._logfrqs).sum(1) / self.X.T.sum(1)

class MelFrequencySpectrumCentroid(Features):
    """
    Mel-Frequency Spectrum Centroid
    """
    def __init__(self, arg, **kwargs):
        kwargs['feature']='cqft'
        Features.__init__(self, arg, kwargs)
    def extract(self):
        Features.extract(self)
        self.X = (self.X.T * self._logfrqs).sum(1) / self.X.T.sum(1)

class LinearFrequencySpectrumSpread(Features):
    """
    Linear-Frequency Spectrum Spread
    """
    def __init__(self, arg, **kwargs):
        kwargs['feature']='stft'
        Features.__init__(self, arg, kwargs)
    def extract(self):
        Features.extract(self)
        mf = (self.X.T * self._fftfrqs).sum(1) / self.X.T.sum(1)
        self.X = (((self.X / self.X.T.sum(1)).T * ((P.atleast_2d(self._fftfrqs).T - mf)).T)**2).sum(1)

class LogFrequencySpectrumSpread(Features):
    """
    Log-Frequency Spectrum Spread
    """
    def __init__(self, arg, **kwargs):
        kwargs['feature']='cqft'
        Features.__init__(self, arg, kwargs)
    def extract(self):
        Features.extract(self)
        mf = (self.X.T * self._logfrqs).sum(1) / self.X.T.sum(1)
        self.X = (((self.X / self.X.T.sum(1)).T * ((P.atleast_2d(self._logfrqs).T - mf)).T)**2).sum(1) 

class MelFrequencySpectrumSpread(Features):
    """
    Mel-Frequency Spectrum Spread
    """
    def __init__(self, arg, **kwargs):
        kwargs['feature']='cqft'
        Features.__init__(self, arg, kwargs)
    def extract(self):
        Features.extract(self)
        mf = (self.X.T * self._logfrqs).sum(1) / self.X.T.sum(1)
        self.X = (((self.X / self.X.T.sum(1)).T * ((P.atleast_2d(self._logfrqs).T - mf)).T)**2).sum(1) 

#TODO: have STFT calculate _fftfreqs so axis plots are easy
#      have feature_plot be intelligent about feature dimensions
#      check frame count and last frame behaviour
#      limit frames read with keyword argument
      
# class LinearFrequencySpectrumFlux(Features):
#     def __init__(self, arg=None, **feature_params):
#         Features.__init__(self, arg, feature_params)

# class LogFrequencySpectrumFlux(Features):
#     def __init__(self, arg=None, **feature_params):
#         Features.__init__(self, arg, feature_params)

# class MelFrequencySpectrumFlux(Features):
#     def __init__(self, arg=None, **feature_params):
#         Features.__init__(self, arg, feature_params)

"""
"""

def separate(cls, x, p, n=5, play_flag=False, **kwargs):
    """
    Features may be separated into auditory streams using probabilistic latent component analysis (PLCA).
    Use the following functions to process features into separate auditory information streams, corresponding to underlying auditory objects in the mixed input signal.

    The static function *separate* is a meta function to call class methods on the passed class.
    ::
        cls  # The PLCA class to use (PLCA, SIPLCA, SIPLCA2, etc.)
        X    # The feature matrix to separate
        n    # The numbre of PLCA components to initialize
        p    # dict containing feature parameters used for X
        kwargs # win=(shifts,len) for frequency and time shift invariance, 
        alphaW # Dirichlet pior on W basis
        alphaZ # Dircihlet prior on Z
        alphaH # Dirichlet prior on H
        betaW # Entropic pior on W basis
        betaZ # Entropic prior on Z
        betaH # Entropic prior on H
    """
    F = Features(x, p)
    w,z,h,norm,recon,logprob = cls.analyze(F.X, n, **kwargs)
    if cls==plca.SIPLCA2:
        X_hat = [invert_component(cls, w[:,k,:], z[k], h[k,:,:]) for k in range(len(z))]
    else:
       X_hat = [invert_component(cls, w[:,:,k], z[k], h[k,:]) for k in range(len(z))]
    if play_flag:
        play_components(X_hat, F, x)    
    return w,z,h,X_hat,F

def invert_component(cls, w, z, h):
    """
    Invert a  single PLCA component to separated features in the original feature space.
    """
    w = P.atleast_2d(w)
    if cls==plca.PLCA: w = w.T 
    h = P.atleast_2d(h)
    return cls.reconstruct(w,z,h)

def invert_components(cls, w, z, h):
    """
    Invert group of PLCA components to separated features in the original feature space.
    """
    if cls==plca.SIPLCA2:
        X_hat = [invert_component(cls, w[:,k,:], z[k], h[k,:,:]) for k in range(len(z))]
    else:
       X_hat = [invert_component(cls, w[:,:,k], z[k], h[k,:]) for k in range(len(z))]
    return X_hat

def play_components(X_hat, F, x=None, orig_mix=0.1):
    """ ::    
        play individual separated feature spectrograms as audio
        separated spectrograms are stored in a list
     """
    for Xh in X_hat:
        x_hat = F.inverse(Xh) # invert to audio to F.x_hat
        x = x_hat if x is None else x
        x = x.mean(1) if len(x.shape)>1 else x
        xn = min(len(x), len(x_hat))
        x_orig = orig_mix*P.atleast_1d(x[:xn]).T
        x_hat = P.atleast_1d(x_hat[:xn] / (x_hat[:xn].max() + 0.01))
        play(P.c_[x_orig, x_hat].T, F.feature_params['sample_rate'])

# convenience handles
imagesc = feature_plot
default_feature_params = Features.default_params
P.ion() # activiate interactive plotting
