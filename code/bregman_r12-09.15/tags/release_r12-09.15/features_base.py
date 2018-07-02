import pylab as P
import error
import glob
import plca
from sound import *
from audiodb import *

# Add filterbank implementation (gammatone, etc.)
# Correct frequency scaling: Mel, Log, etc.
# Support for HCQFT -> CHROMA

# Features Class
class Features(object):
    """        
    ::

        F = Features(arg, feature_params)
        type(arg) is str: load audio from file
        type(arg) is ndarray: set audio from array

        feature_params['feature'] = 
         'stft'   - short-time fourier transform
         'power'  - power
         'cqft'   - constant-q fourier transform
         'mfcc'   - Mel-frequency cepstral coefficients
         'lcqft'  - low-quefrency cepstral coefficients
         'hcqft'  - high-quefrency cepstral coefficients
         'chroma' - chroma (pitch-class) power coefficients

         Features are extracted in the following hierarchy:
          stft->cqft->mfcc->[lcqft,hcqft]->chroma,
          if a later feature was extracted, then an earlier feature is also available

         Once extracted, features have their own plot command:
          F.feature_plot(dbscale=True, normalize=False)          
            dbscale - optional deci-Bel scaling of audio features
            normalize - optional group normalization of features
            norm - optional column L2 norming of features

         Access to feature arrays. Use the following members which are numpy ndarrays:
          F.X - extracted features

         The following are explicit accessors to features in the extractor chain
          F.STFT 
          F.POWER
          F.CQFT
          F.MFCC
          F.LCQFT
          F.HCQFT
          F.CHROMA

         Features can be inverted by calling the inverse() method:
         F.inverse(F.X, pvoc=False)
            pvoc - optionally use phase-vocoder estimation to recover phases
         This produces a new signal F.x_hat 
         play(F.x_hat)
    """
    def __init__(self, arg=None, feature_params=None):
        self._initialize(feature_params)
        if type(arg)==P.ndarray:
            self.set_audio(arg, sr=self.sample_rate)
            self.extract()
        elif type(arg)==str:
            if arg:
                self.load_audio(arg) # open file as MONO signal
                self.extract()

    def _initialize(self, feature_params):
        """
        ::
          Initialize important parameters
        """
        self.reset()
        self.feature_params = self.default_params()
        self._check_feature_params(feature_params)
        self.extract_funs = {
            'power': self._power,
            'chroma': self._chroma,
            'hchroma': self._chroma_hcqft,
            'hcqft': self._hcqft,
            'lcqft': self._lcqft,
            'mfcc': self._mfcc,
            'cqft': self._cqft,
            'stft': self._stft
            }

    @staticmethod
    def default_params():
        """
        ::
            Return a new feature parameter dict. 
            Feature opcodes are listed in the Features documentation.
            default_feature_params = {
                'sample_rate': 44100,
                'feature':'cqft', 
                'nbpo' : 12,
                'ncoef' : 10,
                'lcoef' : 0,
                'lo': 62.5, 
                'hi': 16000,
                'nfft': 16384,
                'wfft': 8192,
                'nhop': 4410,
                'window' : 'hamm',
                'log10': False,
                'magnitude': True,
                'power_ext': ".power",
                'intensify' : False,
                'onsets' : False,
                'verbosity' : 0}
        """

        feature_params = {
            'sample_rate': 44100,
            'feature':'cqft', 
            'nbpo': 12,
            'ncoef' : 10,
            'lcoef' : 1,
            'lo': 62.5, 
            'hi': 16000,
            'nfft': 16384,
            'wfft': 8192,
            'nhop': 4410,
            'window' : 'hamm',
            'log10': False,
            'magnitude': True,
            'power_ext': ".power",
            'intensify' : False,
            'onsets' : False,
            'verbosity' : 0
            }
        return feature_params

    @staticmethod
    def default_feature_params():
        """
        ::
        Deprecated: use Features.default_params or module default_feature_params()
        """
        return Features.default_params()

    def reset(self):
        """
        ::

            Reset the feature extractor state. No signal. No features.
        """
        self._have_x=False
        self.x=None # the audio signal
        self._have_stft=False
        self.STFT=None
        self._have_cqft=False
        self.POWER=None
        self._have_power=False
        self._is_intensified=False
        self.CQFT=None
        self._have_mfcc=False
        self.MFCC=None
        self._have_lcqft=False
        self.LCQFT=None
        self._have_hcqft=False
        self.HCQFT=None
        self._have_chroma=False
        self.CHROMA=None
        self.inverse=None

    def load_audio(self,filename):
        """
        ::

            Open a WAV/AIFC/AU file as a MONO signal [L], sets audio buffer
        """
        self.x=WavOpen(filename, self.nhop, self.verbosity)
        self._have_x = True
        self.sample_rate = self.x.sample_rate

    def set_audio(self, x, sr=44100.):
        """
        ::

            Set audio buffer to extract as an array
        """
        self.reset()
        x = x.mean(1) if len(x.shape) > 1 else x # handle stereo
        pad = pylab.remainder(len(x),self.nhop)
        if pad: x = numpy.r_[x,numpy.zeros(self.nhop-pad)]
        self.x = x.reshape(-1,self.nhop)
        self._have_x=True
        self.sample_rate = sr

    def _check_feature_params(self,feature_params=None):
        self.feature_params = feature_params if feature_params is not None else self.feature_params
        fp = self.default_params()
        for k in fp.keys():
            self.feature_params[k] = self.feature_params.get(k, fp[k])
            self.__setattr__(k, self.feature_params[k])
        return self.feature_params

    def _extract_error(self):
        error.BregmanError("unrecognized feature in Features.extract()")

    def extract(self, feature_params=None):
        """
        ::
        
            Extract audio features according to feature_params specification:
        """
        f = self._check_feature_params(feature_params)['feature']
        self.extract_funs.get(f, self._extract_error)()
        if self.onsets: self._extract_onsets()

    def feature_plot(self,feature=None,normalize=False,dbscale=False, norm=False, interp='nearest', labels=True, nofig=False):
        """
        ::

          Plot the given feature, default is self.feature, 
           returns an error if feature not extracted

          Inputs:
           feature   - the feature to plot self.feature
                        features are extracted in the following hierarchy:
                           stft->cqft->mfcc->[lcqft,hcqft]->chroma,
                        if a later feature was extracted, then an earlier feature can be plotted
           normalize - column-wise normalization ['alse]
           dbscale   - transform linear power to decibels: 20*log10(X) [False]
           norm      - make columns unit norm [False]
           interp    - how to interpolate values in the plot ['nearest']
           labels    - whether to plot labels
           nofig     - whether to make new figure
        """
        feature = self._check_feature_params()['feature'] if feature is None else feature
        # check plots        
        if feature =='stft':
            if not self._have_stft:
                print "Error: must extract STFT first"
            else:
                feature_plot(P.absolute(self.STFT), normalize, dbscale, norm, title_string="STFT", interp=interp, nofig=nofig)
                if labels:
                    self._feature_plot_xticks(float(self.nhop)/float(self.sample_rate))
                    self._feature_plot_yticks(float(self.sample_rate)/(self.nfft))
                    P.xlabel('Time (secs)')
                    P.ylabel('Frequency (Hz)')
        elif feature == 'power':
            if not self._have_power:
                print "Error: must extract POWER first"
            else:
                if not nofig: P.figure()
                P.plot(feature_scale(self.POWER, normalize, dbscale)/20.0)
                if labels:
                    self._feature_plot_xticks(float(self.nhop)/float(self.sample_rate))
                    P.title("Power")
                    P.xlabel("Time (s)")
                    P.ylabel("Power (dB)")
        elif feature == 'cqft':
            if not self._have_cqft:
                print "Error: must extract CQFT first"
            else:
                feature_plot(self.CQFT, normalize, dbscale, norm, title_string="CQFT",interp=interp,nofig=nofig)
                if labels:
                    self._feature_plot_xticks(float(self.nhop)/float(self.sample_rate))
                    #self._feature_plot_yticks(1.)
                    P.yticks(P.arange(0,self._cqtN,self.nbpo), (self.lo*2**(P.arange(0,self._cqtN,self.nbpo)/self.nbpo)).round(1))
                    P.xlabel('Time (secs)')
                    P.ylabel('Frequency (Hz)')
        elif feature == 'mfcc':
            if not self._have_mfcc:
                print "Error: must extract MFCC first"
            else:
                fp = self._check_feature_params()
                X = self.MFCC[self.lcoef:self.lcoef+self.ncoef,:]
                feature_plot(X, normalize, dbscale, norm, title_string="MFCC",interp=interp,nofig=nofig)
                if labels:
                    self._feature_plot_xticks(float(self.nhop)/float(self.sample_rate))
                    P.xlabel('Time (secs)')
                    P.ylabel('Cepstral coeffient')
        elif feature == 'lcqft':
            if not self._have_lcqft:
                print "Error: must extract LCQFT first"
            else:
                feature_plot(self.LCQFT, normalize, dbscale, norm, title_string="LCQFT",interp=interp,nofig=nofig)
                if labels:
                    self._feature_plot_xticks(float(self.nhop)/float(self.sample_rate))                   
                    P.yticks(P.arange(0,self._cqtN,self.nbpo), (self.lo*2**(P.arange(0,self._cqtN,self.nbpo)/self.nbpo)).round(1))
        elif feature == 'hcqft':
            if not self._have_hcqft:
                print "Error: must extract HCQFT first"
            else:
                feature_plot(self.HCQFT, normalize, dbscale, norm, title_string="HCQFT",interp=interp,nofig=nofig)
                if labels:
                    self._feature_plot_xticks(float(self.nhop)/float(self.sample_rate))
                    P.yticks(P.arange(0,self._cqtN,self.nbpo), (self.lo*2**(P.arange(0,self._cqtN,self.nbpo)/self.nbpo)).round(1))
                    P.xlabel('Time (secs)')
                    P.ylabel('Frequency (Hz)')
        elif feature == 'chroma' or feature == 'hchroma':
            if not self._have_chroma:
                print "Error: must extract CHROMA first"
            else:
                feature_plot(self.CHROMA, normalize, dbscale, norm, title_string="CHROMA",interp=interp,nofig=nofig)
                if labels:
                    self._feature_plot_xticks(float(self.nhop)/float(self.sample_rate))
                    P.yticks(P.arange(0,self.nbpo,self.nbpo/12.),['C','C#','D','Eb','E','F','F#','G','G#','A','Bb','B'])
                    P.xlabel('Time (secs)')
                    P.ylabel('Pitch Class')
        else:
            print "Unrecognized feature, skipping plot: ", feature

    def _feature_plot_xticks(self, scale):
        x = P.plt.xticks()[0]
        P.plt.xticks(x[1:-1], numpy.array(x[1:-1]*scale,dtype='float32').round(1))
        P.xlabel('Time (s)')
        P.axis('tight')

    def _feature_plot_yticks(self, scale):
        y = P.plt.yticks()[0]
        P.plt.yticks(y[1:-1], numpy.array(y[1:-1]*scale,dtype='float32').round(1))
        P.axis('tight')
        
    def _stft_specgram(self):
        if not self._have_x:
            print "Error: You need to load a sound file first: use self.load_audio('filename.wav')\n"
            return False
        else:
            fp = self._check_feature_params()
            self.STFT=P.mlab.specgram(self.x, NFFT=self.nfft, noverlap=self.nfft-self.nhop)[0]
            self.STFT/=P.sqrt(self.nfft)
            self._have_stft=True
        if self.verbosity:
            print "Extracted STFT: nfft=%d, hop=%d" %(self.nfft, self.nhop)
        return True

    def _make_log_freq_map(self):
        """
        ::

            For the given ncoef (bands-per-octave) and nfft, calculate the center frequencies
            and bandwidths of linear and log-scaled frequency axes for a constant-Q transform.
        """
        fp = self.feature_params
        bpo = float(self.nbpo) # Bands per octave
        self._fftN = float(self.nfft)
        hi_edge = float( self.hi )
        lo_edge = float( self.lo )
        f_ratio = 2.0**( 1.0 / bpo ) # Constant-Q bandwidth
        self._cqtN = float( P.floor(P.log(hi_edge/lo_edge)/P.log(f_ratio)) )
        self._dctN = self._cqtN
        self._outN = float(self.nfft/2+1)
        if self._cqtN<1: print "warning: cqtN not positive definite"
        mxnorm = P.empty(self._cqtN) # Normalization coefficients        
        fftfrqs = self._fftfrqs #P.array([i * self.sample_rate / float(self._fftN) for i in P.arange(self._outN)])
        logfrqs=P.array([lo_edge * P.exp(P.log(2.0)*i/bpo) for i in P.arange(self._cqtN)])
        logfbws=P.array([max(logfrqs[i] * (f_ratio - 1.0), self.sample_rate / float(self._fftN)) 
                         for i in P.arange(self._cqtN)])
        #self._fftfrqs = fftfrqs
        self._logfrqs = logfrqs
        self._logfbws = logfbws
        self._make_cqt()

    def _make_cqt(self):
        """
        ::    

            Build a constant-Q transform (CQT) from lists of 
            linear center frequencies, logarithmic center frequencies, and
            constant-Q bandwidths.
        """
        fftfrqs = self._fftfrqs
        logfrqs = self._logfrqs
        logfbws = self._logfbws
        fp = self.feature_params
        ovfctr = 0.5475 # Norm constant so CQT'*CQT close to 1.0
        tmp2 = 1.0 / ( ovfctr * logfbws )
        tmp = ( logfrqs.reshape(1,-1) - fftfrqs.reshape(-1,1) ) * tmp2
        self.Q = P.exp( -0.5 * tmp * tmp )
        self.Q *= 1.0 / ( 2.0 * P.sqrt( (self.Q * self.Q).sum(0) ) )
        self.Q = self.Q.T

    def _make_dct(self):
        """
        ::
            Construct the discrete cosine transform coefficients for the 
            current size of constant-Q transform
        """
        DCT_OFFSET = self.lcoef
        nm = 1 / P.sqrt( self._cqtN / 2.0 )
        self.DCT = P.empty((self._dctN, self._cqtN))
        for i in P.arange(self._dctN):
          for j in P.arange(self._cqtN):
            self.DCT[ i, j ] = nm * P.cos( i * (2 * j + 1) * (P.pi / 2.0) / self._cqtN  )
        for j in P.arange(self._cqtN):
            self.DCT[ 0, j ] *= P.sqrt(2.0) / 2.0

    def _shift_insert(self,x, nex, hop):
        nex = nex.mean(1) if len(nex.shape) > 1 else nex # handle stereo
        x[:-hop] = x[hop:]
        x[-hop::] = nex

    def _stft(self):
        if not self._have_x:
            print "Error: You need to load a sound file first: use self.load_audio('filename.wav')"
            return False
        fp = self._check_feature_params()
        num_frames = len(self.x)
        self.STFT = P.zeros((self.nfft/2+1, num_frames), dtype='complex')
        win = P.ones(self.wfft) if self.window=='rect' else P.hamming(self.wfft)
        x = P.zeros(self.wfft)
        buf_frames = 0
        for k, nex in enumerate(self.x):
            self._shift_insert(x, nex, self.nhop)
            if self.nhop >= self.wfft - k*self.nhop : # align buffer on start of audio
                self.STFT[:,k-buf_frames]=P.rfft(win*x, self.nfft).T
            else:
                buf_frames+=1
        self.STFT = self.STFT / self.nfft
        self._fftfrqs = P.arange(0,self.nfft/2+1) * self.sample_rate/float(self.nfft)
        self._have_stft=True
        if self.verbosity:
            print "Extracted STFT: nfft=%d, hop=%d" %(self.nfft, self.nhop)
        self.inverse=self._istftm
        self.X = abs(self.STFT)
        if not self.magnitude:
            self.X = self.X**2
        return True

    def _istftm(self, X_hat=None, Phi_hat=None, pvoc=False, usewin=True):
        """
        :: 
            internal implementation
            Inverse short-time Fourier transform magnitude. Make a signal from a |STFT| transform.
            Uses phases from self.STFT if Phi_hat is None.

            Inputs:
            X_hat - N/2+1 magnitude STFT [None=abs(self.STFT)]
            Phi_hat - N/2+1 phase STFT   [None=exp(1j*angle(self.STFT))]
            pvoc - whether to use phase vocoder [False]      
            usewin - whether to use overlap-add [False]

            Returns:
             x_hat - estimated signal
        """
        if not self._have_stft:
                return None
        X_hat = P.np.abs(self.STFT) if X_hat is None else P.np.abs(X_hat)
        Phi_hat = P.exp( 1j * P.angle(self.STFT)) if Phi_hat is None else Phi_hat
        Phi_hat = self.pvoc(X_hat) if pvoc else Phi_hat
        fp = self._check_feature_params()
        self.X_hat = X_hat *  Phi_hat
        self.x_hat = self._overlap_add( 
            P.real(self.nfft * P.irfft(self.X_hat.T)), usewin=usewin)
        if self.verbosity:
            print "Extracted iSTFTM->self.x_hat"        
        return self.x_hat

    def istftm(self, X_hat=None, Phi_hat=None, pvoc=False, usewin=True):
        """
        ::

            Inverse short-time Fourier transform magnitude. Make a signal from a |STFT| transform.
            Uses phases from self.STFT if Phi_hat is None.

            Inputs:
            X_hat - N/2+1 magnitude STFT [None=abs(self.STFT)]
            Phi_hat - N/2+1 phase STFT   [None=angle(self.STFT)]
            pvoc - whether to use phase vocoder [False]      
            usewin - whether to use overlap-add [False]

            Returns:
             x_hat - estimated signal
        """
        return self._istftm(X_hat, Phi_hat, pvoc, usewin)

    def _power(self):
        if not self._stft():
            return False
        fp = self._check_feature_params()
        self.POWER=(P.absolute(self.STFT)**2).sum(0)
        self._have_power=True
        if self.verbosity:
            print "Extracted POWER"
        self.X=self.POWER
        return True

    def _cqft(self):
        """
        ::

            Constant-Q Fourier transform.
        """

        if not self._power():
            return False
        fp = self._check_feature_params()
        if self.intensify:
            self._cqft_intensified()
        else:
            self._make_log_freq_map()
            self.CQFT=P.sqrt(P.array(P.mat(self.Q)*P.mat(P.absolute(self.STFT)**2)))
            self._is_intensified=False
        self._have_cqft=True
        if self.verbosity:
            print "Extracted CQFT: intensified=%d" %self._is_intensified
        self.inverse=self.icqft
        self.X=self.CQFT
        return True

    def icqft(self, V_hat=None, pvoc=False, usewin=True):
        V_hat = self.CQFT if V_hat is None else V_hat
        return self._icqft(V_hat, pvoc, usewin)

    def _icqft(self, V_hat, pvoc=False, usewin=True):
        """
        ::

            Inverse constant-Q Fourier transform. Make a signal from a constant-Q transform.
        """
        if not self._have_cqft:
                return None
        fp = self._check_feature_params()
        X_hat = P.dot(self.Q.T, V_hat)
        if self.verbosity:
            print "iCQFT->X_hat"
        self.istftm(X_hat, pvoc=pvoc, usewin=usewin)
        return self.x_hat

    def pvoc(self, X_hat):
        """
        ::
            Phase vocoder re-synthesis of magnitude-only short-time Fourier transform
        """
        N = self.nfft
        W = self.wfft
        H = self.nhop
        SR = self.sample_rate
        dphi = H*P.atleast_2d(P.arange(N/2+1))*2*P.pi/N
        phs = P.mod(P.dot(dphi.T,P.atleast_2d(P.arange(X_hat.shape[1]))) + 
                    P.rand(dphi.shape[1],1)*2*P.pi,2*P.pi) - P.pi
        return phs
 
    def _overlap_add(self, X, usewin=True):
        wfft = self.wfft
        nhop = self.nhop
        if usewin:
            win = P.hamming(wfft)
        else:
            win = P.ones(wfft)
        x = P.zeros((X.shape[0] - 1)*nhop + wfft)
        for k in range(X.shape[0]):
            x[ k * nhop : k * nhop + wfft ] += X[ k, 0 : wfft ] * win
        return x
    
    def _cqft_intensified(self):
        """
        ::

            Constant-Q Fourier transform using only max abs(STFT) value in each band
        """
        if not self._have_stft:
            if not self._stft():
                return False
        self._make_log_freq_map()
        r,b=self.Q.shape
        b,c=self.STFT.shape
        self.CQFT=P.zeros((r,c))
        for i in P.arange(r):
            for j in P.arange(c):
                self.CQFT[i,j] = (self.Q[i,:]*P.absolute(self.STFT[:,j])).max()
        self._have_cqft=True
        self._is_intensified=True
        self.inverse=self.icqft
        self.X=self.CQFT
        return True

    def _mfcc(self): 
        """
        ::

            DCT of the Log magnitude CQFT 
        """
        fp = self._check_feature_params()
        if not self._cqft():
            return False
        self._make_dct()
        AA = P.log10(P.clip(self.CQFT,0.0001,self.CQFT.max()))
        self.MFCC = P.dot(self.DCT, AA)
        self._have_mfcc=True
        if self.verbosity:
            print "Extracted MFCC: lcoef=%d, ncoef=%d, intensified=%d" %(self.lcoef, self.ncoef, self.intensify)
        n=self.ncoef
        l=self.lcoef
        self.X=self.MFCC[l:l+n,:]
        return True

    def _lcqft(self):
        """
        ::

            Apply low-lifter to MFCC and invert to CQFT domain
        """
        fp = self._check_feature_params()
        if not self._mfcc():
            return False
        a,b = self.CQFT.shape
        a = (a-1)*2
        n=self.ncoef
        l=self.lcoef
        AA = self.MFCC[l:l+n,:] # apply Lifter
        self.LCQFT = 10**P.dot( self.DCT[l:l+n,:].T, AA )
        self._have_lcqft=True
        if self.verbosity:
            print "Extracted LCQFT: lcoef=%d, ncoef=%d, intensified=%d" %(self.lcoef, self.ncoef, self.intensify)
        self.inverse=self.icqft
        self.X=self.LCQFT
        return True

    def _hcqft(self):
        """
        ::

            Apply high lifter to MFCC and invert to CQFT domain
        """
        fp = self._check_feature_params()
        if not self._mfcc():
            return False
        a,b = self.CQFT.shape
        n=self.ncoef
        l=self.lcoef
        AA = self.MFCC[n+l:a,:] # apply Lifter
        self.HCQFT=10**P.dot( self.DCT[n+l:a,:].T, AA)
        self._have_hcqft=True
        if self.verbosity:
            print "Extracted HCQFT: lcoef=%d, ncoef=%d, intensified=%d" %(self.lcoef, self.ncoef, self.intensify)
        self.inverse=self.icqft
        self.X=self.HCQFT
        return True

    def _chroma(self):
        """
        ::
    
            Chromagram, like 12-BPO CQFT modulo one octave. Energy is folded onto first octave.
        """
        fp = self._check_feature_params()
        lo = self.lo
        self.lo = 63.5444 # set to quarter tone below C
        if not self._cqft():
            return False
        self.lo = lo # restore original lo edge
        a,b = self.CQFT.shape
        complete_octaves = a/self.nbpo # integer division, number of complete octaves
        #complete_octave_bands = complete_octaves * self.nbpo
        # column-major ordering, like a spectrogram, is in FORTRAN order
        self.CHROMA=P.zeros((self.nbpo,b))
        for k in P.arange(complete_octaves):
            self.CHROMA +=  self.CQFT[k*self.nbpo:(k+1)*self.nbpo,:]
        self.CHROMA = (self.CHROMA / complete_octaves)
        self._have_chroma=True
        if self.verbosity:
            print "Extracted CHROMA: intensified=%d" %self.intensify
        self.inverse=self.ichroma
        self.X=self.CHROMA
        return True

    def _chroma_hcqft(self):
        """
        ::

            Chromagram formed by high-pass liftering in cepstral domain, then usual self.nbpo-BPO folding.
        """
        fp = self._check_feature_params()
        if not self._hcqft():
            return False
        a,b = self.HCQFT.shape
        complete_octaves = a/self.nbpo # integer division, number of complete octaves
        #complete_octave_bands = complete_octaves * self.nbpo
        # column-major ordering, like a spectrogram, is in FORTRAN order
        self.CHROMA=P.zeros((self.nbpo,b))
        for k in P.arange(complete_octaves):
            self.CHROMA +=  self.HCQFT[k*self.nbpo:(k+1)*self.nbpo,:]
        self.CHROMA/= complete_octaves
        self._have_chroma=True
        if self.verbosity:
            print "Extracted HCQFT CHROMA: lcoef=%d, ncoef=%d, intensified=%d" %(self.lcoef, self.ncoef, self.intensify)
        self.inverse=self.ichroma
        self.X=self.CHROMA
        return True

    def _ichroma(self, V, pvoc=False):
        """
        ::
        
            Inverse chromagram transform. Make a signal from a folded constant-Q transform.
        """
        if not (self._have_hcqft or self._have_cqft):
            return None
        a,b = self.HCQFT.shape if self._have_hcqft else self.CQFT.shape
        complete_octaves = a/self.nbpo # integer division, number of complete octaves
        if P.remainder(a,self.nbpo):
            complete_octaves += 1
        X = P.repeat(V, complete_octaves, 0)[:a,:] # truncate if necessary
        X /= X.max()
        X *= P.atleast_2d(P.linspace(1,0,X.shape[0])).T # weight the spectrum
        self.x_hat = self._icqft(X, pvoc)
        return self.x_hat

    def ichroma(self, V, pvoc=False):
        """
        ::
        
            Inverse chromagram transform. Make a signal from a folded constant-Q transform.
        """        
        return self._ichroma(V, pvoc)

    def _extract_onsets(self):
        """
        ::
        
           The simplest onset detector in the world: power envelope derivative zero crossings +/-
        """
        fp = self._check_feature_params()
        if not self._have_power:
            return None
        dd = P.diff(P.r_[0,self.POWER])
        self.ONSETS = P.where((dd>0) & (P.roll(dd,-1)<0))[0]
        if self.verbosity:
            print "Extracted ONSETS"
        self._have_onsets = True
        return True

    def valid_features(self):
        """
        ::

            Valid feature extractors:
            stft - short-time Fourier transform
            power- per-frame power
            cqft - constant-Q Fourier transform
            mfcc - Mel-frequency cepstral coefficients
            lcqft - low-cepstra constant-Q Fourier transform
            hcqft - high-cepstra constant-Q Fourier transform
            chroma - self.nbpo-chroma-band pitch-class profile
        """

        print """Valid feature extractors:
        stft - short-time Fourier transform
        cqft - constant-Q Fourier transform
        mfcc - Mel-frequency cepstral coefficients
        lcqft - low-cepstra constant-Q Fourier transform
        hcqft - high-cepstra constant-Q Fourier transform
        chroma - self.nbpo-chroma-band pitch-class profile
        hchroma - high-quefrency self.nbpo-chroma-band pitch-class profile
        """
# Utility functions

def _normalize(x):
    """
    ::

        static method to copy array x to new array with min 0.0 and max 1.0
    """
    y=x.copy()
    y=y-P.np.min(y)
    y=y/P.np.max(y)
    return y

def feature_plot(M, normalize=False, dbscale=False, norm=False, title_string=None, interp='nearest', bels=False, nofig=False):
    """
    ::

        static method for plotting a matrix as a time-frequency distribution (audio features)
    """
    X = feature_scale(M, normalize, dbscale, norm, bels)
    if not nofig: P.figure()
    clip=-100.
    if dbscale or bels:
        if bels: clip/=10.
        P.imshow(P.clip(X,clip,0),origin='lower',aspect='auto', interpolation=interp)
    else:
        P.imshow(X,origin='lower',aspect='auto', interpolation=interp)
    if title_string:
        P.title(title_string)
    P.colorbar()

def feature_scale(M, normalize=False, dbscale=False, norm=False, bels=False):
    """
    ::

        Perform mutually-orthogonal scaling operations, otherwise return identity:
          normalize [False]
          dbscale  [False]
          norm      [False]        
    """
    if not (normalize or dbscale or norm or bels):
        return M
    else:
        X = M.copy() # don't alter the original
        if norm:
            X = X / P.tile(P.sqrt((X*X).sum(0)),(X.shape[0],1))
        if normalize:
            X = _normalize(X)
        if dbscale or bels:
            X = P.log10(P.clip(X,0.0001,X.max()))
            if dbscale:                
                X = 10*X
    return X

