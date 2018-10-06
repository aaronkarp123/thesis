# We'll need numpy for some mathematical operations
import numpy as np

# matplotlib for displaying the output
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

# Librosa for audio
import librosa
# And the display module for visualization
import librosa.display

# Extract features
def get_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)

    # Make a new figure
    plt.figure(figsize=(12,4))

    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

    # Put a descriptive title on the plot
    plt.title('mel power spectrogram')

    # draw a color bar
    plt.colorbar(format='%+02.0f dB')

    # Make the figure layout compact
    plt.tight_layout()
    
    return log_S
    
def get_mfcc(y, sr):
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    
    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)
    
    mfcc        = librosa.feature.mfcc(S=log_S, n_mfcc=13)

    # How do they look?  We'll show each in its own subplot
    plt.figure(figsize=(12, 6))

    plt.subplot(3,1,1)
    librosa.display.specshow(mfcc)
    plt.ylabel('MFCC')
    plt.colorbar()
    
    plt.tight_layout()
    
    return mfcc

def get_rms(y):
    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rmse(S=S)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.semilogy(rms.T, label='RMS Energy')
    plt.xticks([])
    plt.xlim([0, rms.shape[-1]])
    plt.legend(loc='best')
    
    return rms

def get_spectral_centroid(y, sr):
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.semilogy(cent.T, label='Spectral centroid')
    plt.ylabel('Hz')
    plt.xticks([])
    plt.xlim([0, cent.shape[-1]])
    plt.legend()
    
    return cent

def get_all(y, sr):
    # Return [spectrogram, mfcc, rms, spectral centroid] in that order
    features = []
    
    #spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    features.append(log_S)
    
    #mfcc
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    features.append(mfcc)
    
    #rms
    rms = librosa.feature.rmse(y=y)
    features.append(rms)
    
    #spectral centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(cent)
    
    return features