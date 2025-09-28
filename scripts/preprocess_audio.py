# preprocess_audio.py (đã sửa lỗi highcut)
import librosa
import numpy as np
from scipy.signal import butter, lfilter
import noisereduce as nr

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs 
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = lfilter(b, a, data)
    return y

def normalize_audio(data):
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    return data

def apply_fft(audio_data, fs=16000):
    fft_data = np.fft.fft(audio_data)
    fft_freq = np.fft.fftfreq(len(fft_data), d=1/fs)
    positive_freqs = fft_freq[:len(fft_data)//2]
    positive_fft_data = np.abs(fft_data[:len(fft_data)//2])
    return positive_freqs, positive_fft_data

def extract_mfcc(audio_data, sr=16000, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.mean(mfcc, axis=1)
    return mfcc

def reduce_noise(audio_data, sr=16000):
    reduced_noise = nr.reduce_noise(y=audio_data, sr=sr)
    return reduced_noise

def preprocess_audio(y, fs=16000):
    lowcut = 85
    # THAY ĐỔI Ở ĐÂY: Giảm highcut đi 1 đơn vị để nhỏ hơn tần số Nyquist
    highcut = 7999 
    
    y_cleaned = reduce_noise(y, sr=fs)
    y_cleaned = np.nan_to_num(y_cleaned)
    y_filtered = butter_bandpass_filter(y_cleaned, lowcut=lowcut, highcut=highcut, fs=fs)
    y_normalized = normalize_audio(y_filtered)
    positive_freqs, positive_fft_data = apply_fft(y_normalized, fs=fs)
    mfcc = extract_mfcc(y_normalized, sr=fs)
    return np.concatenate((positive_fft_data, mfcc))