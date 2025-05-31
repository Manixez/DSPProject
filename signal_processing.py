import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d

def bandpass_filter(data, fs, lowcut, highcut, order=3):
    b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)
    return filtfilt(b, a, data)

def estimate_bpm(signal, fs, lowcut, highcut):
    if len(signal) < fs * 3:
        return 0.0, np.zeros_like(signal)
    filtered = bandpass_filter(signal, fs, lowcut, highcut)
    filtered = uniform_filter1d(filtered, size=5)
    peaks, _ = find_peaks(filtered, distance=fs//2)
    bpm = 60 * len(peaks) / (len(filtered) / fs)
    return bpm, filtered
