import librosa
import numpy as np
from os import path
from pydub import AudioSegment
import io
import soundfile as sf
import time


# a function to get a pooled version of the signal frane
def get_char(array):
    # returns the max, min, and mean value of a window
    a1 = np.max(array, 2)
    a2 = np.min(array, 2)
    a3 = np.mean(array, 2)
    return np.concatenate([a1, a2, a3],axis=1)


# a function to extract features from the frame
def make_features(y, sr, window_size=30):
    # create the features
    features = [
            librosa.feature.tonnetz(y=y, sr=sr),
            librosa.feature.chroma_cqt(y=y, sr=sr),
            librosa.feature.mfcc(y=y, sr=sr),
            librosa.feature.spectral_contrast(S=np.abs(librosa.stft(y)), sr=sr),
    ]
    # find number of windows that fit in sample
    y_size = features[0].shape[1]
    n_windows = y_size//window_size
    # find difference between sample split by windows
    gap = y_size-window_size*n_windows
    # pad both sides with gap
    start = int(gap/2)
    end = int(start+n_windows*window_size)
    pooled_feature = []
    # for each feature type pool the window values
    for feature in features:
        feature = feature[:,start:end]
        feature = np.split(feature, n_windows, 1)
        feature=np.array(feature)
        # perform pooling
        feature = get_char(feature)
        pooled_feature.append(feature)
    # return vector of values
    return np.concatenate(pooled_feature, axis=1)


# function for converting .mp3 file and encoding into windows
def split_encode_mp3(src, dir):
    # create the path
    src = dir+src
    # retrieve the sound
    sound = AudioSegment.from_mp3(src)
    # convert to a wav array output
    wav_form = sound.export(format="wav")
    # read the data
    data, samplerate = sf.read(io.BytesIO(wav_form.read()))
    # convert the raw data into features
    data = make_features(data, samplerate)
    return data


# a function to get the features from the audio paths
def get_xy(x_values, y_values, n_samples=None):
    # set the limit of samples to take
    if not n_samples:
        n_samples = len(x_values)
    data = []
    labels = []
    times = []
    users = []
    # iterate through values
    for i, p in enumerate(x_values[:n_samples]):
        try:
            start_time = time.time()
            # get the X and y values
            y = y_values[i]
            X = split_encode_mp3(p)
            # add to the outputs
            for x in X:
                data.append(x)
                labels.append(y)
                # add file path to the user as a number
                # this way they can be pooled at the end
                users.append(i)
            end_time = time.time()
            total_time = end_time-start_time
            times.append(total_time)
        except Exception as e:
            print(i, p, e)
    return (data, labels, times, users)
    