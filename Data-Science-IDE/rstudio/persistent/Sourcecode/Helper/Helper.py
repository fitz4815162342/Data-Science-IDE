import sys
import time
import inspect
from os import sep
import numpy as np
import pandas as pd
import pickle
import cv2
import face_recognition
import librosa
import librosa.display
import matplotlib.pyplot as plt
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

stripline = "====================================================================================================="


class bcolors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[31m'
    YELLOW = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    BGRED = '\033[41m'
    WHITE = '\033[37m'



def Dictionary2JSON(dictionary, save_path):
    try:
        with open(save_path, 'w') as fp:
            json.dump(dictionary, fp)
            print(f"[Logger] Dictionary saved at: {save_path}", flush=True)
    except Exception as ex:
        print(f"[Logger] Exception:\n{'Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(ex).__name__, ex}\n", flush=True)


def JSON2Dictionary(save_path):
    try:
        with open(save_path, 'r') as fp:
            dictionary_data = json.load(fp)
            return dictionary_data
    except Exception as ex:
        print(f"[Logger] Exception:\n{'Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(ex).__name__, ex}\n", flush=True)


def serialize(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"\n[serialize] serialized at: {path}")


def deserialize(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"[deserialize] deserialized from: {path}")
    return data


def feature_extraction(samples, hop_length, frame_length, debug):
    for sample in samples:
        try:
            # populate AudioSample object with features
            sample.features["MFCC"] = librosa.feature.mfcc(y=sample.time_series, sr=sample.sampling_rate)
            sample.features["ZeroCrossingRate"] = librosa.feature.zero_crossing_rate(sample.time_series)
            energy = np.array([sum(abs(sample.time_series[i:i + frame_length] ** 2)) for i in
                               range(0, len(sample.time_series), hop_length)])
            sample.features["Energy"] = energy
            S, phase = librosa.magphase(librosa.stft(sample.time_series))
            sample.features["SpectralRollOff"] = librosa.feature.spectral_rolloff(S=S, sr=sample.sampling_rate)
            onset_env = librosa.onset.onset_strength(y=sample.time_series, sr=sample.sampling_rate)
            sample.features["SpectralFlux"] = onset_env
            sample.features["ChromaFeatures"] = librosa.feature.chroma_stft(y=sample.time_series, sr=sample.sampling_rate)
            pitches, magnitudes = librosa.piptrack(y=sample.time_series, sr=sample.sampling_rate)
            sample.features["Pitch"] = pitches
            if debug:
                border_str(f"Features for {sample.sample_name}")
                print(f'Dim MFCC: {sample.features["MFCC"].shape}', flush=True)
                print(f'Dim ZeroCrossingRate: {sample.features["ZeroCrossingRate"].shape}', flush=True)
                print(f'Dim Energy: {sample.features["Energy"].shape}', flush=True)
                print(f'Dim SpectralRollOff: {sample.features["SpectralRollOff"].shape}', flush=True)
                print(f'Dim SpectralFlux: {sample.features["SpectralFlux"].shape}', flush=True)
                print(f'Dim ChromaFeatures: {sample.features["ChromaFeatures"].shape}', flush=True)
                print(f'Dim Pitch: {sample.features["Pitch"].shape}', flush=True)
                with pd.option_context("display.max_rows", 8, "display.max_columns", 5):
                    print("\n", flush=True)
                    border_str("Description MFCC")
                    print(f'\n{pd.DataFrame(sample.features["MFCC"]).describe()}\n', flush=True)
                    border_str("Description ZeroCrossingRate")
                    print(f'\n{pd.DataFrame(sample.features["ZeroCrossingRate"]).describe()}\n', flush=True)
                    border_str("Description Energy")
                    print(f'\n{pd.DataFrame(sample.features["Energy"]).describe()}\n', flush=True)
                    border_str("Description SpectralRollOff")
                    print(f'\n{pd.DataFrame(sample.features["SpectralRollOff"]).describe()}\n', flush=True)
                    border_str("Description SpectralFlux")
                    print(f'\n{pd.DataFrame(sample.features["SpectralFlux"]).describe()}\n', flush=True)
                    border_str("Description ChromaFeatures")
                    print(f'\n{pd.DataFrame(sample.features["ChromaFeatures"]).describe()}\n', flush=True)
                    border_str("Description Pitch")
                    print(f'\n{pd.DataFrame(sample.features["Pitch"]).describe()}\n', flush=True)
                    print(stripline)
        except Exception as ex:
            pass
    samples = [i for i in samples if i]
    return samples



def banner():
    print(bcolors.GREEN + bcolors.BOLD)
    print("""
 █████╗ ██████╗ ███████╗ █████╗     ███████╗ ██╗
██╔══██╗██╔══██╗██╔════╝██╔══██╗    ██╔════╝███║
███████║██████╔╝█████╗  ███████║    ███████╗╚██║
██╔══██║██╔══██╗██╔══╝  ██╔══██║    ╚════██║ ██║
██║  ██║██║  ██║███████╗██║  ██║    ███████║ ██║
╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝    ╚══════╝ ╚═╝
    """.format(), flush=True)
    print(bcolors.ENDC)


def train_test_split_list(lst, fraction):
    dataLength = len(lst)
    testLength = round((fraction * dataLength))
    trainLength = dataLength - testLength
    train = list()
    test = list()
    counter = 0
    for idx in range(0, dataLength, 1):
        if (counter < trainLength):
            train.append(lst[idx])
        else:
            test.append(lst[idx])
        counter = counter + 1
    return train, test


def border_str(msg):
    width = len(msg)
    print('+-' + '-' * width + '-+')
    for line in chunk_str(msg, width):
        print('| {0:^{1}} |'.format(line, width))
    print('+-' + '-' * (width) + '-+')


def chunk_str(msg, width):
    return (msg[0 + i:width + i] for i in range(0, len(msg), width))


def timestamp():
    current_time = time.localtime()
    return time.strftime('%Y%m%d%H%M%S', current_time)
  
  
  
