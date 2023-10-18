# Libraries required for MFCC feature extraction, frame normalization, and reading audio files
import librosa
import numpy as np
from python_speech_features import fbank
from random import choice


# Constant parameters for audio processing and MFCC extraction
SAMPLE_RATE = 16000
NUM_FRAMES = 300
NUM_FBANKS = 64


# Function to normalize audio frames
def normalize_frames(m, epsilon=1e-12):
    # This function performs frame normalization using mean and standard deviation
    return [(v - np.mean(v)) / max(np.std(v), epsilon) for v in m]


# Function to calculate MFCC and filter bank features from audio signal
def mfcc_fbank(signal: np.array, sample_rate: int):  # 1D signal array.
    # The function calculates and returns the MFCC with shape (num_frames, n_filters, 3).
    filter_banks, energies = fbank(signal, samplerate=sample_rate, nfilt=NUM_FBANKS)
    frames_features = normalize_frames(filter_banks)
    return np.array(frames_features, dtype=np.float32)


# Function to pad MFCC features if their length is less than the max_length
def pad_mfcc(mfcc, max_length):
    # This function uses zero-padding to make the length of all MFCC equal
    if len(mfcc) < max_length:
        mfcc = np.vstack(
            (mfcc, np.tile(np.zeros(mfcc.shape[1]), (max_length - len(mfcc), 1)))
        )
    return mfcc


# Function to read audio file
def audio_read(filename, sample_rate=SAMPLE_RATE):
    # It loads a mono audio file and ensures the correct sample rate
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True, dtype=np.float32)
    assert sr == sample_rate
    return audio


# Function to read and process MFCC from an audio file
def read_mfcc(input_filename, sample_rate):
    # The function removes silence from the audio before calculating the MFCC
    audio = audio_read(input_filename, sample_rate)
    energy = np.abs(audio)
    silence_threshold = np.percentile(energy, 95)
    offsets = np.where(energy > silence_threshold)[0]
    audio_voice_only = audio[offsets[0] : offsets[-1]]
    mfcc = mfcc_fbank(audio_voice_only, sample_rate)
    return mfcc


# Function to sample from MFCC features
def sample_from_mfcc(mfcc, max_length):
    # This function either randomly samples from MFCC features or pads them
    if mfcc.shape[0] >= max_length:
        r = choice(range(0, len(mfcc) - max_length + 1))
        s = mfcc[r : r + max_length]
    else:
        s = pad_mfcc(mfcc, max_length)
    return np.expand_dims(s, axis=-1)