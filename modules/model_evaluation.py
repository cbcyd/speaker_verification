from modules.rescnn_model import DeepSpeakerModel
from os.path import join, abspath, dirname

import numpy as np

from modules.audio import (
    NUM_FRAMES,
    SAMPLE_RATE,
    read_mfcc,
    sample_from_mfcc,
)


MODEL_PATH = join(
    abspath(dirname(__file__)), "models", "ResCNN_triplet_training_checkpoint_265.h5"
)


def batch_cosine_similarity(x1, x2):
    """ https://en.wikipedia.org/wiki/Cosine_similarity """

    mul = np.multiply(x1, x2)
    s = np.sum(mul, axis=1)
    return s


def run_model_evaluation(audio_input, model, raw_audio=False):
    """run_model_evaluation.

    Parameters
    ----------
    audio_input : str, Path-like
        Path to audio input for evaluation on prediction value.
    model : DeepSpeakerModel
        Instantiated model with required weights for speaker verification.
    raw_audio : bool
        Boolean value on whether the input audio path is mfcc or raw wav/flac.
    """
    if raw_audio is True:
        mfcc = sample_from_mfcc(read_mfcc(audio_input, SAMPLE_RATE), NUM_FRAMES)
    else:
        mfcc = audio_input
    prediction = model.rescnn.predict(np.expand_dims(mfcc, axis=0))
    return prediction


def run_user_evaluation(enrolment_mfcc, input_audio):
    """run_user_evaluation.

    Instanstiate project model and run evaulation on parameter inputs.

    Parameters
    ----------
    enrolment_mfcc : numpy.array
        MFCC array from sqlite user table for model evaluation.
    input_audio : str, Path-like
        Path to audio input for evaluation on prediction value.
    """
    model = DeepSpeakerModel()
    model.rescnn.load_weights(MODEL_PATH, by_name=True)
    enrolment_evaluation = run_model_evaluation(enrolment_mfcc, model)
    input_evaluation = run_model_evaluation(input_audio, model, raw_audio=True)

    return batch_cosine_similarity(enrolment_evaluation, input_evaluation)