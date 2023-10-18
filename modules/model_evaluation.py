# Import the necessary modules
from modules.rescnn_model import DeepSpeakerModel
from os.path import join, abspath, dirname
import numpy as np
from modules.audio import (
    NUM_FRAMES,
    SAMPLE_RATE,
    read_mfcc,
    sample_from_mfcc,
)

# Define the path to the model
MODEL_PATH = join(
    abspath(dirname(__file__)), "models", "ResCNN_triplet_training_checkpoint_265.h5"
)

def batch_cosine_similarity(x1, x2):
    """
    Compute cosine similarity between two batches
    https://en.wikipedia.org/wiki/Cosine_similarity
    """
    # Multiply elements and sum
    mul = np.multiply(x1, x2)
    s = np.sum(mul, axis=1)

    return s


def run_model_evaluation(audio_input, model, raw_audio=False):
    """
    Run a model evaluation for a given audio input.

    Parameters
    ----------
    audio_input : str, Path-like
        Path to audio input for evaluation.
    model : DeepSpeakerModel
        An instance of model for speaker verification.
    raw_audio : bool
        Flag indicating whether the input is raw audio. Default is False
    """
    # Check the type of input audio and extract MFCC (Mel frequency cepstral coefficients) accordingly
    if raw_audio is True:
        mfcc = sample_from_mfcc(read_mfcc(audio_input, SAMPLE_RATE), NUM_FRAMES)
    else:
        mfcc = audio_input

    # Make a prediction
    prediction = model.rescnn.predict(np.expand_dims(mfcc, axis=0))

    return prediction


def run_user_evaluation(enrolment_mfcc, input_audio):
    """
    Run the speaker verification model for a given input audio and a set of enrolment MFCCs.

    Parameters
    ----------
    enrolment_mfcc : numpy.array
        An array of MFCCs taken from sqlite user table.
    input_audio : str, Path-like
        Path to the input audio for evaluation.
    """
    # Create an instance of the DeepSpeakerModel and load the weights
    model = DeepSpeakerModel()
    model.rescnn.load_weights(MODEL_PATH, by_name=True)

    # Run the model evaluations
    enrolment_evaluation = run_model_evaluation(enrolment_mfcc, model)
    input_evaluation = run_model_evaluation(input_audio, model, raw_audio=True)

    # Compute and return the cosine similarity between enrolment evaluation and input evaluation
    return batch_cosine_similarity(enrolment_evaluation, input_evaluation)