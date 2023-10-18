# Importing required libraries
from fastapi import FastAPI, UploadFile, File
import aiofiles
from os.path import isfile
from modules.audio import NUM_FRAMES, SAMPLE_RATE, read_mfcc, sample_from_mfcc
from modules.model_evaluation import run_user_evaluation
import subprocess
from pydantic import BaseModel
import os

# Initialization of FastAPI
app = FastAPI()

# Creating a pydantic BaseModel for response
class Is_it_needed_voice(BaseModel):
    is_it_needed_voice: bool

# Function to validate audio file using evaluation model
def validation(filename: str):
    # Run user evaluation
    score = run_user_evaluation(mfcc, filename)
    score = score[0] * 100

    # Print the score
    print(score)

    # Return true if score > 85 else false
    return True if score > 85 else False

# POST API endpoint
@app.post("/")
async def post_endpoint(file: UploadFile=File(...)):

    # Get the filename from the uploaded file
    filename = file.filename

    # Open the file and write the content into it
    async with aiofiles.open(filename, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    # Convert filename to .wav format
    wav_filename = ''.join(filename.split('.')[:-1]) + '.wav'

    # Convert original file to .wav using ffmpeg
    subprocess.run(['ffmpeg', '-i', filename, '-loglevel', 'quiet', '-y', wav_filename])

    # Validate the audio using evaluation function
    b = validation(wav_filename)

    # Remove the created files
    os.remove(filename)
    os.remove(wav_filename)

    # Return the validation result
    return Is_it_needed_voice(is_it_needed_voice=b)

# Startup event which occurs when the server starts
@app.on_event("startup")
async def startup_event():
    global mfcc
    # Reading the mfcc sample from example.wav file
    mfcc = sample_from_mfcc(read_mfcc('example.wav', SAMPLE_RATE), NUM_FRAMES)