from fastapi import FastAPI, UploadFile, File
import aiofiles
from pydub import AudioSegment
from os.path import isfile
from modules.sql_utils import select_db_row, establish_sqlite_db, create_db_table, insert_db_row
from modules.audio import NUM_FRAMES, SAMPLE_RATE, read_mfcc, sample_from_mfcc
from modules.model_evaluation import run_user_evaluation
import subprocess
from pydantic import BaseModel
import os


app = FastAPI()


class Is_it_needed_voice(BaseModel):
    is_it_needed_voice: bool


def validation(filename: str):

    user_row = select_db_row('users', '000000000')
    mfcc = user_row[1]

    score = run_user_evaluation(mfcc, filename)
    score = score[0]*100

    print(score)

    return True if score > 85 else False


@app.post("/")
async def post_endpoint(file: UploadFile=File(...)):

    filename = file.filename

    async with aiofiles.open(filename, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content) 
    
    wav_filename = ''.join(filename.split('.')[:-1])+'.wav'
    #!ffmpeg -i {filename} wav_filename
    subprocess.run(['ffmpeg', '-i', filename, '-loglevel', 'quiet', '-y', wav_filename])

    b = validation(wav_filename)

    os.remove(filename)
    os.remove(wav_filename)

    return Is_it_needed_voice(is_it_needed_voice=b)


@app.on_event("startup")
async def startup_event():

    global mfcc

    if not isfile('modules/SQL/sqlite.db'):

        establish_sqlite_db('users')
        mfcc = sample_from_mfcc(read_mfcc('example.wav', SAMPLE_RATE), NUM_FRAMES)
        insert_db_row('users', '000000000', mfcc)
        
    else:

        user_row = select_db_row('users', '000000000')
        mfcc = user_row[1]
