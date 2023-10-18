# Speaker Verification Project

This project is a speaker verification system built with Python. It uses the power of deep learning to verify the identity of a speaker from audio input. 
Please make sure you have the `example.wav` file, which is the source sample of voice that is needed to be verified.

## Dependencies

This project is built on Python 3.8. The dependencies required to run this project are specified in the `requirements.txt` file.

## Installation

To install the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/cbcyd/speaker_verification.git
cd speaker_verification
pip install -r requirements.txt
```


Convert sample of voice to `example.wav`, for example using ffmpeg: `ffmpeg -i sample.mp3 example.wav`. Copy `example.wav`to root directory of project.

## Running the Project

To run the project, use the following command:

```bash
uvicorn main:app --reload
```

This command will start the Uvicorn server with hot-reloading enabled.

## Running with Docker

You can also run this project using Docker. First, build the Docker image:

```bash
docker build -t speaker-verification .
```

Then, run the Docker container:

```bash
docker run -p 80:80 speaker-verification
```

## Based on

This project is based on the following projects:

- [Speaker Verification by OnTrack-UG-Squad](https://github.com/OnTrack-UG-Squad/speaker-verification)
- [Deep Speaker by Philipperemy](https://github.com/philipperemy/deep-speaker)