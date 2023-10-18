# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# Also install libsndfile and ffmpeg using apt-get
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define the command to run your app using CMD which keeps the container running.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]