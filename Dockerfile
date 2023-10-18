# Use an official Python runtime as a parent image
# We are using the slim-buster version to keep the image small and efficient
FROM python:3.8-slim-buster

# Set the working directory in the container to /app
# All the commands that follow in the Dockerfile will be run in this directory
WORKDIR /app

# Copy the current directory contents into the container at /app
# This will include all files in your project directory
COPY . /app

# Install any needed packages specified in requirements.txt
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
# This is the port your application will be served on
EXPOSE 80

# Define the command to run your app using CMD which keeps the container running.
# We use the uvicorn server, as it is recommended by FastAPI
# The --reload flag enables hot reloading which is useful during development
CMD ["uvicorn", "main:app", "--reload"]