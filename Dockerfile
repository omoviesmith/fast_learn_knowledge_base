# Use an official Python runtime as the parent image
FROM python:3.10-slim-buster

# Set environment variables
# ENV PYTHONUNBUFFERED=1
# ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for pycocotools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    python3-setuptools \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 

# Copy requirements.txt and install the Python dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Make the server listen on port 8000 at runtime
EXPOSE 8000

# Make sure Flask runs in production and listens on all interfaces
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080
ENV GUNICORN_WORKERS=4
# Define the command to start the app using gunicorn
# CMD ["gunicorn", "--bind", ":8000", "app:app"]
# Start the server
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 120 app:app

