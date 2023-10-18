# Use an official Python runtime as the parent image
FROM python:3.11-slim-buster

# Set environment variables
# ENV PYTHONUNBUFFERED=1
# ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install the dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Make the server listen on port 8000 at runtime
EXPOSE 8000

# Define the command to start the app using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]