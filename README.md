## Speech-to-Text API
This repository contains a FastAPI application for converting audio files to text using OpenAI's Whisper model. The API provides a simple interface for uploading audio files and receiving transcribed text.

# Features
Automatic Speech Recognition: Transcribe audio files to text using the Whisper large-v3 model.
FastAPI: Lightweight and fast server for handling requests.
CUDA Support: Leverages GPU (if available) for faster processing.
Dockerized Deployment: Includes a Dockerfile for easy containerization.

# Steps
Clone the repository:

 ```bash
git clone https://github.com/IbrarArif/speech-to-text-api.git
cd speech-to-text-api

Install dependencies:

```bash
pip install -r requirements.txt

Run the application:

```bash
python main.py

Access the API at:
```bash
http://localhost:8000
