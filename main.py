from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import uvicorn
import librosa
import soundfile as sf
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and processor
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-tiny"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# API endpoint to upload audio and get the transcribed text
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        # Load the audio file using librosa
        audio, sr = librosa.load(temp_path, sr=16000)

        # Convert to a format that the model can process (in case the file needs reformatting)
        processed_path = temp_path  # Reuse temp file if format is already correct
        sf.write(processed_path, audio, 16000)

        # Pass the processed audio to the pipeline
        result = pipe(processed_path)

        # Remove the temp file after processing
        os.remove(temp_path)

        # Return the transcribed text
        return {"text": result["text"]}
    
    except Exception as e:
        # Clean up temp file in case of error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the speech-to-text API!"}

# Running FastAPI with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
