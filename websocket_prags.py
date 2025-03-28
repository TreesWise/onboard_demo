import os
import io
import wave
from datetime import datetime
import requests
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from helper import *

# Constants
api_key=os.getenv("OPENAI_API_KEY")
TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions"

RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 2 bytes per sample (int16)
SEGMENT_DURATION_SEC = 5
OVERLAP_DURATION_SEC = 1
SEGMENT_SIZE = RATE * SEGMENT_DURATION_SEC * SAMPLE_WIDTH
OVERLAP_SIZE = RATE * OVERLAP_DURATION_SEC * SAMPLE_WIDTH

# Dictionary to store conversation history per client
client_histories = {}

# App
app = FastAPI()

@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    print("🎙️ Client connected")

    buffer = bytearray()
    client_histories[websocket] = []  # Initialize conversation history for this client

    try:
        while True:
            if websocket.client_state != WebSocketState.CONNECTED:
                break

            chunk = await websocket.receive_bytes()
            buffer.extend(chunk)

            if len(buffer) >= SEGMENT_SIZE:
                segment = bytes(buffer[:SEGMENT_SIZE])
                buffer = buffer[SEGMENT_SIZE - OVERLAP_SIZE:]

                # Save the received audio segment for debugging
                # debug_filename = f"received_audio_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.wav"
                # with wave.open(debug_filename, 'wb') as wf:
                #     wf.setnchannels(CHANNELS)
                #     wf.setsampwidth(SAMPLE_WIDTH)
                #     wf.setframerate(RATE)
                #     wf.writeframes(segment)
                # print(f"🔍 Saved received audio to {debug_filename}")
                # Transcribe the segment

                transcription = transcribe_audio_from_bytes(segment)

                if transcription:
                    # Append transcription to user's conversation history
                    client_histories[websocket].append(transcription)

                    # Combine all past transcriptions into a single text block
                    full_history = " ".join(client_histories[websocket])

                    # Process the full history and send it
                    labeled_transcript, result_json = process_transcript(full_history)
                    print("result_json------>",result_json)
                    await websocket.send_text(json.dumps(result_json))
                else:
                    await websocket.send_text(json.dumps({"error": "Transcription failed"}))

    except WebSocketDisconnect:
        print("❌ Client disconnected")
        client_histories.pop(websocket, None)  # Remove history when client disconnects

    except Exception as e:
        print(f"WebSocket error: {e}")

def transcribe_audio_from_bytes(audio_bytes: bytes) -> str:
    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"temp_{timestamp}.wav"

        with io.BytesIO() as buffer:
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(RATE)
                wf.writeframes(audio_bytes)

            buffer.seek(0)

            files = {"file": (filename, buffer, "audio/wav")}
            headers = {"Authorization": f"Bearer {api_key}"}
            data = {"model": "whisper-1"}

            response = requests.post(TRANSCRIPTION_URL, headers=headers, files=files, data=data)

            if response.status_code == 200:
                return response.json().get("text", "")
            else:
                print("Transcription API error:", response.text)
                return None
    except Exception as e:
        print("Error during transcription:", e)
        return None

# Run with Uvicorn
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI WebSocket server on ws://localhost:8000/ws/audio")
    uvicorn.run(app, host="0.0.0.0", port=8000)
