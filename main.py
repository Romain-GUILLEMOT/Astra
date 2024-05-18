import time

import httpx
from elevenlabs import save
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import FileResponse
import openai
import os
from elevenlabs.client import ElevenLabs
from starlette.responses import HTMLResponse
from dotenv import load_dotenv
import whisper
import uuid

load_dotenv()

client = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_KEY")
)

app = FastAPI()
clientTEXT = openai.OpenAI(
    api_key=os.getenv("OPENAI_KEY")
)

HOME_ASSISTANT_URL = os.getenv("HOME_ASSITANT_URL")
HOME_ASSISTANT_TOKEN = os.getenv("HOME_ASSISTANT_TOKEN")
LANGUAGE = os.getenv("LANGUAGE")

async def check_openai_token() -> bool:
    api_key = os.getenv("OPENAI_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(os.getenv("OPENAI_KEY"), headers=headers)
        return response.status_code == 200

async def check_home_assistant_token() -> bool:
    headers = {
        "Authorization": f"Bearer {HOME_ASSISTANT_TOKEN}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(HOME_ASSISTANT_URL + "states", headers=headers)
        return response.status_code == 200

@app.on_event("startup")
async def startup_event():
    first = False
    if not os.path.exists("/tmp/speech"):
        os.makedirs("/tmp/speech")
        first = True
    if not os.path.exists("/tmp/result"):
        os.makedirs("/tmp/result")
        first = True
    if first:
        print("First launch detected. I'm going to download all AI models before continue...")
        whisper.load_model(os.getenv("WHISPERMODEL"))
        print("Model downloaded successfully.")

    print("Test of HomeAssitant connexion")
    connected = await check_home_assistant_token()
    if not connected:
        print("Failed to connect to HomeAssistant. Please check your configuration.")
        exit(1)

    print("Test of OpenAI connexion")
    print("Connected to OpenAI successfully.")
    connected = await check_openai_token()
    if not connected:
        print("Failed to connect to OpenAI. Please check your configuration.")
        exit(1)
    print("Connected to OpenAI successfully.")


@app.post("/speech-to-action/")
async def speech_to_action(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    temp_file_path = f"{file_id}.wav"

    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file.file.read())

    model = whisper.load_model(os.getenv("WHISPERMODEL"))
    result = model.transcribe(temp_file_path, language=LANGUAGE)

    os.remove(temp_file_path)

    return await process_text(result)


@app.post("/text-to-action/")
async def text_to_action_post(text: str = Query(...)):
    return await process_text(text)


@app.get("/text-to-action/")
async def text_to_action_get(text: str = Query(...)):
    return await process_text(text)


async def process_text(text: str):
    # Étape 2 : Envoi du texte à l'API GPT-4
    prompt = f"""Répondre toujours sous le format suivant :
Réponse: [Réponse de l'IA à la première personne simple pas trop long]
Action: [Action à effectuer en un mot sans maj si aucune action n'est fournit renvoie null]
Pièce: [Pièce concernée en un mot sans maj  si aucune piece n'est fournit renvoie null]
Composant: [Composant concernée en un mot sans maj  si aucune piece n'est fournit renvoie null]
Couleur: [Couleur concernée en un mot sans maj  si aucune piece n'est fournit renvoie null]

Demande: {text}
"""
    start_time = time.time()

    response = clientTEXT.chat.completions.create(
        model=os.getenv("GPTMODEL"),
        messages=[
            {"role": "system", "content": "Vous êtes un assistant vocal utile. Tes réponses doivent est consive courte et simple. Indique les piece de l'action ect. Tu dois faire toutes les actions que l'on te demande"},
            {"role": "user", "content": prompt}
        ]
    )
    end_time = time.time()
    request_time = end_time - start_time
    request_time_ms = request_time * 1000

    gpt4_result = response.choices[0].message.content.strip()
    print(gpt4_result)
    print(request_time_ms)
    try:
        response_lines = gpt4_result.split('\n')
        response_message = response_lines[0].replace('Réponse: ', '').strip()
        action = response_lines[1].replace('Action: ', '').strip()
        room = response_lines[2].replace('Pièce: ', '').strip()
    except (IndexError, KeyError):
        return {"error": "Unexpected response format from GPT-4"}

    #  home_assistant_response = requests.post(
    #      f"{HOME_ASSISTANT_URL}{action}",
    #      headers={
    #          "Authorization": f"Bearer {HOME_ASSISTANT_TOKEN}",
    #          "Content-Type": "application/json"
    #      },
    #     json={"entity_id": f"{room}.{action}"}
    #  )

    # if home_assistant_response.status_code != 200:
    #    return {"error": "Failed to send request to Home Assistant"}

    start_timeAUDIO = time.time()
    audio_file = "response.wav"
    audio = client.generate(
        text=response_message,
        voice="Charlie",
        model="eleven_multilingual_v2"
    )
    end_timeAUDIO = time.time()
    request_timeAUDIO = end_timeAUDIO - start_timeAUDIO
    request_time_msAUDIO = request_timeAUDIO * 1000
    print("Audio time " + str(request_time_msAUDIO))
    print("IA time " + str(request_time_ms))
    save(audio, audio_file)

    return FileResponse(path=audio_file, media_type='audio/wav', filename=audio_file)

@app.get("/")
async def form_page():
    html_content = """
    <html>
        <body>
            <h1>Assistant Vocal</h1>
            <form action="/text-to-action/" method="get">
                <label for="text">Demande :</label>
                <input type="text" id="text" name="text">
                <button type="submit">Envoyer</button>
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)