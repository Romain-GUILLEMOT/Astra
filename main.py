import time

from elevenlabs import save
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import FileResponse
import requests
import openai
import os
from elevenlabs.client import ElevenLabs
from starlette.responses import HTMLResponse
from dotenv import load_dotenv


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

@app.post("/speech-to-action/")
async def speech_to_action(file: UploadFile = File(...)):
    audio_content = await file.read()
    response = openai.Audio.transcribe(
        model="whisper-1",
        file=audio_content
    )

    if 'text' in response:
        text = response['text']
    else:
        return {"error": "Could not understand the audio"}

    return await process_text(text)


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
        model=os.getenv("GPTVERSION"),
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