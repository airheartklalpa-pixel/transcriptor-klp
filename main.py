import math
import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

app = FastAPI(title="Backend Transcriptor KLP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MINUTOS_POR_ANUNCIO = 5
ANUNCIOS_RESPALDO = 2

print("Cargando motor Whisper...")
modelo = WhisperModel("tiny", device="cpu", compute_type="int8")
print("¡Motor de transcripción listo!")

def calcular_anuncios(duracion_minutos: float) -> int:
    anuncios_base = math.ceil(duracion_minutos / MINUTOS_POR_ANUNCIO)
    return anuncios_base + ANUNCIOS_RESPALDO

@app.get("/")
def home():
    return {"status": "Servidor KLP activo y listo"}

@app.post("/transcribir")
async def transcribir_audio(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Transcribir con Whisper y obtener la duración de la información del audio
        segmentos, info = modelo.transcribe(temp_filename, language="es")
        
        # Whisper calcula la duración directamente en segundos
        duracion_minutos = info.duration / 60
        num_anuncios = calcular_anuncios(duracion_minutos)

        texto = " ".join([segmento.text.strip() for segmento in segmentos])

        return {
            "exito": True,
            "duracion_minutos": round(duracion_minutos, 2),
            "anuncios_requeridos": num_anuncios,
            "transcripcion": texto
        }

    except Exception as e:
        return {"exito": False, "error": str(e)}

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
