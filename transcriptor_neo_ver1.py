import whisper
from pydub import AudioSegment
from pydub.utils import which
from pyannote.audio import Pipeline
import os
import noisereduce as nr
from scipy.io import wavfile
import numpy as np

# Configurar rutas explícitas para ffmpeg y ffprobe
print("FFmpeg path:", which("ffmpeg"))
print("FFprobe path:", which("ffprobe"))

AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

# Carpetas de trabajo
carpeta_audio = r"DIRECCION DE TU CARPETA PARA AUDIOS"
carpeta_transcripciones = r"DIRECCION DE TU CARPETA PARA TRANSCRIPCIONES"

os.makedirs(carpeta_transcripciones, exist_ok=True)

# Token de Hugging Face
huggingface_token = "TU TOKEN DE HUGGIN FACE EN READ"

# Función para obtener el único archivo de audio en la carpeta
def obtener_archivo_audio(carpeta_audio):
    archivos = [f for f in os.listdir(carpeta_audio) if f.lower().endswith(('.mp3', '.wav', '.mp4', '.m4a', '.aac', '.flac'))]
    if len(archivos) != 1:
        raise Exception("Debe haber un único archivo de audio en la carpeta de trabajo.")
    return os.path.join(carpeta_audio, archivos[0])

# Convertir audio a formato WAV si es necesario
def convertir_a_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)  # Convertir a mono
    audio.export(output_path, format="wav")
    return output_path

# Reducir ruido del archivo WAV por ventanas pequeñas
def reducir_ruido_por_partes(input_path, output_path, ventana=50000):  # Ventana de 5 segundos
    print("Reduciendo ruido del audio por partes...")
    rate, data = wavfile.read(input_path)
    data_denoised = np.zeros_like(data)

    for start in range(0, len(data), ventana):
        end = start + ventana
        parte = data[start:end]
        data_denoised[start:end] = nr.reduce_noise(y=parte, sr=rate)

    wavfile.write(output_path, rate, data_denoised)

# Obtener el único archivo de audio en la carpeta
print("Obteniendo archivo de audio...")
ruta_audio = obtener_archivo_audio(carpeta_audio)
print(f"Archivo de audio encontrado: {ruta_audio}")

# Convertir el archivo seleccionado a WAV
ruta_wav = "audio_convertido.wav"
ruta_wav_limpio = "audio_limpio.wav"
audio = AudioSegment.from_file(ruta_audio)
duracion_audio = len(audio) / 1000  # Duración en segundos
convertir_a_wav(ruta_audio, ruta_wav)

# Reducir ruido del archivo WAV convertido
reducir_ruido_por_partes(ruta_wav, ruta_wav_limpio)

# Cargar el modelo de diarización
print("Cargando modelo de diarización...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=huggingface_token
)

# Realizar diarización en todo el archivo
print("Realizando diarización del archivo...")
diarization = pipeline({"audio": ruta_wav_limpio})

# Transcripción con Whisper
print("Cargando el modelo de transcripción...")
modelo = whisper.load_model("small")  # Usar base, small, medium, large para mayor precisión

print("Transcribiendo el archivo de audio...")
resultado_transcripcion = modelo.transcribe(ruta_wav_limpio, language="es", fp16=False)

# Asignar frases transcritas a cada locutor
print("Asignando transcripciones a locutores...")
locutores_map = {}
locutor_id = 0
transcripcion_final = []

for segmento in resultado_transcripcion["segments"]:
    inicio = segmento["start"]
    fin = segmento["end"]
    texto = segmento["text"]

    # Determinar locutor correspondiente al segmento
    locutor = "Desconocido"
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.start <= inicio <= turn.end:
            if speaker not in locutores_map:
                locutores_map[speaker] = f"Locutor_{locutor_id}"
                locutor_id += 1
            locutor = locutores_map[speaker]
            break

    transcripcion_final.append(f"{locutor}: {texto}")

# Guardar la transcripción final
nombre_archivo = os.path.splitext(os.path.basename(ruta_audio))[0]
archivo_salida = os.path.join(carpeta_transcripciones, f"{nombre_archivo}_transcripcion_diarizada.txt")

with open(archivo_salida, "w", encoding="utf-8") as f:
    for linea in transcripcion_final:
        f.write(linea + "\n")

# Eliminar archivos temporales
try:
    os.remove(ruta_wav)
    os.remove(ruta_wav_limpio)
    print("Archivos temporales eliminados.")
except FileNotFoundError:
    print("Algunos archivos temporales no se encontraron y no pudieron ser eliminados.")

print(f"Transcripción completa con diarización. Guardada en: {archivo_salida}")