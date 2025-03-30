from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import io
from PIL import Image
import numpy as np
import base64
from pydantic import BaseModel
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definir la arquitectura del modelo MLP (igual que en el script de entrenamiento)
class DigitMLP(nn.Module):
    def __init__(self):
        super(DigitMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Crear la app de FastAPI
app = FastAPI(title="API para Reconocimiento de Dígitos")

# Configurar CORS de manera más permisiva
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo global
model = None

class ImageData(BaseModel):
    image: str  # Base64 encoded image data

# Cargar el modelo al iniciar
@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = DigitMLP()
        model_path = "/app/modelo_entrenado/digit_mlp.pth"
        model.load_state_dict(torch.load(model_path))
        model.eval()
        logger.info(f"Modelo cargado correctamente desde {model_path}")
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        model = None

def preprocess_image(image_data):
    # Convertir de base64 a imagen
    try:
        image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Redimensionar a 28x28 (tamaño esperado por MNIST)
        img = img.resize((28, 28))
        
        # Convertir a tensor y normalizar
        img_array = np.array(img, dtype=np.float32) / 255.0
        # Invertir colores si es necesario (en MNIST, 0 es negro y 1 es blanco)
        img_array = 1.0 - img_array  # invertir si el fondo es blanco
        
        # Normalizar usando la media y desviación estándar de MNIST
        img_array = (img_array - 0.1307) / 0.3081
        
        # Convertir a tensor
        tensor = torch.tensor(img_array).unsqueeze(0)  # Agregar dimensión de batch
        
        return tensor
    except Exception as e:
        logger.error(f"Error en el preprocesamiento de la imagen: {e}")
        raise

@app.post("/predict")
async def predict(data: ImageData):
    global model
    
    # Verificar si el modelo está cargado
    if model is None:
        logger.error("Modelo no disponible")
        raise HTTPException(status_code=500, detail="Modelo no disponible")
    
    try:
        logger.info("Recibida solicitud de predicción")
        
        # Preprocesar la imagen
        image_tensor = preprocess_image(data.image)
        
        # Realizar la predicción
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        result = {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities[0].tolist()
        }
        
        logger.info(f"Predicción exitosa: {prediction} con confianza {confidence:.4f}")
        return result
    except Exception as e:
        logger.error(f"Error al procesar la imagen: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None} 