#!/bin/bash

MODEL_PATH="/app/modelo_entrenado/digit_mlp.pth"
MAX_ATTEMPTS=30
ATTEMPT=0

echo "Esperando a que el modelo esté disponible en $MODEL_PATH"

while [ ! -f "$MODEL_PATH" ] && [ $ATTEMPT -lt $MAX_ATTEMPTS ]
do
  ATTEMPT=$((ATTEMPT+1))
  echo "Esperando el modelo... intento $ATTEMPT de $MAX_ATTEMPTS"
  sleep 3
done

if [ -f "$MODEL_PATH" ]
then
  echo "Modelo encontrado. Iniciando la API..."
  exec uvicorn main:app --host 0.0.0.0 --port 8000
else
  echo "Error: Modelo no encontrado después de esperar. Asegúrese de que el entrenamiento se completó correctamente."
  exit 1
fi 