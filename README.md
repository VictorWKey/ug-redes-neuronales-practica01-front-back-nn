# Proyecto de Reconocimiento de Dígitos con MLP

Este proyecto implementa un sistema de reconocimiento de dígitos utilizando una Red Neuronal Artificial tipo Perceptrón Multicapa (MLP). Permite al usuario dibujar un dígito en un canvas HTML y obtener la predicción del modelo entrenado.

## Estructura del Proyecto

El proyecto está dividido en tres componentes principales, cada uno en su propio contenedor Docker:

1. **Modelo**: Entrena un MLP utilizando PyTorch sobre el dataset MNIST
2. **Backend**: API REST con FastAPI que utiliza el modelo entrenado para hacer predicciones
3. **Frontend**: Interfaz web simple con HTML, CSS y JavaScript puro

## Requisitos Previos

- Docker
- Docker Compose

## Cómo Ejecutar

1. Clone este repositorio:
   ```bash
   git clone <URL-del-repositorio>
   cd <nombre-del-directorio>
   ```

2. Inicie los contenedores con Docker Compose:
   ```bash
   docker-compose up
   ```

   Este comando:
   - Construye las imágenes Docker para los tres componentes
   - Entrena el modelo en el primer contenedor
   - Inicia el backend que usa el modelo entrenado
   - Inicia el servidor web para el frontend

3. Abra su navegador y vaya a:
   ```
   http://localhost
   ```

## Cómo Usar

1. Dibuje un dígito (0-9) en el área de dibujo utilizando el ratón o un dispositivo táctil
2. Haga clic en el botón "Predecir" para enviar la imagen al modelo
3. Vea los resultados:
   - El dígito predicho
   - La confianza de la predicción
   - Las probabilidades de cada clase

Para limpiar el área de dibujo, use el botón "Limpiar".

## Arquitectura

### Modelo (PyTorch)
- Implementa un MLP con capas lineales (nn.Linear)
- Se entrena sobre el dataset MNIST
- Guarda el modelo entrenado para ser utilizado por el backend

### Backend (FastAPI)
- Expone una API REST
- Endpoint principal `/predict` para recibir imágenes y devolver predicciones
- Carga el modelo previamente entrenado
- Preprocesa las imágenes para adaptarlas al formato esperado por el modelo

### Frontend (HTML/JS/CSS)
- Interfaz de usuario simple e intuitiva
- Canvas para dibujar dígitos
- Comunicación con el backend mediante fetch API
- Visualización de resultados con barras de probabilidad

## Desarrollo

Para modificar el proyecto:

1. Modelo: Edite `modelo/train_model.py` para cambiar la arquitectura o parámetros
2. Backend: Modifique `backend/main.py` para ajustar la API
3. Frontend: Edite los archivos en `frontend/` para mejorar la interfaz

Después de los cambios, reconstruya e inicie los contenedores:
```bash
docker-compose up --build
``` 