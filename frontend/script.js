document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('drawing-canvas');
    const ctx = canvas.getContext('2d');
    const clearButton = document.getElementById('clear-button');
    const predictButton = document.getElementById('predict-button');
    const loadingElement = document.getElementById('loading');
    const resultElement = document.getElementById('result');
    const errorElement = document.getElementById('error');
    const predictionResult = document.getElementById('prediction-result');
    const confidenceValue = document.getElementById('confidence-value');
    const probabilitiesContainer = document.getElementById('probabilities-container');
    
    // URL del backend
    const API_URL = 'http://localhost:8000';
    
    // Variables para el dibujo
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    
    // Configurar el lienzo
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.lineWidth = 15;
    ctx.strokeStyle = 'black';
    
    // Fondo blanco
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Funciones de dibujo
    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }
    
    function draw(e) {
        if (!isDrawing) return;
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }
    
    function stopDrawing() {
        isDrawing = false;
    }
    
    // Función para limpiar el lienzo
    function clearCanvas() {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Ocultar resultados anteriores
        resultElement.classList.add('hidden');
        errorElement.classList.add('hidden');
    }
    
    // Función para obtener la imagen del canvas como base64
    function getImageData() {
        return canvas.toDataURL('image/png');
    }
    
    // Función para enviar la imagen al backend
    async function predictDigit() {
        const imageData = getImageData();
        
        // Mostrar cargando y ocultar otros elementos
        loadingElement.classList.remove('hidden');
        resultElement.classList.add('hidden');
        errorElement.classList.add('hidden');
        
        try {
            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData }),
            });
            
            if (!response.ok) {
                throw new Error('Error en la respuesta del servidor');
            }
            
            const data = await response.json();
            
            // Ocultar cargando y mostrar resultados
            loadingElement.classList.add('hidden');
            resultElement.classList.remove('hidden');
            
            // Mostrar la predicción
            predictionResult.textContent = data.prediction;
            confidenceValue.textContent = (data.confidence * 100).toFixed(2);
            
            // Generar barras de probabilidad
            displayProbabilities(data.probabilities);
            
        } catch (error) {
            console.error('Error:', error);
            loadingElement.classList.add('hidden');
            errorElement.classList.remove('hidden');
        }
    }
    
    // Función para mostrar las probabilidades como barras
    function displayProbabilities(probabilities) {
        probabilitiesContainer.innerHTML = '';
        
        probabilities.forEach((prob, index) => {
            const probPercent = (prob * 100).toFixed(2);
            
            const barDiv = document.createElement('div');
            barDiv.className = 'prob-bar';
            
            const digitSpan = document.createElement('span');
            digitSpan.textContent = index;
            
            const barContainer = document.createElement('div');
            barContainer.className = 'bar-container';
            
            const bar = document.createElement('div');
            bar.className = 'bar';
            bar.style.width = `${probPercent}%`;
            
            const valueSpan = document.createElement('span');
            valueSpan.className = 'prob-value';
            valueSpan.textContent = `${probPercent}%`;
            
            barContainer.appendChild(bar);
            barDiv.appendChild(digitSpan);
            barDiv.appendChild(barContainer);
            barDiv.appendChild(valueSpan);
            
            probabilitiesContainer.appendChild(barDiv);
        });
    }
    
    // Event listeners
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    // Soporte para dispositivos táctiles
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault(); // Prevenir el scroll
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousedown', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    });
    
    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousemove', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    });
    
    canvas.addEventListener('touchend', () => {
        const mouseEvent = new MouseEvent('mouseup', {});
        canvas.dispatchEvent(mouseEvent);
    });
    
    // Botones
    clearButton.addEventListener('click', clearCanvas);
    predictButton.addEventListener('click', predictDigit);
    
    // Inicialización
    clearCanvas();
}); 