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
    
    const API_URL = 'http://localhost:8000';
    
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.lineWidth = 15;
    ctx.strokeStyle = 'black';
    
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
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
    
    function clearCanvas() {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        resultElement.classList.add('hidden');
        errorElement.classList.add('hidden');
    }
    
    function getImageData() {
        return canvas.toDataURL('image/png');
    }

    async function predictDigit() {
        const imageData = getImageData();
        
        loadingElement.classList.remove('hidden');
        resultElement.classList.add('hidden');
        errorElement.classList.add('hidden');
        
        try {
            console.log(`Enviando petición a: ${API_URL}/predict`);
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
            console.log('Respuesta recibida:', data);
            
            loadingElement.classList.add('hidden');
            resultElement.classList.remove('hidden');
            
            predictionResult.textContent = data.prediction;
            confidenceValue.textContent = (data.confidence * 100).toFixed(2);
            
            displayProbabilities(data.probabilities);       
        } catch (error) {
            console.error('Error:', error);
            loadingElement.classList.add('hidden');
            errorElement.classList.remove('hidden');
        }
    }
    
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
    
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    clearButton.addEventListener('click', clearCanvas);
    predictButton.addEventListener('click', predictDigit);
    
    clearCanvas();
}); 