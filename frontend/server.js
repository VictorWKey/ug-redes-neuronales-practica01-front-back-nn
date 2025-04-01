const express = require('express');
const path = require('path');
const cors = require('cors');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 80;
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

app.use(cors());

app.use(express.static(path.join(__dirname, '.'), {
  setHeaders: (res, path) => {
    if (path.endsWith('.js')) {
      res.set('Content-Type', 'application/javascript');
    }
  },
  index: false
}));

app.get('/script.js', (req, res) => {
  try {
    let jsContent = fs.readFileSync(path.join(__dirname, 'script.js'), 'utf8');
    
    // Reemplazar la URL del backend con la variable de entorno
    jsContent = jsContent.replace(
      /const API_URL = ['"].*?['"]/,
      `const API_URL = '${BACKEND_URL}'`
    );
    
    res.set('Content-Type', 'application/javascript');
    res.send(jsContent);
    console.log(`Servido script.js con API_URL = ${BACKEND_URL}`);
  } catch (error) {
    console.error('Error al servir script.js:', error);
    res.status(500).send('Error al cargar el script');
  }
});

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Servidor frontend corriendo en http://localhost:${PORT}`);
  console.log(`Conectado al backend en: ${BACKEND_URL}`);
}); 