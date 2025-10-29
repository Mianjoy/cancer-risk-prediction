// Convierte un checkbox en 0/1 según su estado
function getBool(id) {
    return document.getElementById(id).checked ? 1 : 0;
}

// Construye el payload desde el formulario y realiza la llamada a la API
function predict() {
    const data = {
        age: parseInt(document.getElementById('age').value),
        gender: document.getElementById('gender').value,
        bmi: parseFloat(document.getElementById('bmi').value),
        alcohol_consumption: document.getElementById('alcohol_consumption').value,
        smoking_status: document.getElementById('smoking_status').value,
        hepatitis_b: getBool('hepatitis_b'),
        hepatitis_c: getBool('hepatitis_c'),
        liver_function_score: parseFloat(document.getElementById('liver_function_score').value),
        alpha_fetoprotein_level: parseFloat(document.getElementById('alpha_fetoprotein_level').value),
        cirrhosis_history: getBool('cirrhosis_history'),
        family_history_cancer: getBool('family_history_cancer'),
        physical_activity_level: document.getElementById('physical_activity_level').value,
        diabetes: getBool('diabetes')
    };

    // Llamada al endpoint FastAPI local
    fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
    .then(res => {
        if (!res.ok) throw new Error('Error en la API');
        return res.json();
    })
    .then(result => {
        // Colorea el porcentaje de riesgo según umbral 50%
        const color = result.risk_percentage > 50 ? 'red' : 'green';
        document.getElementById('result').innerHTML = `
            <h3>📊 Resultado de Predicción</h3>
            <p><strong>Riesgo de cáncer de hígado:</strong> 
            <span style="font-size:1.6em; color:${color}; font-weight:bold;">${result.risk_percentage}%</span></p>
            <p><strong>Acción clínica recomendada:</strong> ${result.clinical_message}</p>
        `;
    })
    .catch(err => {
        // Manejo básico de errores de red/API
        document.getElementById('result').innerHTML = `<p style="color:red;">❌ Error: ${err.message}</p>`;
    });
}