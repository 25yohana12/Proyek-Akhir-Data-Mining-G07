from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model yang telah dilatih
model = joblib.load('model.pkl')

# Kolom fitur yang digunakan untuk prediksi (tanpa surveyId)
feature_columns = [
    'lat', 'lon', 'geoUncertaintyInM',
    'Soilgrid-bdod', 'Soilgrid-cec', 'Soilgrid-cfvo',
    'Soilgrid-clay', 'Soilgrid-nitrogen', 'Soilgrid-phh2o',
    'Soilgrid-sand', 'Soilgrid-silt', 'Soilgrid-soc'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil semua input dari form dan konversi ke float
        values = [float(request.form.get(col)) for col in feature_columns]

        # Konversi ke array 2D agar cocok dengan input model
        features = np.array([values])

        # Lakukan prediksi dengan model
        predicted_species_id = model.predict(features)[0]

        # Ambil nilai surveyId dari form (bukan untuk model, hanya tampilan)
        survey_id = request.form.get('surveyId')

        return render_template(
            'index.html',
            prediction_text=str(predicted_species_id),
            survey_id=survey_id
        )
    
    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Terjadi kesalahan: {str(e)}'
        )

if __name__ == '__main__':
    app.run(debug=True)
