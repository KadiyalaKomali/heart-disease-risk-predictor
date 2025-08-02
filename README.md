# 💓 Heart Disease Risk Predictor

This Streamlit app predicts the risk of **heart disease** based on user input using a **Decision Tree Classifier** trained on real-world medical data.

🔗 Live Demo : https://heartdiseaseriskpredicting.streamlit.app/

## 🧠 Model

- Trained on `heart.csv` dataset
- 11 input features
- Binary classification (`HeartDisease`: 0 or 1)
- Model: DecisionTreeClassifier
- Label encoding for categorical features

## 📋 Features

- User form to enter medical details
- Real-time risk prediction
- Clean, mobile-friendly UI
- Reproducible model with saved encoders + `model.pkl`

## 📁 Dataset Columns

| Column            | Description                       |
|-------------------|-----------------------------------|
| Age               | Age of the patient                |
| Sex               | Gender (M/F)                      |
| ChestPainType     | Type of chest pain                |
| RestingBP         | Resting blood pressure            |
| Cholesterol       | Serum cholesterol                 |
| FastingBS         | Fasting blood sugar (0/1)         |
| RestingECG        | Resting electrocardiographic res. |
| MaxHR             | Maximum heart rate achieved       |
| ExerciseAngina    | Exercise-induced angina (Y/N)     |
| Oldpeak           | ST depression                     |
| ST_Slope          | Slope of ST segment               |
| HeartDisease      | Target variable (0 = No, 1 = Yes) |

## 🚀 Run the App Locally

```bash
git clone https://github.com/yourusername/heart-disease-predictor.git
cd heart-disease-predictor

# (Optional) Create virtual env
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
python train_model.py

# Run Streamlit
streamlit run app.py


