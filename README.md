# 🔎 Cyber Web Threats – Anomaly Detection Dashboard

A **Streamlit-based dashboard** for detecting suspicious web traffic using **Isolation Forest** (unsupervised anomaly detection) and **Random Forest** (supervised classification).  
The project integrates **data preprocessing, ML explainability (SHAP), visualization, and interactive analysis** to help cybersecurity professionals analyze abnormal traffic patterns.

---

## 📌 Features

- ✅ **Data Preprocessing** – automatic feature engineering (`session_duration`, `bytes_total`, `avg_packet_size`)  
- ✅ **Isolation Forest** – unsupervised anomaly detection for suspicious web traffic  
- ✅ **Random Forest** – supervised classification with confusion matrix & classification report  
- ✅ **Explainability with SHAP** – feature importance analysis for RandomForest  
- ✅ **Interactive Visualizations**:
  - Scatter plots (Normal vs Suspicious traffic)  
  - Suspicious traffic **IP network graph**  
  - **GeoIP map** for suspicious countries  
- ✅ **Download suspicious records** (CSV export)  
- ✅ **Combined anomaly + classification results**  

---

## 📂 Project Structure

```
cyber-web-threats/
│
├── app/                        # Streamlit app
│   ├── app.py                  # Main Streamlit dashboard
│   ├── network.html            # Saved network visualization
│
├── data/
│   └── CloudWatch_Traffic_Web_Attack.csv   # Example dataset
│
├── src/                        # Source modules
│   ├── load_data.py
│   ├── preprocess.py
│   ├── isolation_model.py
│   ├── random_forest_model.py
│   ├── explain_model.py
│   ├── eda_plot.py
│   ├── evaluate.py
│   ├── logger.py
│   └── network_graph.py
│
├── lib/                        # JS/CSS libs for network visualizations
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md                   # Project documentation
```

---

## ⚡ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Abinash-Kalita/cyber-web-threats.git
cd cyber-web-threats
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Linux/Mac
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App
```bash
streamlit run app/app.py
```

The app will be available at:
- Local: [http://localhost:8501](http://localhost:8501)  
- Network: `http://<your-ip>:8501`

---

## 📊 Example Outputs

- **Summary Dashboard** – rows, suspicious/normal counts  
- **Scatter Plot** – Bytes In vs Bytes Out with anomalies highlighted  
- **Network Graph** – Suspicious IP connections  
- **GeoIP Map** – Suspicious traffic by country  
- **SHAP Feature Importances** – Explainability for RandomForest  
- **Classification Report** – Precision, Recall, F1-score  
- **Downloadable CSV** – Suspicious records and combined results  

*(📸 Screenshots can be added here once you run the app and capture images.)*

---

## 🧪 Tech Stack

- **Python 3.9+**
- **Streamlit** – interactive UI
- **scikit-learn** – ML models (Isolation Forest, RandomForest)
- **pandas / numpy** – data processing
- **matplotlib / seaborn / plotly** – visualizations
- **networkx + pyvis** – suspicious traffic network graphs
- **SHAP** – explainability

---

## 🚀 Future Improvements

- Deploy live app using **Streamlit Cloud / Render**  
- Add **deep learning anomaly detection** (autoencoders, LSTMs)  
- Expand dataset integration with **real-time network traffic**  
- Add **alerting system** for suspicious activity  

---

## 📜 License

This project is licensed under the **MIT License** – feel free to use and modify it.

---

## 👨‍💻 Author

**Abinash Kalita**  
🔗 [GitHub Profile](https://github.com/Abinash-Kalita)  
