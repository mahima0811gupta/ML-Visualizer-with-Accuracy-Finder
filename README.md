# 📊 ML Visualizer with Accuracy Finder

An interactive **Streamlit-based Machine Learning Visualizer** that allows users to upload datasets, train multiple ML models, visualize performance metrics, and find the most accurate algorithm for their data.  
This tool automates data analysis, preprocessing, model comparison, and detailed result visualization — all in one place.

---
## 🚀 Features

- **Automatic Task Detection:** Determines whether the problem is Classification or Regression.  
- **Interactive Visualizations:** Explore data distributions, correlations, and model performance using Plotly and Seaborn.  
- **Model Comparison:** Quickly compare algorithms like Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, and MLP.  
- **Detailed Evaluation Metrics:** Displays accuracy, precision, recall, F1-score, confusion matrix, and R² score.  
- **Cross-Validation Support:** Evaluate model stability across multiple folds.  
- **Feature Importance Analysis:** Understand which features most influence the model's predictions.  
- **User-Friendly Interface:** Upload your CSV, select target and features, and visualize results effortlessly.

---

## 🧠 Supported Models

### Classification
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- MLP (Neural Network)

### Regression
- Linear Regression  
- Polynomial Regression  
- Ridge & Lasso Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Support Vector Regressor (SVR)  
- KNN Regressor  
- MLP Regressor  

---

## ⚙️ Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Scikit-learn, NumPy, Pandas  
- **Visualization:** Plotly, Seaborn, Matplotlib  
- **Other:** Python 3.8+, Machine Learning Libraries  

---

## 📂 Project Structure

```

ML-Visualizer-with-Accuracy-Finder/
│
├── app.py                   # Main Streamlit app
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation



## 🧩 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ML-Visualizer-with-Accuracy-Finder.git
   cd ML-Visualizer-with-Accuracy-Finder
````

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

4. **Open in Browser**

   ```
   http://localhost:8501
   ```
## 📊 How It Works

1. Upload your dataset (`.csv` file).
2. The app automatically detects the problem type (classification/regression).
3. Select target and feature columns.
4. Compare multiple algorithms and visualize the best-performing model.
5. View detailed metrics, confusion matrix, radar charts, and feature importance.

---

## 📈 Example Outputs

* Model Accuracy Comparison Chart
* Confusion Matrix Heatmap
* Precision-Recall-F1 Radar Chart
* Feature Importance Visualization
* Residual Analysis for Regression

---

## 🏆 Highlights

* End-to-end ML visualization workflow.
* No-code interface for ML model evaluation.
* Dynamic insights into model strengths and weaknesses.
* Perfect for students, researchers, and data scientists.



