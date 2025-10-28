

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error, r2_score, classification_report, mean_absolute_error
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
import warnings
import numpy as np
import traceback

st.set_page_config(page_title="ML Visualizer", page_icon="📊", layout="wide")
warnings.filterwarnings("ignore")
pio.templates.default = "plotly_white"

def get_model(model_name, is_classification=True, fast=False):
    """Returns model instance with optional fast mode for comparison"""
    if is_classification:
        if model_name == "Logistic Regression":
            return LogisticRegression(max_iter=500 if fast else 1000, random_state=42)
        elif model_name == "Decision Tree":
            return DecisionTreeClassifier(random_state=42, max_depth=10 if fast else None)
        elif model_name == "Random Forest":
            return RandomForestClassifier(random_state=42, n_estimators=50 if fast else 100)
        elif model_name == "SVM":
            return SVC(probability=True, random_state=42, kernel='linear' if fast else 'rbf')
        elif model_name == "KNN":
            return KNeighborsClassifier()
        elif model_name == "Naive Bayes":
            return GaussianNB()
        elif model_name == "MLP (Neural Network)":
            return MLPClassifier(max_iter=300 if fast else 1000, 
                               hidden_layer_sizes=(32,) if fast else (64, 32),
                               random_state=42, early_stopping=True, n_iter_no_change=10)
    else:
        if model_name == "Linear Regression":
            return LinearRegression()
        elif model_name == "Polynomial Regression":
            return Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())])
        elif model_name == "Ridge Regression":
            return Ridge(random_state=42)
        elif model_name == "Lasso Regression":
            return Lasso(random_state=42)
        elif model_name == "Decision Tree":
            return DecisionTreeRegressor(random_state=42, max_depth=10 if fast else None)
        elif model_name == "Random Forest":
            return RandomForestRegressor(random_state=42, n_estimators=50 if fast else 100)
        elif model_name == "SVM":
            return SVR(kernel='linear' if fast else 'rbf')
        elif model_name == "KNN":
            return KNeighborsRegressor()
        elif model_name == "MLP (Neural Network)":
            return MLPRegressor(max_iter=300 if fast else 1000,
                              hidden_layer_sizes=(32,) if fast else (64, 32),
                              random_state=42, early_stopping=True, n_iter_no_change=10)
    return None

def is_classification_task(y):
    if pd.api.types.is_numeric_dtype(y):
        return not (y.nunique() > 20 or pd.api.types.is_float_dtype(y))
    return True

st.sidebar.title("ML Visualizer 📊")
st.sidebar.markdown("Advanced Machine Learning Analysis Tool")
st.sidebar.header("⚙️ Controls & Configuration")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

st.title("🎯 ML Visualizer with Accuracy Finder")
st.write("Upload your data, train models, and **find accuracy** with detailed performance metrics!")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success(f"✅ Data loaded successfully! Shape: {data.shape}")
    except Exception as e:
        st.error(f"❌ Error loading CSV file: {e}")
        st.stop()

    st.header("📊 1. Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{data.shape[0]:,}")
    with col2:
        st.metric("Total Columns", f"{data.shape[1]:,}")
    with col3:
        st.metric("Missing Values", f"{data.isnull().sum().sum():,}")
    with col4:
        st.metric("Duplicate Rows", f"{data.duplicated().sum():,}")

    with st.expander("📋 View Data Preview & Statistics"):
        st.dataframe(data.head(10))
        st.subheader("Descriptive Statistics")
        st.dataframe(data.describe())

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    st.info(f"📊 Found {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")

    st.subheader("📊 Data Distribution Overview")
    non_null_counts = data.count()
    fig_dist = px.bar(non_null_counts, x=non_null_counts.index, y=non_null_counts.values,
                      labels={'x': 'Column', 'y': 'Non-Null Count'},
                      title="Non-Null Value Counts per Column",
                      color=non_null_counts.values, color_continuous_scale='Blues')
    st.plotly_chart(fig_dist, use_container_width=True)
    
    st.header("🎨 2. Automated Visualizations")
    with st.expander("Click to view automated plots for all columns"):
        st.subheader("Numeric Column Distributions")
        if numeric_cols:
            for col in numeric_cols:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                sns.histplot(data[col], kde=True, ax=ax1, color='blue', fill=True)
                ax1.set_title(f"Histogram of {col}")
                sns.boxplot(y=data[col], ax=ax2, color='lightblue')
                ax2.set_title(f"Boxplot of {col}")
                st.pyplot(fig)
                st.markdown("---")
        else:
            st.write("No numeric columns found.")

        st.subheader("Categorical Column Distributions")
        if categorical_cols:
            for col in categorical_cols:
                if data[col].nunique() > 50:
                    st.write(f"Skipping '{col}': Too many unique values ({data[col].nunique()}).")
                else:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.countplot(y=data[col], ax=ax, order=data[col].value_counts().index, palette='Blues_r')
                    ax.set_title(f"Count Plot of {col}")
                    st.pyplot(fig)
                    st.markdown("---")
        else:
            st.write("No categorical columns found.")

        st.subheader("Correlation Heatmap (Numeric Columns)")
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(data[numeric_cols].corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax)
            st.pyplot(fig)
        else:
            st.write("Not enough numeric columns for correlation heatmap.")

    st.sidebar.markdown("---")
    st.sidebar.header("🤖 Model Configuration")

    if len(data.columns) == 0:
        st.sidebar.warning("No columns in data.")
        st.stop()

    target_col = st.sidebar.selectbox("🎯 Select Target Variable (y)", data.columns, index=len(data.columns)-1)
    
    is_class_task = is_classification_task(data[target_col])
    task_type = "Classification" if is_class_task else "Regression"
    st.sidebar.success(f"**Detected Task Type:** {task_type}")
    st.sidebar.write(f"**Target Variable Info:**")
    st.sidebar.write(f"- Unique values: {data[target_col].nunique()}")
    st.sidebar.write(f"- Missing values: {data[target_col].isnull().sum()}")

    default_features = [col for col in numeric_cols if col != target_col]
    feature_cols = st.sidebar.multiselect("📝 Select Feature Variables (X)",
                                          [col for col in data.columns if col != target_col],
                                          default=default_features)
    
    if not feature_cols:
        st.sidebar.warning("⚠️ Please select at least one feature!")

    model_list = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN", "Naive Bayes", "MLP (Neural Network)"] if is_class_task else ["Linear Regression", "Polynomial Regression", "Ridge Regression", "Lasso Regression", "Decision Tree", "Random Forest", "SVM", "KNN", "MLP (Neural Network)"]
    
    model_name = st.sidebar.selectbox("🔧 Select Model for Detailed Analysis", model_list)

    with st.sidebar.expander("⚙️ Advanced Options"):
        test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
        use_cv = st.checkbox("Use Cross-Validation", value=True)
        if use_cv:
            cv_folds = st.slider("CV Folds", 3, 10, 5)
        run_comparison = st.checkbox("Run Algorithm Comparison", value=True,
                                    help="Train all models to find the best one.")

    train_button = st.sidebar.button("🚀 Train Model & Find Accuracy", type="primary")

    if train_button:
        st.header("🎯 MODEL TRAINING STARTED")

        if not feature_cols:
            st.error("❌ Please select at least one feature (X).")
            st.stop()
        if target_col in feature_cols:
            st.error("❌ Target variable cannot be in features.")
            st.stop()

        try:
            st.write("### 📦 Data Preparation")
            model_data = data[feature_cols + [target_col]].copy()
            initial_rows = len(model_data)
            model_data.dropna(inplace=True)
            final_rows = len(model_data)

            if final_rows < initial_rows:
                st.warning(f"⚠️ Dropped {initial_rows - final_rows} rows due to missing values.")
            if final_rows == 0:
                st.error("❌ No data left after removing missing values.")
                st.stop()

            st.success(f"✅ Using {final_rows} rows for training")

            X = model_data[feature_cols]
            y = model_data[target_col]

            X_encoded = X.copy()
            encoded_cols = []
            for col in X_encoded.columns:
                if X_encoded[col].dtype == 'object':
                    X_encoded[col] = pd.Categorical(X_encoded[col]).codes
                    encoded_cols.append(col)
            if encoded_cols:
                st.info(f"🔤 Encoded categorical features: {', '.join(encoded_cols)}")

            if is_class_task and y.dtype == 'object':
                y = pd.Categorical(y).codes
                st.info(f"🔤 Encoded target variable: {target_col}")

            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, random_state=42)
            st.write(f"**Train:** {len(X_train)} | **Test:** {len(X_test)}")

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            st.success("✅ Data preparation complete!")

            if run_comparison:
                st.header("🏁 Algorithm Comparison")
                st.info("ℹ️ Running fast versions of all models.")
                comparison_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, m_name in enumerate(model_list):
                    status_text.info(f"Training {m_name}...")
                    try:
                        m = get_model(m_name, is_classification=is_class_task, fast=True)
                        if m:
                            if m_name == "Polynomial Regression":
                                m.fit(X_train, y_train)
                                y_p = m.predict(X_test)
                            else:
                                m.fit(X_train_scaled, y_train)
                                y_p = m.predict(X_test_scaled)
                            score = accuracy_score(y_test, y_p) if is_class_task else r2_score(y_test, y_p)
                            comparison_results.append({"Algorithm": m_name, "Score": score})
                    except Exception as e:
                        st.warning(f"⚠️ {m_name} failed: {str(e)}")
                    progress_bar.progress((idx + 1) / len(model_list))

                status_text.success("✅ Comparison complete!")

                if comparison_results:
                    metric_name = "Accuracy" if is_class_task else "R-Squared"
                    comp_df = pd.DataFrame(comparison_results).rename(columns={'Score': metric_name})
                    comp_df = comp_df.sort_values(by=metric_name, ascending=False)
                    st.dataframe(comp_df, use_container_width=True)

                    fig_comp = px.bar(comp_df, x='Algorithm', y=metric_name, color=metric_name,
                                     color_continuous_scale='Blues', title='Algorithm Performance',
                                     text=metric_name)
                    fig_comp.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
                    best = comp_df.iloc[0]
                    st.success(f"🏆 Best: **{best['Algorithm']}** - {metric_name}: **{best[metric_name]:.4f}**")

            st.header(f"🎯 Detailed Performance: {model_name}")
            model = get_model(model_name, is_classification=is_class_task, fast=False)
            
            with st.spinner(f'🔄 Training {model_name}...'):
                if model_name == "Polynomial Regression":
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_train_pred = model.predict(X_train)
                else:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_train_pred = model.predict(X_train_scaled)
            
            st.success("✅ Model training complete!")

            if is_class_task:
                train_acc = accuracy_score(y_train, y_train_pred)
                test_acc = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                st.markdown("### 🎯 ACCURACY RESULTS")
                col1, col2, col3 = st.columns(3)
                col1.metric("Training Accuracy", f"{train_acc*100:.2f}%")
                col2.metric("Test Accuracy", f"{test_acc*100:.2f}%", delta=f"{(test_acc-train_acc)*100:.2f}%")
                col3.metric("Status", "⚠️ Overfit" if abs(train_acc-test_acc)>0.05 else "✓ Good")

                st.markdown("### 📈 Additional Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Precision", f"{precision:.4f}")
                col2.metric("Recall", f"{recall:.4f}")
                col3.metric("F1-Score", f"{f1:.4f}")

                plot_col1, plot_col2 = st.columns(2)
                with plot_col1:
                    st.markdown("### 📡 Performance Radar")
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[test_acc, precision, recall, f1],
                        theta=['Accuracy', 'Precision', 'Recall', 'F1'],
                        fill='toself', line=dict(color='blue')))
                    fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0, 1])))
                    st.plotly_chart(fig_radar, use_container_width=True)

                if model_name == "MLP (Neural Network)" and hasattr(model, 'loss_curve_'):
                    with plot_col2:
                        st.markdown("### 📉 Training Loss")
                        epoch_df = pd.DataFrame({'Epoch': range(1, len(model.loss_curve_)+1),
                                                'Loss': model.loss_curve_})
                        fig_epoch = px.line(epoch_df, x='Epoch', y='Loss', markers=True)
                        st.plotly_chart(fig_epoch, use_container_width=True)

                st.markdown("### 🔲 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig_cm)

                with st.expander("📄 Classification Report"):
                    try:
                        rep = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
                        st.dataframe(rep.style.background_gradient(cmap='Blues'))
                    except:
                        st.warning("Could not generate report")

            else:
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)

                st.markdown("### 🎯 R² SCORE RESULTS")
                col1, col2, col3 = st.columns(3)
                col1.metric("Training R²", f"{train_r2:.4f}")
                col2.metric("Test R²", f"{test_r2:.4f}", delta=f"{test_r2-train_r2:.4f}")
                col3.metric("Quality", "✓ Good" if test_r2>=0.7 else "⚠️ Improve")

                st.markdown("### 📉 Error Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("RMSE", f"{rmse:.4f}")
                col2.metric("MSE", f"{mse:.4f}")
                col3.metric("MAE", f"{mae:.4f}")

                if model_name == "MLP (Neural Network)" and hasattr(model, 'loss_curve_'):
                    st.markdown("### 📉 Training Loss")
                    epoch_df = pd.DataFrame({'Epoch': range(1, len(model.loss_curve_)+1),
                                            'Loss': model.loss_curve_})
                    fig_epoch = px.line(epoch_df, x='Epoch', y='Loss', markers=True)
                    st.plotly_chart(fig_epoch, use_container_width=True)

                st.markdown("### 📈 Predicted vs Actual")
                fig_pred = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'},
                                     title=f'R² = {test_r2:.3f}', opacity=0.6)
                fig_pred.add_shape(type='line', x0=y_test.min(), y0=y_test.min(),
                                  x1=y_test.max(), y1=y_test.max(),
                                  line=dict(color='red', dash='dash'))
                st.plotly_chart(fig_pred, use_container_width=True)

                st.markdown("### 📉 Residual Analysis")
                residuals = y_test - y_pred
                rc1, rc2 = st.columns(2)
                with rc1:
                    fig_res = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted', 'y': 'Residuals'},
                                        title='Residual Plot')
                    fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_res, use_container_width=True)
                with rc2:
                    fig_dist = px.histogram(residuals, nbins=50, title='Error Distribution')
                    st.plotly_chart(fig_dist, use_container_width=True)

            if use_cv:
                st.markdown("### 🔄 Cross-Validation")
                try:
                    metric = 'accuracy' if is_class_task else 'r2'
                    cv_X = X_train if model_name == "Polynomial Regression" else X_train_scaled
                    cv_scores = cross_val_score(model, cv_X, y_train, cv=cv_folds, scoring=metric)
                    col1, col2 = st.columns(2)
                    col1.metric("CV Mean", f"{cv_scores.mean():.4f}")
                    col2.metric("CV Std", f"±{cv_scores.std():.4f}")
                    with st.expander("📊 All CV Scores"):
                        st.dataframe(pd.DataFrame({'Fold': [f'Fold {i+1}' for i in range(len(cv_scores))],
                                                   'Score': cv_scores}))
                except Exception as e:
                    st.warning(f"⚠️ CV failed: {e}")

            st.markdown("### 🔍 Feature Importance")
            if model_name == "Polynomial Regression":
                st.info("Not available for Polynomial Regression")
            elif hasattr(model, 'feature_importances_'):
                imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_})
                imp_df = imp_df.sort_values('Importance', ascending=False)
                fig_imp = px.bar(imp_df.head(15), x='Importance', y='Feature', orientation='h')
                fig_imp.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_imp, use_container_width=True)
            elif hasattr(model, 'coef_'):
                coeffs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': coeffs})
                coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)
                fig_coef = px.bar(coef_df.head(15), x='Coefficient', y='Feature', orientation='h')
                fig_coef.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_coef, use_container_width=True)
            else:
                st.write(f"Not available for {model_name}")

            st.markdown("---")
            st.markdown("## 📋 Summary")
            summary = {"Metric": ["Model", "Features", "Train Samples", "Test Samples"],
                      "Value": [model_name, len(feature_cols), len(X_train), len(X_test)]}
            if is_class_task:
                summary["Metric"].extend(["Test Accuracy", "Precision", "Recall", "F1"])
                summary["Value"].extend([f"{test_acc:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])
            else:
                summary["Metric"].extend(["Test R²", "RMSE", "MAE"])
                summary["Value"].extend([f"{test_r2:.4f}", f"{rmse:.4f}", f"{mae:.4f}"])
            st.table(pd.DataFrame(summary).set_index("Metric"))

            st.success("🎉 TRAINING AND ANALYSIS COMPLETE!")

        except Exception as e:
            st.error(f"❌ ERROR: {str(e)}")
            st.code(traceback.format_exc())

else:
    st.info("👆 Upload CSV file to begin")
    
