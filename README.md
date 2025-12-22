# LoanGuard: Banking Intelligence Dashboard

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.24.0-FF4B4B)
![Plotly](https://img.shields.io/badge/plotly-5.0-3F4F75)
![SHAP](https://img.shields.io/badge/XAI-SHAP-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

**LoanGuard** is an enterprise-grade banking analytics platform designed to optimize loan recovery strategies. It combines machine learning (Random Forest) with Explainable AI (SHAP) to categorize borrowers, predict default risks, and generate procedural interaction histories for a comprehensive "Customer 360" view.

## 🚀 Key Features

*   **🏦 Banking-Grade UI/UX**: A responsive, theme-aware dashboard (Dark/Light mode) designed for financial professionals.
*   **🧠 Explainable AI (XAI)**: Integrated **SHAP (SHapley Additive exPlanations)** analysis to explain *why* a specific risk score was assigned (Regulatory Compliance).
*   **📊 Advanced Visualization**: 
    *   **Sunburst Charts** for hierarchical portfolio exposure.
    *   **Parallel Categories** diagrams to trace borrower journeys.
    *   **Correlation Heatmaps** for statistical factor analysis.
*   **📖 Procedural Story Engine**: Generates dynamic, realistic chronological activity logs (calls, legal notices, EMI bounces) for every borrower based on their risk profile.
*   **🎯 Risk Segmentation**: Uses K-Means clustering and predictive modeling to group borrowers into actionable risk tiers.

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Modular UI Architecture)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Random Forest, K-Means)
- **Explainability**: SHAP (Shapley Values)
- **Visualization**: Plotly Express & Graph Objects

## 📂 Project Structure

The project follows a modular, production-ready directory structure:

```
loanguard/
├── app.py                  # Main Application Entry Point (Router)
├── config/                 # Centralized Configuration & Settings
├── data/                   # Datasets (loan-recovery.csv)
├── models/                 # Trained Model Artifacts (.joblib)
├── notebooks/              # Jupyter Notebooks for Analysis
├── scripts/                # Utility & Training Scripts
├── src/                    # Source Code
│   ├── ui/                 # UI Modules (Dashboard, Portfolio, etc.)
│   ├── features.py         # Feature Engineering Logic
│   ├── model.py            # ML Model Wrappers
│   └── utils.py            # Helper Functions & Story Engine
└── tests/                  # Unit Tests
```

## ⚡ Quick Start

### Prerequisites
- Python 3.8 or higher

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/BLShaw/LoanGuard
    cd LoanGuard
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the Models**
    Before running the dashboard, you must generate the risk and segmentation models.
    ```bash
    python scripts/train_model.py
    ```

4.  **Run the Dashboard**
    ```bash
    streamlit run app.py
    ```

## 🖥️ Dashboard Views

### 1. Executive Dashboard
Real-time overview of portfolio health, recovery rates, and exposure using Sunburst charts.

### 2. Portfolio Management
A filterable, high-density view of the loan book with "Borrower Journey" flow diagrams (Parallel Categories).

### 3. Risk Analysis Engine
Deep dive into model performance, feature correlations, and statistical risk distribution.

### 4. Customer 360
A detailed single-borrower view featuring:
*   **Risk Speedometer**: Gauge chart for instant risk assessment.
*   **XAI Analysis**: "Why this Score?" chart showing top positive/negative factors.
*   **Activity Log**: Dynamically generated history of interactions and payments.

## 🤝 Contributing

1.  Fork the repository
2.  Create a feature branch (`git checkout -b feature/NewFeature`)
3.  Commit your changes (`git commit -m 'Add some New Feature'`)
4.  Push to the branch (`git push origin feature/NewFeature`)
5.  Open a Pull Request

## 📄 License

This project is available under the [MIT License](LICENSE).