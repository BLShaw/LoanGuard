# LoanGuard: A Smart Loan Recovery System

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/streamlit-1.24.0-yellow)

A machine learning-powered system that uses historical loan repayment data, borrower profiles, and payment behaviors to optimize collection efforts, minimize recovery costs, and maximize loan repayments.

## Features

- **Data Visualization**: Interactive charts showing loan amount distribution, payment history analysis, and recovery status
- **Borrower Segmentation**: K-Means clustering to group borrowers into risk segments
- **Risk Prediction**: Random Forest model to identify high-risk borrowers
- **Recovery Strategies**: Dynamic strategy assignment based on risk scores
- **Individual Borrower Lookup**: Search and view detailed information for specific borrowers
- **Professional Dashboard**: Clean, responsive interface with KPIs and visualizations

## Quick Start

### Prerequisites
- Python 3.7 or higher

### Installation
1. Clone the repository
   ```bash
   git clone https://github.com/BLShaw/LoanGuard
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Dashboard Sections

The application provides an interactive dashboard with the following sections:

### Key Performance Indicators
Shows overall metrics like:
- Total Borrowers
- High Risk Borrowers 
- Recovery Rate
- Average Missed Payments

### Borrower Segments
Displays borrower categorization using K-Means clustering:
- Moderate Income, High Loan Burden
- High Income, Low Default Risk
- Moderate Income, Medium Risk
- High Loan, Higher Default Risk

### Risk Prediction Model
Shows the performance of the Random Forest risk prediction model with:
- Training and test accuracy metrics
- Recovery strategy distribution
- Sample risk predictions

### Data Visualizations
Interactive charts including:
- Loan Amount Distribution
- Payment History vs Recovery Status
- Borrower Segments Visualization

### Individual Borrower Lookup
Search for specific borrowers and view their complete profile including risk assessment.

## Technology Stack

- **Backend**: Python 3.7+
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly
- **Clustering**: K-Means from Scikit-learn

## Project Structure

```
├── app.py                         # Streamlit web application
├── smart_loan_recovery_system.ipynb # Original analysis notebook
├── loan-recovery.csv              # Dataset file
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── LICENSE                        # License file
└── .gitignore                     # Git ignore file
```

## Security Features

- File path validation to prevent path traversal
- Input validation for data access
- File size limits to prevent resource exhaustion
- Proper error handling
- Data access controls

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add some New Feature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## License

This project is available under the [MIT License](LICENSE).

## Acknowledgments

- The loan recovery dataset used in this project
- Streamlit community for the excellent framework
- Plotly for interactive visualization capabilities