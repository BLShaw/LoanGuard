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
<img width="1479" height="176" alt="image" src="https://github.com/user-attachments/assets/9a8bb23c-c615-4f8b-9d87-c522179c2afd" />

### Borrower Segments
Displays borrower categorization using K-Means clustering:
- Moderate Income, High Loan Burden
- High Income, Low Default Risk
- Moderate Income, Medium Risk
- High Loan, Higher Default Risk
<img width="1775" height="352" alt="image" src="https://github.com/user-attachments/assets/12842e0c-c5ea-4481-8d86-32e7eff415a7" />


### Risk Prediction Model
Shows the performance of the Random Forest risk prediction model with:
- Training and test accuracy metrics
- Recovery strategy distribution
- Sample risk predictions
<img width="1026" height="81" alt="image" src="https://github.com/user-attachments/assets/39768d03-8b0e-461a-a000-c59ec68bc0b5" />


### Data Visualizations
Interactive charts including:
- Loan Amount Distribution
- Payment History vs Recovery Status
- Borrower Segments Visualization
<img width="1821" height="490" alt="image" src="https://github.com/user-attachments/assets/311deda6-cc41-4c48-8ae9-c8c58f79dc14" />
<img width="1766" height="550" alt="image" src="https://github.com/user-attachments/assets/4396772b-2e46-40e8-a28e-59783663ac67" />

### Individual Borrower Lookup
Search for specific borrowers and view their complete profile including risk assessment.
<img width="1774" height="508" alt="image" src="https://github.com/user-attachments/assets/c3b4543d-556d-4b7c-bcc2-d165c2cf02f7" />


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
