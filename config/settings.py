import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

# File Paths
DATA_PATH = os.path.join(DATA_DIR, 'loan-recovery.csv')
RISK_MODEL_PATH = os.path.join(MODEL_DIR, 'risk_model.joblib')
SEGMENT_MODEL_PATH = os.path.join(MODEL_DIR, 'segment_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

# App Settings
APP_TITLE = "LoanGuard | Banking Intelligence"
APP_ICON = "🏦"
APP_VERSION = "v2.2.0 Enterprise"

# Theme Colors
COLOR_SUCCESS = "#28a745"
COLOR_WARNING = "#ffc107"
COLOR_DANGER = "#dc3545"
COLOR_PRIMARY = "#1a237e"
