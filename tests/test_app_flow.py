"""End-to-End UI Logic Tests using Streamlit AppTest."""
import os
import pytest
from streamlit.testing.v1 import AppTest

# Path to the main app file
APP_PATH = "app.py"

@pytest.fixture
def app():
    """Fixture to load the Streamlit app before each test."""
    if not os.path.exists(APP_PATH):
        pytest.skip(f"App file {APP_PATH} not found")
    return AppTest.from_file(APP_PATH, default_timeout=30)

def test_app_loads(app):
    """Verify the app loads without error."""
    app.run()
    assert not app.exception
    assert len(app.title) > 0
    assert "LoanGuard" in app.title[0].value

def test_risk_analysis_flow(app):
    """Test the Risk Analysis Engine flow."""
    app.run()

    sidebar = app.sidebar
    if not sidebar.radio:
        pytest.skip("Sidebar navigation not found")
    
    nav = sidebar.radio[0] 
    nav.set_value("Risk Analysis Engine").run()
    
    assert not app.exception
    return

def test_what_if_simulator_flow(app):
    """Test the What-If Simulator flow."""
    app.run()
    sidebar = app.sidebar
    if sidebar.radio:
        sidebar.radio[0].set_value("What-If Simulator").run()
    
    assert not app.exception
    assert any("What-If" in h.body for h in app.header)

def test_dashboard_flow(app):
    """Test the Dashboard flow."""
    app.run()
    sidebar = app.sidebar
    if sidebar.radio:
        sidebar.radio[0].set_value("Dashboard Overview").run()
    
    assert not app.exception
    # Check for metrics
    assert len(app.metric) > 0
