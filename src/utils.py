from datetime import datetime, timedelta
import pandas as pd
import random
import streamlit as st

def format_currency(value):
    return f"${value:,.0f}"

def format_percentage(value):
    return f"{value:.1f}%"

def get_status_color(status):
    if status == 'Fully Recovered': return 'green'
    if status == 'Partially Recovered': return 'orange'
    return 'red'

def generate_borrower_history(data):
    """Procedurally generates a realistic activity log based on borrower attributes."""
    # Seed random based on borrower ID for reproducibility
    borrower_seed = hash(data.get('Borrower_ID', 'default')) % (2**32)
    random.seed(borrower_seed)
    
    history = []
    today = datetime.now()
    
    # 1. Today's System Action
    history.append({
        "Date": today.strftime("%Y-%m-%d"),
        "Action": "AI Risk Model Evaluation",
        "Agent": "System / Loanguard Core",
        "Result": f"Risk Score Updated to {data['Risk_Score']:.3f}"
    })
    
    # 2. Recovery/Closure Events
    if data['Recovery_Status'] == 'Fully Recovered':
        days_ago = random.randint(1, 10)
        history.append({
            "Date": (today - timedelta(days=days_ago)).strftime("%Y-%m-%d"),
            "Action": "Loan Closure Processing",
            "Agent": "Senior Officer",
            "Result": "Account Closed - Full Payment Received"
        })
    elif data['Recovery_Status'] == 'Written Off':
         days_ago = random.randint(1, 5)
         history.append({
            "Date": (today - timedelta(days=days_ago)).strftime("%Y-%m-%d"),
            "Action": "Portfolio Write-off",
            "Agent": "Risk Manager",
            "Result": "Asset Classified as NPA"
        })

    # 3. Legal Actions
    if data['Legal_Action_Taken'] == 'Yes':
        days_ago = random.randint(10, 45)
        history.append({
            "Date": (today - timedelta(days=days_ago)).strftime("%Y-%m-%d"),
            "Action": "Legal Notice Dispatch",
            "Agent": "Legal Dept",
            "Result": "Notice Served via Reg. Post"
        })
        
        # Preceding review
        history.append({
            "Date": (today - timedelta(days=days_ago + 2)).strftime("%Y-%m-%d"),
            "Action": "Legal Case File Preparation",
            "Agent": "Legal Officer",
            "Result": "Approved for Litigation"
        })

    # 4. Collection Attempts based on Method
    attempts = int(data.get('Collection_Attempts', 0))
    method = data.get('Collection_Method', 'None')
    
    for i in range(attempts):
        # Stagger attempts back in time
        days_back = random.randint(2, 60) + (i * 5)
        date_str = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        if method == 'Calls':
            action = "Outbound Collection Call"
            result = random.choice(["Customer Hung Up", "Promise to Pay", "No Answer", "Number Busy"])
        elif method == 'Settlement Offer':
            action = "Settlement Proposal Sent"
            result = "Offer valid for 7 days"
        elif method == 'Legal Notice':
            # Covered in Legal Action, but adding specific drafting log
            action = "Drafting Warning Letter"
            result = "Template Generated"
        else:
            action = f"{method} Initiated"
            result = "Processing"
            
        history.append({
            "Date": date_str,
            "Action": action,
            "Agent": "Collection Agent",
            "Result": result
        })

    # 5. Missed Payments (The Root Cause)
    missed = int(data['Num_Missed_Payments'])
    if missed > 0:
        # Generate EMI bounce logs
        current_dpd = int(data['Days_Past_Due'])
        # Estimate bounce dates based on DPD
        
        # Most recent bounce
        if current_dpd > 0:
            bounce_date = today - timedelta(days=current_dpd)
            history.append({
                "Date": bounce_date.strftime("%Y-%m-%d"),
                "Action": "EMI Auto-Debit Attempt",
                "Agent": "Banking System",
                "Result": "FAILED - Insufficient Funds"
            })
            
        # Previous bounces (simulated)
        for i in range(1, missed):
            # Assume monthly EMI
            prev_bounce = today - timedelta(days=current_dpd + (i * 30))
            history.append({
                "Date": prev_bounce.strftime("%Y-%m-%d"),
                "Action": "EMI Auto-Debit Attempt",
                "Agent": "Banking System",
                "Result": "FAILED - Insufficient Funds"
            })

    # Sort by Date Descending
    df_hist = pd.DataFrame(history)
    if not df_hist.empty:
        df_hist['DateObj'] = pd.to_datetime(df_hist['Date'])
        df_hist = df_hist.sort_values(by='DateObj', ascending=False).drop(columns=['DateObj'])
    
    # Restore random state to avoid affecting other code
    random.seed()
    return df_hist
