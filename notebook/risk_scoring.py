# ============================================================
# PROJECT: Career Progression & Promotion Gap Analysis
# Company: Palo Alto Networks
# Step 4: Risk Scoring + Retention Opportunity Identification
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import sys
sys.stdout.reconfigure(encoding='utf-8')

# ──────────────────────────────────────────────
# 1. LOAD CLUSTERED DATASET
# ──────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Clustered Dataset")
print("=" * 60)

df = pd.read_csv('data/hr_data_clustered.csv')
print(f"✅ Loaded! Rows: {df.shape[0]} | Columns: {df.shape[1]}")

# ──────────────────────────────────────────────
# 2. PROMOTION GAP RISK SCORE (Numeric 0-100)
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Calculating Promotion Gap Risk Score (0-100)")
print("=" * 60)

def calculate_risk_score(row):
    score = 0

    # Factor 1: Promotion Gap Ratio (max 30 points)
    score += min(row['Promotion_Gap_Ratio'] * 50, 30)

    # Factor 2: Role Stagnation Index (max 25 points)
    score += min(row['Role_Stagnation_Index'] * 40, 25)

    # Factor 3: Years Since Last Promotion (max 20 points)
    score += min(row['YearsSinceLastPromotion'] * 2, 20)

    # Factor 4: Low Career Growth Rate (max 15 points)
    if row['Career_Growth_Rate'] < 0.1:
        score += 15
    elif row['Career_Growth_Rate'] < 0.2:
        score += 8

    # Factor 5: Low Training (max 10 points)
    if row['TrainingTimesLastYear'] == 0:
        score += 10
    elif row['TrainingTimesLastYear'] == 1:
        score += 5

    return round(min(score, 100), 2)

df['Risk_Score'] = df.apply(calculate_risk_score, axis=1)

print("✅ Risk Score calculated (0 = No Risk, 100 = Highest Risk)")
print(f"\n   Min Score  : {df['Risk_Score'].min()}")
print(f"   Max Score  : {df['Risk_Score'].max()}")
print(f"   Avg Score  : {df['Risk_Score'].mean():.2f}")
print(f"   Median     : {df['Risk_Score'].median():.2f}")

# ──────────────────────────────────────────────
# 3. RISK CATEGORY (Low / Medium / High)
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Risk Category Assignment")
print("=" * 60)

def assign_risk_category(score):
    if score >= 60:
        return 'High'
    elif score >= 30:
        return 'Medium'
    else:
        return 'Low'

df['Risk_Category'] = df['Risk_Score'].apply(assign_risk_category)

cat_counts = df['Risk_Category'].value_counts()
print("\n📊 Risk Category Distribution:")
for cat in ['Low', 'Medium', 'High']:
    count = cat_counts.get(cat, 0)
    pct = count / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"   {cat:6s}: {count:4d} employees ({pct:.1f}%) {bar}")

# ──────────────────────────────────────────────
# 4. KPI CALCULATIONS
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: KPI Calculations")
print("=" * 60)

# KPI 1: Retention Opportunity Index
# Employees who are NOT yet disengaged but show stagnation signals
df['Retention_Opportunity'] = (
    (df['Attrition'] == 0) &
    (df['Risk_Category'].isin(['Medium', 'High'])) &
    (df['JobSatisfaction'] >= 2)
).astype(int)

# KPI 2: Training Need Indicator
df['Training_Need'] = (
    (df['TrainingTimesLastYear'] <= 1) &
    (df['Risk_Score'] > 30)
).astype(int)

# KPI 3: Manager Stability Impact
df['Manager_Impact'] = (
    (df['Manager_Stability_Indicator'] < 0.3) &
    (df['Risk_Score'] > 40)
).astype(int)

retention_opps = df['Retention_Opportunity'].sum()
training_needs = df['Training_Need'].sum()
manager_impacts = df['Manager_Impact'].sum()

print(f"\n📌 KPI Results:")
print(f"   Retention Opportunity Index  : {retention_opps} employees need career intervention")
print(f"   Training Need Indicator      : {training_needs} employees need training")
print(f"   Manager Stability Impact     : {manager_impacts} employees affected by manager change")

# ──────────────────────────────────────────────
# 5. SUGGESTED ACTIONS
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Suggested Retention Actions")
print("=" * 60)

def suggest_action(row):
    actions = []
    if row['YearsSinceLastPromotion'] >= 3:
        actions.append("Promotion Review")
    if row['TrainingTimesLastYear'] <= 1:
        actions.append("Training Program")
    if row['Manager_Stability_Indicator'] < 0.3:
        actions.append("Manager Assignment Review")
    if row['Role_Stagnation_Index'] > 0.6:
        actions.append("Role Rotation")
    if row['JobSatisfaction'] <= 2:
        actions.append("Engagement Check-in")
    if not actions:
        actions.append("Monitor")
    return " | ".join(actions)

df['Suggested_Action'] = df.apply(suggest_action, axis=1)
print("✅ Suggested actions assigned to all employees")

# Top actions summary
from collections import Counter
all_actions = []
for actions in df['Suggested_Action']:
    all_actions.extend(actions.split(" | "))
action_counts = Counter(all_actions)
print("\n📋 Top Recommended Actions:")
for action, count in action_counts.most_common():
    pct = count / len(df) * 100
    print(f"   {action:30s}: {count:4d} ({pct:.1f}%)")

# ──────────────────────────────────────────────
# 6. TOP AT-RISK EMPLOYEES
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Top 10 At-Risk Employees (Still Employed)")
print("=" * 60)

top_risk = df[df['Attrition'] == 0].nlargest(10, 'Risk_Score')[
    ['Age', 'Department', 'JobRole', 'YearsAtCompany',
     'YearsSinceLastPromotion', 'Risk_Score', 'Risk_Category',
     'Cluster_Label', 'Suggested_Action']
]
print(top_risk.to_string(index=False))

# ──────────────────────────────────────────────
# 7. DEPARTMENT-WISE RISK ANALYSIS
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Department-wise Risk Analysis")
print("=" * 60)

dept_risk = df.groupby('Department').agg(
    Avg_Risk_Score=('Risk_Score', 'mean'),
    High_Risk_Count=('Risk_Category', lambda x: (x == 'High').sum()),
    Retention_Opportunities=('Retention_Opportunity', 'sum'),
    Attrition_Rate=('Attrition', 'mean')
).round(2)
dept_risk['Attrition_Rate'] = (dept_risk['Attrition_Rate'] * 100).round(1)
print(dept_risk.to_string())

# ──────────────────────────────────────────────
# 8. VISUALIZATIONS
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Creating Risk Score Visualizations...")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Palo Alto Networks — Risk Scoring & Retention Analysis',
             fontsize=15, fontweight='bold')

# Plot 1: Risk Score Distribution
ax1 = axes[0, 0]
ax1.hist(df['Risk_Score'], bins=30, color='#e74c3c', edgecolor='white', alpha=0.8)
ax1.axvline(x=30, color='orange', linestyle='--', linewidth=2, label='Medium threshold')
ax1.axvline(x=60, color='red', linestyle='--', linewidth=2, label='High threshold')
ax1.set_title('Risk Score Distribution', fontweight='bold')
ax1.set_xlabel('Risk Score'); ax1.set_ylabel('Count')
ax1.legend()

# Plot 2: Risk Category Pie
ax2 = axes[0, 1]
cat_vals = [cat_counts.get(c, 0) for c in ['Low', 'Medium', 'High']]
ax2.pie(cat_vals, labels=['Low', 'Medium', 'High'],
        colors=['#2ecc71', '#f39c12', '#e74c3c'],
        autopct='%1.1f%%', startangle=90)
ax2.set_title('Risk Category Distribution', fontweight='bold')

# Plot 3: Risk Score vs Attrition
ax3 = axes[0, 2]
stayed = df[df['Attrition'] == 0]['Risk_Score']
left   = df[df['Attrition'] == 1]['Risk_Score']
ax3.boxplot([stayed, left], labels=['Stayed', 'Left'],
            patch_artist=True,
            boxprops=dict(facecolor='#3498db', alpha=0.6))
ax3.set_title('Risk Score vs Attrition', fontweight='bold')
ax3.set_ylabel('Risk Score')

# Plot 4: Department Risk
ax4 = axes[1, 0]
dept_risk_sorted = dept_risk['Avg_Risk_Score'].sort_values()
ax4.barh(dept_risk_sorted.index, dept_risk_sorted.values,
         color='#9b59b6', alpha=0.8)
ax4.set_title('Avg Risk Score by Department', fontweight='bold')
ax4.set_xlabel('Average Risk Score')

# Plot 5: Retention Opportunity by Department
ax5 = axes[1, 1]
ret_dept = df[df['Retention_Opportunity'] == 1].groupby('Department').size()
ax5.bar(ret_dept.index, ret_dept.values,
        color='#1abc9c', edgecolor='white')
ax5.set_title('Retention Opportunities by Department', fontweight='bold')
ax5.set_ylabel('Employees Needing Intervention')
for i, (dept, val) in enumerate(ret_dept.items()):
    ax5.text(i, val + 1, str(val), ha='center', fontweight='bold')

# Plot 6: Risk Score by Cluster
ax6 = axes[1, 2]
cluster_risk = df.groupby('Cluster_Label')['Risk_Score'].mean().sort_values()
colors_cl = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db', '#9b59b6']
bars = ax6.barh([label[:20] for label in cluster_risk.index],
                cluster_risk.values,
                color=colors_cl[:len(cluster_risk)])
ax6.set_title('Avg Risk Score by Cluster', fontweight='bold')
ax6.set_xlabel('Average Risk Score')

plt.tight_layout()
plt.savefig('data/risk_scoring_plots.png', dpi=150, bbox_inches='tight')
print("✅ Plots saved: data/risk_scoring_plots.png")
plt.show()

# ──────────────────────────────────────────────
# 9. SAVE FINAL DATASET
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Saving Final Dataset")
print("=" * 60)

df.to_csv('data/hr_data_final.csv', index=False)
print(f"✅ Saved: data/hr_data_final.csv")
print(f"   Rows: {len(df)} | Columns: {len(df.columns)}")

print("\n" + "=" * 60)
print("🎉 RISK SCORING COMPLETE!")
print("=" * 60)
print(f"""
📊 Final Summary:
   Total Employees          : {len(df)}
   High Risk Employees      : {cat_counts.get('High', 0)}
   Medium Risk Employees    : {cat_counts.get('Medium', 0)}
   Low Risk Employees       : {cat_counts.get('Low', 0)}
   Retention Opportunities  : {retention_opps}
   Training Needs           : {training_needs}
   Manager Impact Cases     : {manager_impacts}

➡️  Next Step: Run streamlit_app.py (Step 5 — Final Dashboard!)
""")
