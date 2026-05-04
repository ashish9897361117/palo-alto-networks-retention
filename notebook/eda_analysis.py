# ============================================================
# PROJECT: Career Progression & Promotion Gap Analysis
# Company: Palo Alto Networks
# Step 2: EDA + Feature Engineering
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
# 1. LOAD DATASET
# ──────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_csv(r'data/Palo Alto Networks.csv')

print(f"✅ Dataset Loaded!")
print(f"   Rows    : {df.shape[0]}")
print(f"   Columns : {df.shape[1]}")

# ──────────────────────────────────────────────
# 2. BASIC OVERVIEW
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Basic Overview")
print("=" * 60)

print("\n📋 First 3 rows:")
print(df.head(3))

print("\n❓ Missing Values:", df.isnull().sum().sum(), "(None - Clean Dataset!)")

attrition_counts = df['Attrition'].value_counts()
print(f"\n📊 Attrition — Stayed: {attrition_counts.get(0,0)} | Left: {attrition_counts.get(1,0)}")
print(f"   Attrition Rate: {df['Attrition'].mean()*100:.1f}%")

# ──────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Feature Engineering")
print("=" * 60)

df['Promotion_Gap_Ratio'] = np.where(
    df['YearsAtCompany'] > 0,
    df['YearsSinceLastPromotion'] / df['YearsAtCompany'], 0)

df['Role_Stagnation_Index'] = np.where(
    df['YearsAtCompany'] > 0,
    df['YearsInCurrentRole'] / df['YearsAtCompany'], 0)

df['Training_Intensity_Score'] = np.where(
    df['YearsAtCompany'] > 0,
    df['TrainingTimesLastYear'] / df['YearsAtCompany'],
    df['TrainingTimesLastYear'])

df['Manager_Stability_Indicator'] = np.where(
    df['YearsAtCompany'] > 0,
    df['YearsWithCurrManager'] / df['YearsAtCompany'], 0)

df['Career_Growth_Rate'] = np.where(
    df['TotalWorkingYears'] > 0,
    df['JobLevel'] / df['TotalWorkingYears'], 0)

print("✅ 5 New Features Created Successfully!")

# ──────────────────────────────────────────────
# 4. PROMOTION GAP RISK SCORING
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Promotion Gap Risk Scoring")
print("=" * 60)

def assign_risk(row):
    r = row['Promotion_Gap_Ratio']
    s = row['Role_Stagnation_Index']
    if r > 0.6 and s > 0.6:
        return 'High'
    elif r > 0.3 or s > 0.4:
        return 'Medium'
    else:
        return 'Low'

df['Promotion_Gap_Score'] = df.apply(assign_risk, axis=1)
score_counts = df['Promotion_Gap_Score'].value_counts()
for score in ['Low', 'Medium', 'High']:
    count = score_counts.get(score, 0)
    pct = count / len(df) * 100
    print(f"   {score:6s}: {count:4d} employees ({pct:.1f}%)")

# ──────────────────────────────────────────────
# 5. VISUALIZATIONS
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Creating Visualizations...")
print("=" * 60)

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('Palo Alto Networks — Career Progression EDA',
             fontsize=16, fontweight='bold')

new_features = ['Promotion_Gap_Ratio','Role_Stagnation_Index',
                'Training_Intensity_Score','Manager_Stability_Indicator','Career_Growth_Rate']

# Plot 1: Attrition Pie
ax1 = axes[0, 0]
ax1.pie([attrition_counts.get(0,0), attrition_counts.get(1,0)],
        labels=['Stayed','Left'], colors=['#2ecc71','#e74c3c'],
        autopct='%1.1f%%', startangle=90)
ax1.set_title('Attrition Distribution', fontweight='bold')

# Plot 2: Promotion Gap Ratio
ax2 = axes[0, 1]
ax2.hist(df['Promotion_Gap_Ratio'], bins=30, color='#3498db', edgecolor='white', alpha=0.8)
ax2.set_title('Promotion Gap Ratio', fontweight='bold')
ax2.set_xlabel('Ratio'); ax2.set_ylabel('Count')

# Plot 3: Role Stagnation Index
ax3 = axes[0, 2]
ax3.hist(df['Role_Stagnation_Index'], bins=30, color='#e67e22', edgecolor='white', alpha=0.8)
ax3.set_title('Role Stagnation Index', fontweight='bold')
ax3.set_xlabel('Index'); ax3.set_ylabel('Count')

# Plot 4: Risk Score Distribution
ax4 = axes[1, 0]
order = ['Low','Medium','High']
vals = [score_counts.get(s,0) for s in order]
bars = ax4.bar(order, vals, color=['#2ecc71','#f39c12','#e74c3c'], edgecolor='white')
ax4.set_title('Promotion Gap Risk Score', fontweight='bold')
ax4.set_ylabel('Employees')
for b, v in zip(bars, vals):
    ax4.text(b.get_x()+b.get_width()/2, b.get_height()+5, str(v), ha='center', fontweight='bold')

# Plot 5: Attrition Rate by Risk Score
ax5 = axes[1, 1]
atr_score = df.groupby('Promotion_Gap_Score')['Attrition'].mean()*100
atr_score = atr_score.reindex(order)
bars2 = ax5.bar(atr_score.index, atr_score.values, color=['#2ecc71','#f39c12','#e74c3c'], edgecolor='white')
ax5.set_title('Attrition Rate by Risk Score', fontweight='bold')
ax5.set_ylabel('Attrition Rate (%)')
for b, v in zip(bars2, atr_score.values):
    ax5.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{v:.1f}%', ha='center', fontweight='bold')

# Plot 6: YearsSinceLastPromotion Boxplot
ax6 = axes[1, 2]
ax6.boxplot([df[df['Attrition']==0]['YearsSinceLastPromotion'],
             df[df['Attrition']==1]['YearsSinceLastPromotion']],
            labels=['Stayed','Left'], patch_artist=True,
            boxprops=dict(facecolor='#3498db', alpha=0.6))
ax6.set_title('Years Since Promotion vs Attrition', fontweight='bold')
ax6.set_ylabel('Years Since Last Promotion')

# Plot 7: Dept-wise Promotion Gap
ax7 = axes[2, 0]
dept_gap = df.groupby('Department')['Promotion_Gap_Ratio'].mean().sort_values()
ax7.barh(dept_gap.index, dept_gap.values, color='#9b59b6', alpha=0.8)
ax7.set_title('Avg Promotion Gap by Department', fontweight='bold')
ax7.set_xlabel('Avg Promotion Gap Ratio')

# Plot 8: Scatter - Manager Stability vs Promotion Gap
ax8 = axes[2, 1]
ax8.scatter(df[df['Attrition']==0]['Manager_Stability_Indicator'],
            df[df['Attrition']==0]['Promotion_Gap_Ratio'],
            alpha=0.3, color='#2ecc71', label='Stayed', s=15)
ax8.scatter(df[df['Attrition']==1]['Manager_Stability_Indicator'],
            df[df['Attrition']==1]['Promotion_Gap_Ratio'],
            alpha=0.5, color='#e74c3c', label='Left', s=15)
ax8.set_title('Manager Stability vs Promotion Gap', fontweight='bold')
ax8.set_xlabel('Manager Stability'); ax8.set_ylabel('Promotion Gap Ratio')
ax8.legend()

# Plot 9: Correlation Heatmap
ax9 = axes[2, 2]
corr_cols = new_features + ['Attrition']
sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f',
            cmap='RdYlGn', ax=ax9, square=True, linewidths=0.5)
ax9.set_title('Feature Correlation Heatmap', fontweight='bold')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
print("✅ Plots saved: eda_plots.png")
plt.show()

# ──────────────────────────────────────────────
# 6. SAVE PROCESSED DATASET
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Saving Processed Dataset")
print("=" * 60)

df.to_csv('hr_data_processed.csv', index=False)
print(f"✅ Saved: hr_data_processed.csv")
print(f"   Rows: {len(df)} | Columns: {len(df.columns)}")
print(f"\n🎉 EDA Complete! Next: Run clustering.py")