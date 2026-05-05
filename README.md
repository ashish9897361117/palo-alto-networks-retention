# 🔒 Palo Alto Networks — Career Progression & Retention Analysis

A comprehensive **Data Science & HR Analytics** project analyzing employee career progression patterns, promotion gaps, and attrition risks to optimize retention strategies.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

---

## 📊 Project Overview

This project identifies **high-risk employees** and **retention opportunities** using machine learning clustering and risk scoring models. It provides actionable insights through an interactive Streamlit dashboard for HR teams and management.

### 🎯 Key Objectives
- Analyze career progression patterns across the organization
- Identify promotion gap risks and stagnation signals
- Build ML models to cluster employees by career trajectory
- Create risk scores to prioritize retention interventions
- Deliver insights via interactive dashboard

---

## ✨ Features

### 🗺️ **Career Path Clustering**
- K-Means clustering to identify 4-5 distinct career trajectories
- PCA visualization of employee career paths
- Hierarchical clustering for validation
- Cluster labeling: Fast-Track Performers, Promotion-Stalled, etc.

### 📈 **Promotion Gap Analysis**
- Custom risk scoring (0-100 scale)
- Feature engineering: Promotion Gap Ratio, Role Stagnation Index
- Department and role-level promotion gap insights

### 🎯 **Retention Opportunity Detection**
- Identifies at-risk but still-engaged employees
- Suggests targeted interventions (training, promotion review, role rotation)
- Calculates retention opportunity index

### 👔 **Managerial Insights**
- Manager stability impact analysis
- Team stagnation signals
- Actionable recommendations for leadership

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.11+ |
| **Data Analysis** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (K-Means, PCA, Hierarchical Clustering) |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Streamlit |
| **Development** | Jupyter Notebook, VS Code |

---

## 📁 Project Structure

```
palo-alto-networks-retention/
│
├── app/
│   └── streamlit_app.py          # Interactive dashboard
│
├── data/
│   ├── Palo_Alto_Networks.csv    # Original dataset
│   ├── hr_data_processed.csv     # After feature engineering
│   ├── hr_data_clustered.csv     # After clustering
│   └── hr_data_final.csv         # Final dataset with risk scores
│
├── notebooks/
│   ├── eda_analysis.py           # Exploratory Data Analysis
│   ├── clustering.py             # K-Means clustering
│   └── risk_scoring.py           # Risk score calculation
│
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/palo-alto-networks-retention.git
cd palo-alto-networks-retention
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Analysis Scripts
```bash
cd notebooks
python eda_analysis.py      # Step 1: EDA + Feature Engineering
python clustering.py         # Step 2: ML Clustering
python risk_scoring.py       # Step 3: Risk Scoring
```

### Step 4: Launch Dashboard
```bash
cd ../app
streamlit run streamlit_app.py
```

Dashboard will open at: `https://palo-alto-networks-retention-hjuunavjffc7q4dsnppjyz.streamlit.app/`

---

## 📊 Key Findings

### Employee Risk Distribution
- **🔴 High Risk:** 287 employees (19.5%)
- **🟠 Medium Risk:** 674 employees (45.9%)
- **🟢 Low Risk:** 509 employees (34.6%)

### Career Clusters Identified
1. **🚀 Fast-Track Performers** (15.3%) — High growth rate, higher attrition risk (32.9%)
2. **🟠 Promotion-Stalled** (84.7%) — Moderate stagnation, moderate attrition (13.1%)

### Actionable Insights
- **678 employees** need career intervention
- **88 employees** require immediate training programs
- **75 employees** affected by manager instability

### Top Recommended Actions
| Action | Employees Affected |
|--------|-------------------|
| Role Rotation | 826 (56.2%) |
| Engagement Check-in | 569 (38.7%) |
| Promotion Review | 373 (25.4%) |

---

## 📸 Dashboard Preview

### Career Path Clustering
Interactive PCA visualization showing distinct employee career trajectories with cluster-based color coding.

### Promotion Gap Monitor
Department-wise promotion gap analysis with boxplots and histograms identifying stagnation hotspots.

### Retention Opportunity Panel
Risk score distributions and targeted intervention recommendations for at-risk employees.

### Managerial Insights
Manager stability vs risk correlation analysis for team-level decision making.

---

## 🧪 Methodology

### 1. Feature Engineering
Created 5 custom features:
- **Promotion Gap Ratio** = Years Since Last Promotion / Years at Company
- **Role Stagnation Index** = Years in Current Role / Years at Company
- **Training Intensity Score** = Training Times / Years at Company
- **Manager Stability Indicator** = Years with Current Manager / Years at Company
- **Career Growth Rate** = Job Level / Total Working Years

### 2. Machine Learning
- **K-Means Clustering** with Elbow Method and Silhouette Score optimization
- **Hierarchical Clustering** for validation (~overlap validation)
- **PCA** for 2D visualization

### 3. Risk Scoring
Multi-factor risk score (0-100):
- Promotion Gap Ratio (30 points)
- Role Stagnation Index (25 points)
- Years Since Last Promotion (20 points)
- Career Growth Rate (15 points)
- Training Frequency (10 points)

---

## 📈 Future Enhancements

- [ ] Add predictive attrition model (Random Forest/XGBoost)
- [ ] Integrate real-time data pipeline
- [ ] Add sentiment analysis from employee feedback
- [ ] Build manager-specific dashboards
- [ ] Deploy automated email alerts for high-risk cases

---

## 👨‍💻 Author

**Your Name**
- LinkedIn: https://www.linkedin.com/in/ashish-kushwah-382a722b1/
- GitHub: [@ashish9897361117]
- Email: ashishkush9758581707@gmail.com

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset:** IBM HR Analytics Employee Attrition Dataset
- **Project Guidance:** Unified Mentor
- **Inspiration:** Real-world HR analytics challenges in tech companies

---

## 📞 Contact

For questions, suggestions, or collaboration opportunities:
- Open an [Issue](https://github.com/ashish9897361117/palo-alto-networks-retention/issues)
- Connect on [LinkedIn](https://www.linkedin.com/in/ashish-kushwah-382a722b1/)

---

<div align="center">
  
**⭐ Star this repo if you found it helpful!**

Made with ❤️ for HR Analytics & Data Science

</div>

