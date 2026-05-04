# ============================================================
# PROJECT: Career Progression & Promotion Gap Analysis
# Company: Palo Alto Networks
# Step 5: Streamlit Web Application (Final Dashboard)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Palo Alto Networks — Career Intelligence",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #fff; }
    .metric-label { font-size: 0.85rem; color: #9ca3af; margin-top: 4px; }
    .risk-high   { color: #ef4444; }
    .risk-medium { color: #f59e0b; }
    .risk-low    { color: #10b981; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
        border-left: 4px solid #6366f1;
        padding-left: 12px;
        margin: 20px 0 10px 0;
    }
    div[data-testid="stSidebar"] { background-color: #13151f; }
    .stTabs [data-baseweb="tab-list"] { background-color: #1a1d2e; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: #9ca3af; }
    .stTabs [aria-selected="true"] { color: #6366f1 !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
import os

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'data', 'hr_data_final.csv')
    df = pd.read_csv(file_path)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.info("Run these files first:\n1. notebooks/eda_analysis.py\n2. notebooks/clustering.py\n3. notebooks/risk_scoring.py")
    st.stop()

# ──────────────────────────────────────────────
# SIDEBAR FILTERS
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Palo_Alto_Networks_logo_%282022%29.svg/320px-Palo_Alto_Networks_logo_%282022%29.svg.png", width=200)
    st.markdown("---")
    st.markdown("### 🎛️ Filters")

    # Department Filter
    departments = ['All'] + sorted(df['Department'].unique().tolist())
    selected_dept = st.selectbox("🏢 Department", departments)

    # Job Role Filter
    if selected_dept != 'All':
        roles = ['All'] + sorted(df[df['Department'] == selected_dept]['JobRole'].unique().tolist())
    else:
        roles = ['All'] + sorted(df['JobRole'].unique().tolist())
    selected_role = st.selectbox("👤 Job Role", roles)

    # Risk Category Filter
    risk_options = ['All', 'High', 'Medium', 'Low']
    selected_risk = st.selectbox("⚠️ Risk Category", risk_options)

    # Promotion Gap Threshold Slider
    st.markdown("### 📊 Thresholds")
    promo_threshold = st.slider("Promotion Gap Threshold", 0.0, 1.0, 0.3, 0.05)
    risk_score_min = st.slider("Min Risk Score", 0, 100, 0)

    # Career Stage
    career_stages = ['All', 'Early (0-5 yrs)', 'Mid (5-15 yrs)', 'Senior (15+ yrs)']
    selected_stage = st.selectbox("📈 Career Stage", career_stages)

    st.markdown("---")
    st.markdown("### 📁 Data Info")
    st.info(f"Total Employees: **{len(df):,}**")

# ──────────────────────────────────────────────
# APPLY FILTERS
# ──────────────────────────────────────────────
filtered = df.copy()

if selected_dept != 'All':
    filtered = filtered[filtered['Department'] == selected_dept]
if selected_role != 'All':
    filtered = filtered[filtered['JobRole'] == selected_role]
if selected_risk != 'All':
    filtered = filtered[filtered['Risk_Category'] == selected_risk]
if selected_stage == 'Early (0-5 yrs)':
    filtered = filtered[filtered['YearsAtCompany'] <= 5]
elif selected_stage == 'Mid (5-15 yrs)':
    filtered = filtered[(filtered['YearsAtCompany'] > 5) & (filtered['YearsAtCompany'] <= 15)]
elif selected_stage == 'Senior (15+ yrs)':
    filtered = filtered[filtered['YearsAtCompany'] > 15]

filtered = filtered[
    (filtered['Promotion_Gap_Ratio'] >= promo_threshold) |
    (filtered['Risk_Score'] >= risk_score_min)
]

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg, #1e1b4b, #312e81);
            padding: 25px 30px; border-radius: 16px; margin-bottom: 25px;
            border: 1px solid #4338ca;'>
    <h1 style='color: white; margin: 0; font-size: 1.8rem;'>
        🔒 Palo Alto Networks
    </h1>
    <p style='color: #a5b4fc; margin: 6px 0 0 0; font-size: 1rem;'>
        Career Progression & Promotion Gap Analysis — Retention Intelligence Dashboard
    </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TOP KPI METRICS
# ──────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)

total        = len(filtered)
high_risk    = len(filtered[filtered['Risk_Category'] == 'High'])
medium_risk  = len(filtered[filtered['Risk_Category'] == 'Medium'])
ret_opps     = filtered['Retention_Opportunity'].sum()
training_n   = filtered['Training_Need'].sum()
avg_risk     = filtered['Risk_Score'].mean()

with c1:
    st.metric("👥 Employees", f"{total:,}")
with c2:
    st.metric("🔴 High Risk", f"{high_risk:,}", delta=f"{high_risk/total*100:.1f}%")
with c3:
    st.metric("🟠 Medium Risk", f"{medium_risk:,}", delta=f"{medium_risk/total*100:.1f}%")
with c4:
    st.metric("🎯 Retention Opps", f"{int(ret_opps):,}")
with c5:
    st.metric("📚 Training Needs", f"{int(training_n):,}")
with c6:
    st.metric("📊 Avg Risk Score", f"{avg_risk:.1f}/100")

st.markdown("---")

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Career Path Clustering",
    "📈 Promotion Gap Monitor",
    "🎯 Retention Opportunity",
    "👔 Managerial Insights"
])

# ════════════════════════════════════════════
# TAB 1: CAREER PATH CLUSTERING
# ════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Career Path Clustering Dashboard</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1])

    with col1:
        # PCA Scatter Plot
        fig_pca = px.scatter(
            filtered, x='PCA1', y='PCA2',
            color='Cluster_Label',
            hover_data=['Department', 'JobRole', 'YearsAtCompany', 'Risk_Score'],
            title='Employee Career Clusters (PCA 2D View)',
            color_discrete_sequence=px.colors.qualitative.Bold,
            template='plotly_dark'
        )
        fig_pca.update_traces(marker=dict(size=6, opacity=0.75))
        fig_pca.update_layout(height=420, legend=dict(orientation='h', y=-0.2))
        st.plotly_chart(fig_pca, use_container_width=True)

    with col2:
        # Cluster Distribution Donut
        cluster_counts = filtered['Cluster_Label'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        fig_donut = px.pie(
            cluster_counts, names='Cluster', values='Count',
            hole=0.55, title='Cluster Distribution',
            color_discrete_sequence=px.colors.qualitative.Bold,
            template='plotly_dark'
        )
        fig_donut.update_layout(height=420, legend=dict(orientation='h', y=-0.25, font=dict(size=10)))
        st.plotly_chart(fig_donut, use_container_width=True)

    # Cluster Feature Heatmap
    st.markdown('<div class="section-header">Cluster Feature Comparison</div>', unsafe_allow_html=True)
    feature_cols = ['Promotion_Gap_Ratio', 'Role_Stagnation_Index',
                    'Career_Growth_Rate', 'Training_Intensity_Score',
                    'Manager_Stability_Indicator']

    cluster_avg = filtered.groupby('Cluster_Label')[feature_cols].mean().round(3)
    fig_heat = px.imshow(
        cluster_avg,
        text_auto=True, aspect='auto',
        color_continuous_scale='RdYlGn',
        title='Average Feature Values per Cluster',
        template='plotly_dark'
    )
    fig_heat.update_layout(height=300)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Cluster Summary Table
    st.markdown('<div class="section-header">Cluster Summary</div>', unsafe_allow_html=True)
    cluster_summary = filtered.groupby('Cluster_Label').agg(
        Employees=('Cluster_Label', 'count'),
        Avg_Risk_Score=('Risk_Score', 'mean'),
        Attrition_Rate=('Attrition', 'mean'),
        Avg_Promo_Gap=('Promotion_Gap_Ratio', 'mean'),
        Avg_Stagnation=('Role_Stagnation_Index', 'mean')
    ).round(2).reset_index()
    cluster_summary['Attrition_Rate'] = (cluster_summary['Attrition_Rate'] * 100).round(1).astype(str) + '%'
    st.dataframe(cluster_summary, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════
# TAB 2: PROMOTION GAP MONITOR
# ════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Promotion Gap Monitor</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Years Since Last Promotion by Dept
        fig_promo = px.box(
            filtered, x='Department', y='YearsSinceLastPromotion',
            color='Department',
            title='Years Since Last Promotion by Department',
            template='plotly_dark',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_promo.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_promo, use_container_width=True)

    with col2:
        # Promotion Gap Ratio Distribution
        fig_gap = px.histogram(
            filtered, x='Promotion_Gap_Ratio',
            color='Risk_Category',
            nbins=30,
            title='Promotion Gap Ratio Distribution',
            template='plotly_dark',
            color_discrete_map={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}
        )
        fig_gap.update_layout(height=380)
        st.plotly_chart(fig_gap, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Role Stagnation by Job Level
        fig_stag = px.box(
            filtered, x='JobLevel', y='Role_Stagnation_Index',
            color='Risk_Category',
            title='Role Stagnation Index by Job Level',
            template='plotly_dark',
            color_discrete_map={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}
        )
        fig_stag.update_layout(height=380)
        st.plotly_chart(fig_stag, use_container_width=True)

    with col4:
        # Top Stagnated Job Roles
        role_stag = filtered.groupby('JobRole')['YearsSinceLastPromotion'].mean().sort_values(ascending=True).reset_index()
        fig_roles = px.bar(
            role_stag, x='YearsSinceLastPromotion', y='JobRole',
            orientation='h',
            title='Avg Years Since Promotion by Job Role',
            template='plotly_dark',
            color='YearsSinceLastPromotion',
            color_continuous_scale='Reds'
        )
        fig_roles.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_roles, use_container_width=True)

    # High Gap Employees Table
    st.markdown('<div class="section-header">🔴 High Promotion Gap Employees</div>', unsafe_allow_html=True)
    high_gap = filtered[filtered['Promotion_Gap_Ratio'] >= promo_threshold].sort_values(
        'YearsSinceLastPromotion', ascending=False
    )[['Department', 'JobRole', 'Age', 'YearsAtCompany',
       'YearsSinceLastPromotion', 'Promotion_Gap_Ratio',
       'Risk_Category', 'Suggested_Action']].head(20)
    st.dataframe(high_gap, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════
# TAB 3: RETENTION OPPORTUNITY
# ════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Retention Opportunity Panel</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Risk Score Distribution
        fig_risk = px.histogram(
            filtered, x='Risk_Score',
            color='Risk_Category',
            nbins=30,
            title='Risk Score Distribution',
            template='plotly_dark',
            color_discrete_map={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}
        )
        fig_risk.add_vline(x=30, line_dash='dash', line_color='orange', annotation_text='Medium')
        fig_risk.add_vline(x=60, line_dash='dash', line_color='red', annotation_text='High')
        fig_risk.update_layout(height=380)
        st.plotly_chart(fig_risk, use_container_width=True)

    with col2:
        # Retention Opportunity by Dept
        ret_dept = filtered[filtered['Retention_Opportunity'] == 1].groupby('Department').size().reset_index()
        ret_dept.columns = ['Department', 'Count']
        fig_ret = px.bar(
            ret_dept, x='Department', y='Count',
            color='Department',
            title='Retention Opportunities by Department',
            template='plotly_dark',
            color_discrete_sequence=px.colors.qualitative.Bold,
            text='Count'
        )
        fig_ret.update_traces(textposition='outside')
        fig_ret.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_ret, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Suggested Actions breakdown
        from collections import Counter
        all_actions = []
        for a in filtered['Suggested_Action']:
            all_actions.extend(a.split(' | '))
        action_df = pd.DataFrame(Counter(all_actions).items(), columns=['Action', 'Count'])
        action_df = action_df.sort_values('Count', ascending=True)
        fig_act = px.bar(
            action_df, x='Count', y='Action',
            orientation='h',
            title='Recommended Intervention Actions',
            template='plotly_dark',
            color='Count',
            color_continuous_scale='Oranges'
        )
        fig_act.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_act, use_container_width=True)

    with col4:
        # Risk Score by Attrition
        fig_box = px.box(
            filtered, x='Attrition', y='Risk_Score',
            color='Attrition',
            title='Risk Score: Stayed vs Left',
            template='plotly_dark',
            color_discrete_map={0: '#10b981', 1: '#ef4444'},
            labels={'Attrition': 'Attrition (0=Stayed, 1=Left)'}
        )
        fig_box.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    # Employees Needing Intervention
    st.markdown('<div class="section-header">🎯 Employees Needing Career Intervention</div>', unsafe_allow_html=True)
    intervention = filtered[
        (filtered['Retention_Opportunity'] == 1)
    ].sort_values('Risk_Score', ascending=False)[
        ['Department', 'JobRole', 'Age', 'YearsAtCompany',
         'Risk_Score', 'Risk_Category', 'Promotion_Gap_Score',
         'Suggested_Action']
    ].head(25)
    st.dataframe(intervention, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════
# TAB 4: MANAGERIAL INSIGHTS
# ════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Managerial Insight Dashboard</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Manager Stability vs Risk Score
        fig_mgr = px.scatter(
            filtered, x='Manager_Stability_Indicator', y='Risk_Score',
            color='Risk_Category',
            size='YearsSinceLastPromotion',
            hover_data=['Department', 'JobRole'],
            title='Manager Stability vs Risk Score',
            template='plotly_dark',
            color_discrete_map={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}
        )
        fig_mgr.update_layout(height=400)
        st.plotly_chart(fig_mgr, use_container_width=True)

    with col2:
        # Manager tenure vs career growth
        fig_career = px.scatter(
            filtered, x='YearsWithCurrManager', y='Career_Growth_Rate',
            color='Department',
            hover_data=['JobRole', 'JobLevel', 'PercentSalaryHike'],
            title='Manager Tenure vs Career Growth Rate',
            template='plotly_dark',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_career.update_layout(height=400)
        st.plotly_chart(fig_career, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Avg Manager Stability by Dept
        mgr_dept = filtered.groupby('Department').agg(
            Avg_Manager_Stability=('Manager_Stability_Indicator', 'mean'),
            Avg_Risk_Score=('Risk_Score', 'mean')
        ).round(3).reset_index()
        fig_mgr2 = px.bar(
            mgr_dept, x='Department', y='Avg_Manager_Stability',
            color='Avg_Risk_Score',
            title='Manager Stability by Department',
            template='plotly_dark',
            color_continuous_scale='RdYlGn_r',
            text='Avg_Manager_Stability'
        )
        fig_mgr2.update_traces(textposition='outside')
        fig_mgr2.update_layout(height=380)
        st.plotly_chart(fig_mgr2, use_container_width=True)

    with col4:
        # Team Stagnation Signals
        team_stag = filtered.groupby('Department').agg(
            Avg_Stagnation=('Role_Stagnation_Index', 'mean'),
            High_Risk_Count=('Risk_Category', lambda x: (x == 'High').sum()),
            Attrition_Rate=('Attrition', 'mean')
        ).round(3).reset_index()
        team_stag['Attrition_Rate'] = (team_stag['Attrition_Rate'] * 100).round(1)

        fig_team = px.scatter(
            team_stag, x='Avg_Stagnation', y='Attrition_Rate',
            size='High_Risk_Count', color='Department',
            text='Department',
            title='Team Stagnation vs Attrition Rate',
            template='plotly_dark',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_team.update_traces(textposition='top center')
        fig_team.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_team, use_container_width=True)

    # Manager Impact Table
    st.markdown('<div class="section-header">⚠️ Manager Stability Impact Cases</div>', unsafe_allow_html=True)
    mgr_impact = filtered[filtered['Manager_Impact'] == 1].sort_values(
        'Risk_Score', ascending=False
    )[['Department', 'JobRole', 'YearsAtCompany', 'YearsWithCurrManager',
       'Manager_Stability_Indicator', 'Risk_Score', 'Risk_Category',
       'Suggested_Action']].head(20)
    st.dataframe(mgr_impact, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; font-size: 0.8rem; padding: 10px;'>
    🔒 Palo Alto Networks — Career Intelligence Dashboard &nbsp;|&nbsp;
    Built with Streamlit &nbsp;|&nbsp;
    Unified Mentor Data Science Project
</div>
""", unsafe_allow_html=True)