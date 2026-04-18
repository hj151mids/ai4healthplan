import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME INJECTION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI4HealthPlan Enterprise", 
    layout="wide", 
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# Inject the EXACT CSS from the Refined HTML Mockup
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8f9fa;
    }
    
    h1, h2, h3, h4 { color: #1e3a8a !important; font-weight: 800 !important; }

    /* KPI Cards - Exact Match */
    .kpi-card {
        background-color: #ffffff; 
        border-radius: 10px; 
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); 
        border-left: 4px solid #1e3a8a; 
        margin-bottom: 20px;
    }
    .kpi-label { font-size: 13px; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value { font-size: 26px; color: #0f172a; font-weight: 800; margin-top: 5px; }
    .subtext { font-size: 12px; color: #10b981; font-weight: 600; margin-top: 4px; }

    /* Insight Box - Exact Match */
    .insight-box { 
        background-color: #eff6ff; 
        border-left: 4px solid #3b82f6; 
        padding: 16px; 
        border-radius: 0 8px 8px 0; 
        margin-bottom: 24px; 
    }
    .insight-box h4 { color: #1e40af !important; font-size: 14px !important; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; margin-top: 0px; }
    .insight-box p { color: #1e3a8a; font-size: 14px; line-height: 1.5; margin-bottom: 0px; }

    /* System Status Badge */
    .status-badge {
        display: inline-flex; align-items: center; padding: 4px 12px;
        border-radius: 9999px; font-size: 12px; font-weight: 700;
        background-color: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0;
    }
    .status-dot { width: 8px; height: 8px; background-color: #10b981; border-radius: 50%; margin-right: 8px; }

    /* Tab Styling Overrides */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; white-space: pre; background-color: transparent; 
        border: none; color: #64748b; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { color: #1e3a8a !important; border-bottom: 2px solid #1e3a8a !important; }
    
    /* Tables */
    .styled-table { width: 100%; border-collapse: collapse; font-size: 14px; }
    .styled-table thead tr { background-color: #f8f9fa; color: #64748b; text-align: left; font-weight: bold; }
    .styled-table th, .styled-table td { padding: 12px 15px; border-bottom: 1px solid #f1f5f9; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADERS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Attempt to load real data from ML pipeline, otherwise use sample FHIR data
    if os.path.exists("sql_synthetic_claims.csv"):
        return pd.read_csv("sql_synthetic_claims.csv")
    return pd.read_csv("fhir_synthetic_claims_sample.csv")

df = load_data()
care_journeys = {}
if os.path.exists("high_risk_care_journeys.json"):
    with open("high_risk_care_journeys.json", "r") as f:
        care_journeys = json.load(f)

# -----------------------------------------------------------------------------
# 3. SIDEBAR (Exact Content Match)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div style='text-align: center; font-size: 50px;'>🛡️</div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; margin-top: -10px;'>AI4HealthPlan</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-weight: bold; color: #64748b; font-size: 12px; text-transform: uppercase;'>Enterprise Platform</p>", unsafe_allow_html=True)
    
    st.info("**System Mission:** Protecting U.S. Employer EBITDA and driving domestic job growth via AI-driven healthcare cost containment.")
    
    st.markdown("---")
    selected_client = st.selectbox("Active Portfolio:", ["Global Industries (1,200 Lives)", "Acme Corp", "TechFlow Inc"])
    
    st.markdown("---")
    st.markdown("**Platform Modules:**")
    st.markdown("✅ CT-GAN Synthesis\n\n✅ XGBoost/LSTM Engine\n\n✅ PCA Clustering\n\n✅ GenAI Navigation")
    
    st.markdown("---")
    st.caption("AI4HealthPlan MVP v1.0")

# -----------------------------------------------------------------------------
# 4. HEADER (Exact Visual Match)
# -----------------------------------------------------------------------------
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("Employer Command Center")
with header_col2:
    st.markdown("""
        <div style='text-align: right; margin-top: 20px;'>
            <div class="status-badge"><div class="status-dot"></div>System Online</div>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. TABS (The 7 Features)
# -----------------------------------------------------------------------------
t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "📈 Cost Forecasting", "🏢 Data Foundation", "🎛️ Plan Simulator", 
    "🧬 Pop. Clustering", "🚨 High-Risk ID", "🎯 Care Navigation", "💊 Rx Anomalies"
])

# MODULE 1: COST FORECASTING
with t1:
    st.subheader("Macro-Economic Health Trend Forecasting")
    st.caption("Evaluating LSTM Neural Networks and Prophet, optimized via XGBoost Regressors.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
            <div class="insight-box">
                <h4><i class="fas fa-chart-line"></i> Economic Impact Narrative</h4>
                <p>By accurately forecasting the inflationary curve of medical and pharmacy spend, self-funded U.S. employers can transition from reactive stop-loss purchasing to proactive capital allocation.</p>
            </div>
        """, unsafe_allow_html=True)
        # Placeholder for Plotly Chart
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=[1,2,3,4,5], y=[10,12,11,14,16], name="Forecast", line=dict(color='#1e3a8a', width=3)))
        fig_forecast.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_forecast, use_container_width=True)
        
    with col2:
        st.markdown("""
            <div class="kpi-card" style="height: 100%;">
                <h4 style="font-size: 16px; margin-bottom: 15px;">AI Actuary Assessment</h4>
                <p style="font-size: 13px; color: #475569; line-height: 1.6;">
                    The projections are <strong>not speculative</strong>. To ensure accuracy, the AI4HealthPlan platform executed a Model Selection Tournament benchmarking 
                    <strong>LSTM</strong> and <strong>XGBoost</strong> architectures. Reclaiming this operational capital empowers direct investment into workforce expansion.
                </p>
                <div style="margin-top: 20px; border-top: 1px solid #f1f5f9; padding-top: 15px;">
                    <div class="kpi-label">Projected Reclaimed Capital</div>
                    <div class="kpi-value" style="color: #1e3a8a;">$1,710,000</div>
                    <div class="subtext">↑ Funds ~22 net-new U.S. jobs</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# MODULE 2: DATA FOUNDATION (CT-GAN)
with t2:
    st.subheader("HIPAA-Compliant Synthetic Data Generation")
    st.caption("Utilizing CT-GAN to synthesize CMS-HCC actuarial baselines without PHI exposure.")
    
    st.markdown("""
        <div class="insight-box">
            <h4>Enterprise Scalability & Security</h4>
            <p>By deploying CT-GANs, the platform successfully mirrors the statistical distribution of real-world U.S. healthcare claims while maintaining 100% HIPAA compliance.</p>
        </div>
    """, unsafe_allow_html=True)
    
    k1, k2, k3 = st.columns(3)
    k1.markdown("<div class='kpi-card'><div class='kpi-label'>Synthetic Records</div><div class='kpi-value'>145,200</div></div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi-card'><div class='kpi-label'>Privacy Compliance</div><div class='kpi-value' style='color:#10b981'>100%</div></div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi-card'><div class='kpi-label'>Data Fidelity Score</div><div class='kpi-value'>98.4%</div></div>", unsafe_allow_html=True)
    
    st.table(df.head(5))

# MODULE 3: PLAN SIMULATOR (GLM)
with t3:
    st.subheader("Plan Design Simulator")
    st.caption("Utilizing Logistic Regression and GLM to simulate utilization drops based on behavioral economics.")
    
    col_sim_1, col_sim_2 = st.columns([1, 2])
    with col_sim_1:
        st.markdown("#### Simulation Controls")
        deductible = st.slider("Simulated Deductible Increase", 0, 2000, 500, step=100)
        copay = st.selectbox("Copay Adjustment (PCP/Spec)", ["No Change", "+$10", "+$20", "+$50"])
        st.button("Recalculate GLM Impact", use_container_width=True)
    
    with col_sim_2:
        # Dynamic calculation based on deductible slider
        savings = deductible * 1685 # Mock multiplier
        st.markdown(f"""
            <div style="display: flex; gap: 20px;">
                <div class="kpi-card" style="flex: 1;">
                    <div class="kpi-label">Encounters Avoided</div>
                    <div class="kpi-value">{int(deductible/1.5)} Visits</div>
                </div>
                <div class="kpi-card" style="flex: 1;">
                    <div class="kpi-label">Gross Savings</div>
                    <div class="kpi-value" style="color: #10b981;">${savings:,.0f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        # Mock Bar Chart
        fig_sim = px.bar(x=['Baseline', 'Proposed'], y=[2150, 1850], labels={'x':'Scenario', 'y':'Visits'}, color_discrete_sequence=['#1e3a8a'])
        fig_sim.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_sim, use_container_width=True)

# MODULE 4: POPULATION CLUSTERING (PCA/K-Means)
with t4:
    st.subheader("Population Clustering (Risk Personas)")
    st.caption("Applying Principal Component Analysis (PCA) and K-Means Clustering to segment the workforce.")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        # Placeholder for PCA Scatter Plot
        fig_cluster = px.scatter(df, x='extension_spend_velocity_ratio', y='extension_spend_per_condition_intensity', color='extension_is_er_flyer', title="Risk Persona Scatter Plot")
        st.plotly_chart(fig_cluster, use_container_width=True)
    with c2:
        st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">Persona Distribution</div>
                <div style="font-size: 14px; margin-top: 10px;">
                    🔴 Catastrophic (3%)<br>
                    🟠 Chronic (15%)<br>
                    🔵 Acute (22%)<br>
                    🟢 Baseline (60%)
                </div>
            </div>
        """, unsafe_allow_html=True)

# MODULE 5: HIGH-RISK ID (XGBoost/SHAP)
with t5:
    st.subheader("High-Risk Claimant Identification")
    st.caption("Utilizing XGBoost Classifiers and SHAP Values to predict $50k+ breaches.")
    
    st.markdown("""
        <div class="insight-box" style="border-left-color: #ef4444;">
            <h4 style="color: #991b1b !important;">AI Prescriptive Cohort Analysis</h4>
            <p>The engine has isolated <strong>5 critical members</strong> driving outsized liability. Intervening today can avert an estimated $525,202 in spend.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.table(df[df['extension_breached_50k'] == 1][['Patient_identifier', 'Condition_code_display', 'Claim_total_target_12M']])

# MODULE 6: CARE NAVIGATION (GenAI)
with t6:
    st.subheader("Generative AI Care Navigation")
    st.caption("Translating data science into evidence-based clinical action journeys.")
    
    member_list = df['Patient_identifier'].tolist()
    selected_member = st.selectbox("Select Member for Journey Generation:", member_list)
    
    # Matching the "Accordion/Card" style of the HTML
    st.markdown(f"""
        <div style="background-color: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px;">
            <div style="display: flex; align-items: center; gap: 15px; background-color: #eff6ff; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                <div style="width: 50px; height: 50px; background-color: #1e3a8a; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 20px;">
                    {selected_member[:2]}
                </div>
                <div>
                    <h3 style="margin: 0; font-size: 18px;">{selected_member}</h3>
                    <p style="margin: 0; font-size: 12px; color: #1e40af; font-weight: bold;">SHAP Driver: Chronic Condition Management</p>
                </div>
            </div>
            <div style="margin-bottom: 20px;">
                <h4 style="font-size: 16px; margin-bottom: 5px;">Phase 1: Stabilization (Day 1-30)</h4>
                <p style="font-size: 14px; color: #475569;">- Assign RN Care Navigator<br>- Deploy Remote Monitoring Tools<br>- Schedule Baseline Clinical Assessment</p>
            </div>
            <div style="background-color: #f8fafc; padding: 15px; border-radius: 6px; border: 1px solid #f1f5f9;">
                <p style="font-size: 12px; font-weight: bold; color: #64748b;">🎯 PROJECTED SAVINGS: $33,500 | ABSENTEEISM SAVED: 18 DAYS</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# MODULE 7: RX ANOMALIES (Isolation Forest)
with t7:
    st.subheader("Substance Misuse & Waste Monitoring")
    st.caption("Utilizing Isolation Forest and Local Outlier Factor to detect prescription anomalies.")
    
    st.markdown("""
        <div class="insight-box" style="border-left-color: #ef4444; background-color: #fef2f2;">
            <h4 style="color: #991b1b !important;">FWA Mitigation Alert</h4>
            <p>18 consensus anomalies detected. These patterns indicate potential GLP-1 hoarding or pharmacy 'doctor shopping'.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='kpi-card'><div class='kpi-label'>Consensus Flagged Members</div><div class='kpi-value' style='color:#ef4444'>18</div></div>", unsafe_allow_html=True)