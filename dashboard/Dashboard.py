
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import joblib
import plotly.graph_objects as go



# ──────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────
st.set_page_config(page_title="Placement Dashboard", page_icon="🎯", layout="wide")

# ──────────────────────────────────────
#  LOAD DATA
# ──────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Job_accept_Final_analysis.csv"
MODEL_PATH = BASE_DIR / "models" / "best_placement_model.joblib"
COMPARISON_PATH = BASE_DIR / "models" / "model_comparison.csv"
GITHUB_URL = "https://github.com/Vishal2010s/Employee-Job-Acceptance-Prediction-System"

@st.cache_resource
def load_prediction_bundle():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dashboard_data():
    return pd.read_csv(DATA_PATH)

df = load_dashboard_data()

# ──────────────────────────────────────
#  HEADER
# ──────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;color:#2E86C1;'>🎯 Placement Analytics Dashboard</h1>"
    "<p style='text-align:center;color:#7F8C8D;font-size:16px;'>"
    "Recruitment Performance | Interview Outcomes | Candidate Insights</p><hr>",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────
#  SIDEBAR FILTERS
# ──────────────────────────────────────
st.sidebar.title("📌 Navigation")
st.sidebar.markdown("---")

query = st.sidebar.selectbox("🔎 Select Analysis", [
    "1. Academic scores vs placement outcome",
    "2. Skills match vs interview performance",
    "3. Certification impact on job acceptance",
    "4. Acceptance rate by company tier",
    "5. Experience vs placement success",
    "6. Interview score vs placement probability",
    "7. Employability test score analysis",
])

st.sidebar.markdown("---")
st.sidebar.subheader("🎚️ Filters")

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=sorted(df["gender"].dropna().unique()),
    default=sorted(df["gender"].dropna().unique()),
)

exp_filter = st.sidebar.multiselect(
    "Experience Category",
    options=sorted(df["experience_category"].dropna().unique()),
    default=sorted(df["experience_category"].dropna().unique()),
)

# ──────────────────────────────────────
#  APPLY FILTERS
# ──────────────────────────────────────
fdf = df[df["gender"].isin(gender_filter) & df["experience_category"].isin(exp_filter)]

if fdf.empty:
    st.warning("No data matches current filters.")
    st.stop()

# ──────────────────────────────────────
#  COMMON CALCULATIONS
# ──────────────────────────────────────
placed = fdf[fdf["status"] == "Placed"]
total = len(fdf)
n_placed = len(placed)

placement_rate = round(n_placed / total * 100, 2) if total else 0

offered = fdf[fdf["placement_prob_score"] >= 0.6]
n_offered = len(offered)

accepted_offers = offered[offered["status"] == "Placed"]
n_accepted = len(accepted_offers)

acceptance_rate = round(n_accepted / n_offered * 100, 2) if n_offered else 0

dropout_rate = round(
    len(offered[offered["status"] != "Placed"]) / n_offered * 100, 2
) if n_offered else 0

avg_interview = round(fdf["avg_interview_score"].mean(), 2)
avg_skills = round(fdf["skills_match_percentage"].mean(), 2)

high_risk = fdf[fdf["placement_prob_score"] < 0.4]
high_risk_pct = round(len(high_risk) / total * 100, 2) if total else 0


# ──────────────────────────────────────
#  TABS
# ──────────────────────────────────────
tab_kpi, tab_analysis, tab_model, tab_prediction, tab_method = st.tabs([
    "📊 KPIs", "🔍 Analysis", "🧪 Model Comparison",
    "🎯 Predict Candidate", "📘 Project & Methodology"
])


## ══════════════════════════════════════
#  TAB 1: KPIs
# ══════════════════════════════════════

st.markdown("""
<style>
    .kpi-card {
        border-radius: 10px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .kpi-value {
        font-size: 30px;
        font-weight: bold;
        margin: 5px 0;
    }
    .kpi-label {
        font-size: 16px;
        opacity: 0.9;
    }
    .metric-card-1 { background: #1E3A5F; }
    .metric-card-2 { background: #16665D; }
    .metric-card-3 { background: #1E3A5F; }
    .metric-card-4 { background: #813D42; }
    .metric-card-5 { background: #16665D; }
    .metric-card-6 { background: #1E3A5F; }
    .metric-card-7 { background: #813D42; }
    html, body, [data-testid="stAppViewContainer"] {font-size: 16px;}
    .stTabs [aria-selected="true"] {
        background-color: #FF6B35 !important;
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)


with tab_kpi:

    st.subheader("📌 Key Metrics")

    # ── Rows ──
    c1, c2, c3, c4 = st.columns(4)
    

    with c1:
        st.markdown(f'<div class="kpi-card metric-card-1"><div class="kpi-label">👥 Total Candidates</div><div class="kpi-value">{total:,}</div></div>',unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card metric-card-2"><div class="kpi-label">✅ Placement Rate</div><div class="kpi-value">{placement_rate}%</div></div>',unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card metric-card-3"><div class="kpi-label">🤝 Acceptance Rate</div><div class="kpi-value">{acceptance_rate}%</div></div>',unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card metric-card-4"><div class="kpi-label">📉 Dropout Rate</div><div class="kpi-value">{dropout_rate}%</div></div>',unsafe_allow_html=True)

    # ── Row 2: 3 cards ──
    c5, c6, c7, _ = st.columns(4)

    with c5:
        st.markdown(f'<div class="kpi-card metric-card-5"><div class="kpi-label">⭐ Avg Interview Score</div><div class="kpi-value">{avg_interview}</div></div>',unsafe_allow_html=True)
    with c6:
        st.markdown(f'<div class="kpi-card metric-card-6"><div class="kpi-label">🛠️ Avg Skills Match</div><div class="kpi-value">{avg_skills}%</div></div>',unsafe_allow_html=True)

    with c7:
        st.markdown(f'<div class="kpi-card metric-card-7"><div class="kpi-label">⚠️ High-Risk Candidates</div><div class="kpi-value">{high_risk_pct}%</div></div>',unsafe_allow_html=True)
    
    st.markdown("---")

    # ── Charts ──
    st.subheader("📈 Quick KPI Visuals")
    col_a, col_b = st.columns(2)

    with col_a:
        status_counts = fdf["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]

        fig_pie = px.pie(
            status_counts, names="status", values="count",
            hole=0.55, color="status",
            color_discrete_map={"Placed": "#27AE60", "Not Placed": "#E74C3C"},
            title="Placement Status",
        )
        fig_pie.update_layout(height=380)
        fig_pie.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        tier_rate = fdf.groupby("company_tier")["placement_rate"].mean().reset_index()
        tier_rate["placement_pct"] = round(tier_rate["placement_rate"] * 100, 2)

        fig_bar = px.bar(
            tier_rate, x="company_tier", y="placement_pct",
            color="company_tier", text="placement_pct",
            title="Placement Rate by Company Tier",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_bar.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig_bar.update_layout(height=380, yaxis_title="%", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
# ══════════════════════════════════════
#  DATA PREVIEW (below tabs, always visible)
# ══════════════════════════════════════
    st.markdown("---")
    st.subheader("📋 Filtered Data Preview")
    st.caption(f"Showing first 200 rows of {len(fdf):,} filtered records")
    st.dataframe(fdf.head(200), use_container_width=True, height=520)

# ══════════════════════════════════════
#  TAB 2: ANALYSIS
# ══════════════════════════════════════
with tab_analysis:
    st.subheader(f"🔍 {query}")

    # ── Query 1 ──────────────────────
    if "Academic" in query:

        st.markdown("Do academic scores affect placement?")

        score_cols = ["ssc_percentage", "hsc_percentage", "degree_percentage"]
        melted = fdf[score_cols + ["status"]].melt(id_vars="status", var_name="Exam", value_name="Score")

        fig = px.box(melted,x="Exam", y="Score", color="status",
            color_discrete_map={"Placed": "#27AE60", "Not Placed": "#E74C3C"},
            title="Academic Scores by Placement Outcome",
        )
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Summary by Academic Band")
        band = fdf.groupby("academic_band")["status"].value_counts().unstack(fill_value=0).reset_index()
        band["total"] = band.get("Placed", 0) + band.get("Not Placed", 0)
        band["placement_rate_%"] = round(band.get("Placed", 0) / band["total"] * 100, 2)
        st.dataframe(band, use_container_width=True, hide_index=True)

        st.markdown("##### Average Scores by Status")
        avg_scores = fdf.groupby("status")[score_cols].mean().round(2).reset_index()
        st.dataframe(avg_scores, use_container_width=True, hide_index=True)

    # ── Query 2 ──────────────────────
    elif "Skills match" in query:

        st.markdown("Does skills match correlate with interview score?")

        skills_match_level = pd.cut(fdf['skills_match_percentage'],bins=5,labels=['Very Low (<55%)', 'Low (56%-65%)', 'Medium (66%-75%)', 'High (76%-90%)', 'Very High (>90%)'])
        placement = fdf.groupby(skills_match_level)['placement_rate'].mean() * 100
        plot_data1 = placement.reset_index()
        plot_data1.rename(columns={'skills_match_percentage':'skills_match_level'}, inplace=True)
        
        fig = px.bar(
            plot_data1,
            x='skills_match_level',
            y='placement_rate',
            title='Placement Rate by Skills Match Level',
            text='placement_rate',
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(title_x=0.5, height=700, width=700)
        st.plotly_chart(fig, use_container_width=True)        

        fdf = fdf.copy()
        fdf['skills_match_level'] = skills_match_level

        perf = fdf.groupby('skills_match_level').agg(
            count=('status', 'count'),
            avg_skill_percentage=('skills_match_percentage', 'mean'),
            avg_placement_percentage=('placement_rate', 'mean'),
        ).round(2).reset_index()

        # Convert placement to percentage (same as the chart)
        perf['avg_placement_percentage'] = (perf['avg_placement_percentage'] * 100)

        st.dataframe(perf, use_container_width=True, hide_index=True)

    # ── Query 3 ──────────────────────
    elif "Certification" in query:

        st.markdown("Do certifications help with job acceptance?")

        fdf_cert = fdf.copy()
        fdf_cert["has_cert"] = (fdf_cert["certifications_count"] > 0).map({True: "Yes", False: "No"})

        fig = px.histogram(
            fdf_cert, x="certifications_count", color="status",
            barmode="group",
            color_discrete_map={"Placed": "#27AE60", "Not Placed": "#E74C3C"},
            title="Certifications vs Placement",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        cert_agg = fdf_cert.groupby("has_cert").agg(
            candidates=("status", "count"),
            placed=("status", lambda s: (s == "Placed").sum()),
            avg_interview_score=("avg_interview_score", "mean"),
        ).round(2).reset_index()
        cert_agg["job_acceptance_%"] = round(cert_agg["placed"] / cert_agg["candidates"] * 100, 2)
        st.dataframe(cert_agg, use_container_width=True, hide_index=True)

    # ── Query 4 ──────────────────────
    elif "Acceptance" in query:

        st.markdown("How does company tier affect acceptance rates?")

        tier = fdf.groupby("company_tier").agg(
            candidates=("status", "count"),
            placed=("status", lambda s: (s == "Placed").sum()),
            avg_ctc=("expected_ctc_lpa", "mean"),
        ).round(2).reset_index()
        tier["acceptance_%"] = round(tier["placed"] / tier["candidates"] * 100, 2)
        tier = tier.sort_values("acceptance_%", ascending=False)

        fig1 = px.bar(
            tier, x="company_tier", y="acceptance_%", color="company_tier",
            text="acceptance_%", title="Acceptance Rate by Company Tier",
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig1.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig1.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

        st.dataframe(tier, use_container_width=True, hide_index=True)

    # ── Query 5 ──────────────────────
    elif "Experience" in query:

        st.markdown("How does experience affect placement?")

        exp = fdf.groupby("experience_category").agg(
            candidates=("status", "count"),
            placed=("status", lambda s: (s == "Placed").sum()),
            avg_interview=("avg_interview_score", "mean"),
            avg_ctc=("expected_ctc_lpa", "mean")
        ).round(2).reset_index()
        exp["placement_%"] = round(exp["placed"] / exp["candidates"] * 100, 2)
        exp = exp.sort_values("placement_%", ascending=False)


        
        fig1 = px.bar(
            exp, x="experience_category", y="placement_%", color="experience_category",
            text="placement_%", title="Placement by Experience",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig1.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig1.update_layout(height=500, width=850,  showlegend=False)
        st.plotly_chart(fig1, use_container_width=False)

        
        st.dataframe(exp, use_container_width=True, hide_index=True)

    # ── Query 6 ──────────────────────
    elif "Interview score" in query:

        st.markdown("Does interview score predict placement?")

        avg_interview_score = (fdf['technical_score']+fdf['aptitude_score']+fdf['communication_score']) / 3
        score_levels = pd.cut(avg_interview_score,bins=5,labels=['Very Low (Score<55)', 'Low (Score 56-65)', 'Medium (Score 66-75)', 'High (Score 76-85)', 'Very High (Score>85)'])
        placement_rate = fdf.groupby(score_levels).agg(
            candidates=("status", "count"),
            placed=("status", lambda s: (s == "Placed").sum()),
            ).reset_index()
        placement_rate["placement_%"] = round(placement_rate["placed"] / placement_rate["candidates"] * 100, 2)
        
        plot_data = placement_rate.reset_index()
        plot_data.rename(columns={'index':'score_levels'},inplace=True)
        fig = px.bar(plot_data,x='score_levels',y='placement_%',title='Placement Rate by Interview Score Level',text='placement_%')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout( height=500)
        #Visualization
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(placement_rate, use_container_width=True, hide_index=True)
        
        
    # ── Query 7 ──────────────────────
    elif "Employability" in query:

        st.markdown("How do test scores relate to placement?")

        test_cols = ["technical_score", "aptitude_score", "communication_score"]

        fdf_emp = fdf.copy()
        fdf_emp["employability_avg"] = fdf_emp[test_cols].mean(axis=1).round(2)
        fdf_emp["Employment_band"] = pd.cut(
            fdf_emp["employability_avg"],
            bins=5,
            labels=["Poor (avg_test_Score : 40-55)", "Below Avg (avg_test_Score : 55-65)", "Average (avg_test_Score :66-75)", "Good (avg_test_Score : 75-85)", "Excellent (avg_test_Score > 85)"],
        )

        c1, c2 = st.columns(2)

        with c1:
            avg_by_status = fdf_emp.groupby("status")[test_cols].mean().round(2).reset_index()
            melted_avg = avg_by_status.melt(id_vars="status", var_name="Test", value_name="Score")

            fig1 = px.bar(
                melted_avg, x="Test", y="Score", color="status", barmode="group",
                text="Score", title="Avg Test Scores by Outcome",
                color_discrete_map={"Placed": "#27AE60", "Not Placed": "#E74C3C"},
            )
            fig1.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig1.update_layout(height=420)
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            band_agg = fdf_emp.groupby("Employment_band", observed=True).agg(
                candidates=("status", "count"),
                placed=("status", lambda s: (s == "Placed").sum()),
            ).reset_index()
            band_agg["placement_%"] = round(band_agg["placed"] / band_agg["candidates"] * 100, 2)

            fig2 = px.bar(
                band_agg, x="Employment_band", y="placement_%", color="Employment_band",
                text="placement_%", title="Placement by Employability Band",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig2.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig2.update_layout(height=420, xaxis=dict(showticklabels=False), xaxis_title="Employment_band_based on avg score", showlegend=True)
            st.plotly_chart(fig2, use_container_width=True)


        st.dataframe(band_agg, use_container_width=True, hide_index=True)

with tab_model:
    st.subheader("🧪 Six-Model Performance Comparison")
    st.caption("These metrics come from the master training run; the dashboard never retrains models. Scores reflect the evaluation scheme used by that script.")
    if COMPARISON_PATH.exists():
        measured = pd.read_csv(COMPARISON_PATH)
        if "j" in measured.columns:
            measured = measured.rename(columns={"j": "Model"})
        if "ROC-AUC" in measured.columns:
            measured = measured.sort_values("ROC-AUC", ascending=False)
        st.dataframe(measured, hide_index=True, use_container_width=True)
        if {"Model", "ROC-AUC"}.issubset(measured.columns):
            chart = px.bar(measured, x="Model", y="ROC-AUC", text_auto=".3f",
                           title="Measured ROC-AUC by algorithm", color="ROC-AUC",
                           color_continuous_scale="Blues")
            chart.update_layout(coloraxis_showscale=False, yaxis_title="ROC-AUC (0–1)")
            st.plotly_chart(chart, use_container_width=True)
    else:
        st.warning("Comparison metrics are not yet deployed. Run the updated master script, then commit models/model_comparison.csv.")
    st.info("The original training script ranks models using test-set ROC-AUC. For a stronger unbiased assessment, select via cross-validation or a validation set, then evaluate once on an untouched test set.")
    st.markdown("[View training code on GitHub](" + GITHUB_URL + "/tree/main/src)")

with tab_prediction:
        st.subheader("🎯 New Candidate Placement Prediction")
        st.write(
            "Enter the candidate details below. "
            "The best-performing machine learning model will estimate "
            "whether the candidate is likely to be placed."
        )

        try:
            model_bundle = load_prediction_bundle()
            best_model = model_bundle["model"]
            best_name = model_bundle["model_name"]
            best_metric = float(model_bundle["roc_auc"])
            encoders = model_bundle["encoders"]
            feature_columns = model_bundle["feature_columns"]
        except Exception as error:
            st.error(f"Saved model could not be loaded: {error}")
            st.info("Train the master script once and commit models/best_placement_model.joblib.")
            st.stop()
        
        st.info(
            f"Current Prediction Model: {best_name} | "
            f"ROC-AUC: {best_metric:.4f}"
        )

        with st.form("candidate_prediction_form"):
            st.markdown("### 👤 Personal & Academic Details")
            c1, c2, c3 = st.columns(3)

            with c1:
                age_years = st.number_input("Age", min_value=18, max_value=60, value=24)
                ssc_percentage = st.number_input(
                    "SSC Percentage", min_value=0.0, max_value=100.0, value=82.0
                )

            with c2:
                gender = st.selectbox("Gender", list(encoders["gender"].classes_))
                hsc_percentage = st.number_input(
                    "HSC Percentage", min_value=0.0, max_value=100.0, value=79.0
                )

            with c3:
                degree_percentage = st.number_input(
                    "Degree Percentage", min_value=0.0, max_value=100.0, value=81.5
                )
                degree_specialization = st.selectbox(
                    "Degree Specialization",
                    list(encoders["degree_specialization"].classes_),
                )

            st.markdown("---")
            st.markdown("### 📝 Assessment & Skills")
            c1, c2, c3 = st.columns(3)

            with c1:
                technical_score = st.number_input(
                    "Technical Score", min_value=0, max_value=100, value=85
                )
                skills_match_percentage = st.number_input(
                    "Skills Match Percentage",
                    min_value=0.0,
                    max_value=100.0,
                    value=88.0,
                )

            with c2:
                aptitude_score = st.number_input(
                    "Aptitude Score", min_value=0, max_value=100, value=78
                )
                certifications_count = st.number_input(
                    "Certifications Count", min_value=0, value=3
                )

            with c3:
                communication_score = st.number_input(
                    "Communication Score", min_value=0, max_value=100, value=82
                )
                internship_experience = st.selectbox(
                    "Internship Experience",
                    list(encoders["internship_experience"].classes_),
                )

            st.markdown("---")
            st.markdown("### 💼 Experience")
            c1, c2, c3 = st.columns(3)

            with c1:
                years_of_experience = st.number_input(
                    "Years of Experience", min_value=0.0, value=1.5
                )
                previous_ctc_lpa = st.number_input(
                    "Previous CTC (LPA)", min_value=0.0, value=3.5
                )

            with c2:
                career_switch_willingness = st.selectbox(
                    "Career Switch Willingness",
                    list(encoders["career_switch_willingness"].classes_),
                )
                expected_ctc_lpa = st.number_input(
                    "Expected CTC (LPA)", min_value=0.0, value=5.5
                )

            with c3:
                relevant_experience = st.selectbox(
                    "Relevant Experience",
                    list(encoders["relevant_experience"].classes_),
                )
                employment_gap_months = st.number_input(
                    "Employment Gap (Months)", min_value=0, value=0
                )

            st.markdown("---")
            st.markdown("### 🏢 Job & Company Details")
            c1, c2, c3 = st.columns(3)

            with c1:
                company_tier = st.selectbox(
                    "Company Tier", list(encoders["company_tier"].classes_)
                )
                bond_requirement = st.selectbox(
                    "Bond Requirement", list(encoders["bond_requirement"].classes_)
                )
                layoff_history = st.selectbox(
                    "Layoff History", list(encoders["layoff_history"].classes_)
                )

            with c2:
                job_role_match = st.selectbox(
                    "Job Role Match", list(encoders["job_role_match"].classes_)
                )
                notice_period_days = st.number_input(
                    "Notice Period (Days)", min_value=0, value=30
                )
                relocation_willingness = st.selectbox(
                    "Relocation Willingness",
                    list(encoders["relocation_willingness"].classes_),
                )

            with c3:
                competition_level = st.selectbox(
                    "Competition Level", list(encoders["competition_level"].classes_)
                )

            st.markdown("---")
            predict_button = st.form_submit_button(
                "🎯 Predict Placement", use_container_width=True
            )

        if predict_button:
            candidate_data = {
                "age_years": age_years,
                "gender": gender,
                "ssc_percentage": ssc_percentage,
                "hsc_percentage": hsc_percentage,
                "degree_percentage": degree_percentage,
                "degree_specialization": degree_specialization,
                "technical_score": technical_score,
                "aptitude_score": aptitude_score,
                "communication_score": communication_score,
                "skills_match_percentage": skills_match_percentage,
                "certifications_count": certifications_count,
                "internship_experience": internship_experience,
                "years_of_experience": years_of_experience,
                "career_switch_willingness": career_switch_willingness,
                "relevant_experience": relevant_experience,
                "previous_ctc_lpa": previous_ctc_lpa,
                "expected_ctc_lpa": expected_ctc_lpa,
                "company_tier": company_tier,
                "job_role_match": job_role_match,
                "competition_level": competition_level,
                "bond_requirement": bond_requirement,
                "notice_period_days": notice_period_days,
                "layoff_history": layoff_history,
                "employment_gap_months": employment_gap_months,
                "relocation_willingness": relocation_willingness,
            }

            try:
                candidate_df = pd.DataFrame([candidate_data])

                # Encode categorical values using the same encoders fitted on training data
                for col, encoder in encoders.items():
                    if col in candidate_df.columns:
                        candidate_df[col] = encoder.transform(
                            candidate_df[col].astype(str).str.title().str.strip()
                        )

                # Keep exactly the same feature order used during model training
                candidate_df = candidate_df[feature_columns]

                prediction = int(best_model.predict(candidate_df)[0])
                probability = float(best_model.predict_proba(candidate_df)[0][1] * 100)

                st.markdown("## 📊 Prediction Result")
                r1, r2 = st.columns(2)

                with r1:
                    if prediction == 1:
                        st.success("✅ Predicted Outcome: PLACED")
                    else:
                        st.error("❌ Predicted Outcome: NOT PLACED")

                with r2:
                    st.metric("Placement Probability", f"{probability:.2f}%")

                st.progress(min(max(float(probability) / 100, 0.0), 1.0))

                st.markdown("### 🔎 What influenced this prediction?")
                st.caption("SHAP describes contributions to this model's output, not causal factors or reasons to make a hiring decision. Categorical features use numeric codes fitted during training.")
                if best_name.lower() == "xgb":
                    try:
                        import shap
                        explainer = shap.TreeExplainer(best_model)
                        explanation = explainer(candidate_df)
                        values = np.asarray(explanation.values)
                        if values.ndim == 3:
                            values = values[:, :, 1]
                        contributions = pd.DataFrame({
                            "Feature": feature_columns,
                            "SHAP contribution": values[0]
                        })
                        contributions["Absolute impact"] = contributions["SHAP contribution"].abs()
                        contributors = contributions.nlargest(10, "Absolute impact").sort_values("SHAP contribution")
                        chart = px.bar(
                            contributors, x="SHAP contribution", y="Feature",
                            orientation="h", color="SHAP contribution",
                            color_continuous_scale="RdBu", color_continuous_midpoint=0,
                            title="Top 10 feature contributions (model output scale)"
                        )
                        chart.update_layout(height=460, coloraxis_showscale=False)
                        st.plotly_chart(chart, use_container_width=True)
                        st.caption("Positive values move the model output toward the Placed class; negative values move it away. Values are not percentage-point changes in probability.")
                    except ImportError:
                        st.info("Install `shap` in requirements.txt to enable explanations.")
                    except Exception as explain_error:
                        st.warning(f"SHAP explanation unavailable: {explain_error}")
                else:
                    st.info("Individual SHAP explanations are currently supported for the saved XGBoost model.")


                if probability >= 70:
                    st.success("The model indicates a high probability of placement.")
                elif probability >= 40:
                    st.warning("The model indicates a moderate probability of placement.")
                else:
                    st.error("The model indicates a relatively low probability of placement.")
            except Exception as error:
                st.error(f"Prediction could not be completed: {error}")


with tab_method:
    st.subheader("📘 Project Overview & Methodology")
    st.write("Interactive exploration of candidate placement outcomes and a demonstration machine-learning classifier for estimating placement likelihood from candidate and job attributes.")
    st.markdown("""**Workflow**: data preparation and categorical normalization → descriptive analysis and feature engineering → model comparison (Logistic Regression, Decision Tree, Random Forest, XGBoost, KNN and Gaussian Naive Bayes) → save the selected model with its fitted encoders → fast candidate inference.""")
    st.markdown("**Dataset:** See the repository's `data/` folder and README for origin, definitions, scope, and any licensing constraints.")
    st.markdown("**Model selection:** The supplied master script compares ROC-AUC scores on a held-out split. Because that split is also used to choose the winner, its score should not be presented as an untouched final-test estimate.")
    st.markdown("**Interpretation:** The output is an estimated model probability, not a guarantee of placement, offer acceptance, or future job performance. SHAP contributions show associations learned by the model rather than causes.")
    st.markdown("**Limitations:** Potential dataset bias, unverified generalization, artificial ordinal coding of categories, and reliance on the original data distribution. Avoid using protected characteristics or this tool for real employment decisions without appropriate fairness, validity, legal and human-oversight review.")
    st.link_button("📁 GitHub source, README and dataset", GITHUB_URL)
    st.link_button("📄 Training methodology (master script)", GITHUB_URL + "/tree/main/src")
