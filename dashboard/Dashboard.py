import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# ──────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────
st.set_page_config(page_title="Placement Dashboard", page_icon="🎯", layout="wide")

# ──────────────────────────────────────
#  LOAD DATA
# ──────────────────────────────────────
df = pd.read_csv("Job_accept_Final_analysis.csv")

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
tab_kpi, tab_analysis, tab_prediction = st.tabs(["📊 KPIs", "🔍 Analysis", "🎯 Predict Candidate"])


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
        font-size: 28px;
        font-weight: bold;
        margin: 5px 0;
    }
    .kpi-label {
        font-size: 14px;
        opacity: 0.9;
    }
    .metric-card-1 { background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%); }
    .metric-card-2 { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .metric-card-3 { background: linear-gradient(135deg, #4A00E0 0%, #8E2DE2 100%); }
    .metric-card-4 { background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%); }
    .metric-card-5 { background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%); }
    .metric-card-6 { background: linear-gradient(135deg, #f857a6 0%, #ff5858 100%); }
    .metric-card-7 { background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%); }
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

with tab_prediction:
        st.subheader("🎯 New Candidate Placement Prediction")
        st.write(
            "Enter the candidate details below. "
            "The best-performing machine learning model will estimate "
            "whether the candidate is likely to be placed."
        )

        try:
            # Use the same model-building logic as Employee_Placement_Master.py
            model_df = pd.read_csv("Job_accept_Final_analysis.csv")

            drop_columns = [
                "placement_rate",
                "avg_interview_score",
                "placement_prob_score",
                "skills_match_level",
                "experience_category",
                "academic_band",
                "interview_performance",
            ]

            y_model = (model_df["status"] == "Placed").astype(int)
            X_model = model_df.drop(columns=["status"] + drop_columns)

            encoders = {}
            for col in X_model.columns:
                if X_model[col].dtype in ["object", "category"]:
                    le = LabelEncoder()
                    X_model[col] = le.fit_transform(X_model[col])
                    encoders[col] = le

            feature_columns = X_model.columns.tolist()

            X_train, X_test, y_train, y_test = train_test_split(
                X_model,
                y_model,
                test_size=0.2,
                random_state=42,
                stratify=y_model,
            )

            models = {
                "lr": LogisticRegression(max_iter=2000, random_state=42),
                "dt": DecisionTreeClassifier(random_state=42),
                "rf": RandomForestClassifier(random_state=42),
                "xgb": XGBClassifier(random_state=42),
                "knn": KNeighborsClassifier(n_neighbors=5),
                "nb": GaussianNB(),
            }

            best_metric = -1
            best_model = None
            best_name = ""

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)

                if auc > best_metric:
                    best_metric = auc
                    best_model = model
                    best_name = model_name

        except Exception as error:
            st.error(f"Prediction model could not be loaded: {error}")
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

                if probability >= 70:
                    st.success("The model indicates a high probability of placement.")
                elif probability >= 40:
                    st.warning("The model indicates a moderate probability of placement.")
                else:
                    st.error("The model indicates a relatively low probability of placement.")
            except Exception as error:
                st.error(f"Prediction could not be completed: {error}")
