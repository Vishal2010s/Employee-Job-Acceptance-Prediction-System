import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
placement_rate = round(n_placed / total * 100, 2)

offered = fdf[fdf["placement_prob_score"] >= 0.6]
n_offered = len(offered)
acceptance_rate = round(n_placed / n_offered * 100, 2) if n_offered else 0
dropout_rate = round(len(offered[offered["status"] != "Placed"]) / n_offered * 100, 2) if n_offered else 0

avg_interview = round(fdf["avg_interview_score"].mean(), 2)
avg_skills = round(fdf["skills_match_percentage"].mean(), 2)

high_risk = fdf[fdf["placement_prob_score"] < 0.4]
high_risk_pct = round(len(high_risk) / total * 100, 2)


# ──────────────────────────────────────
#  TABS (only 2 now)
# ──────────────────────────────────────
tab_kpi, tab_analysis = st.tabs(["📊 KPIs", "🔍 Analysis"])


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

        skills_match_level = pd.cut(fdf['skills_match_percentage'],bins=5,labels=['Very Low (<45%)', 'Low (46%-60%)', 'Medium (61%-75%)', 'High (76%-90%)', 'Very High(>90%)'])
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
        score_levels = pd.cut(avg_interview_score,bins=5,labels=['Very Low (Score<40)', 'Low (Score 40-55)', 'Medium (Score 55-65)', 'High (Score 65-75)', 'Very High (Score>75)'])
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


