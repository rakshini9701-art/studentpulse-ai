"""
StudentPulse AI
AI-Based Student Attendance Analytics, Academic Performance Prediction
and Early Intervention Recommendation System

Run:  streamlit run app.py
"""
import base64
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import auth
import core

ASSETS = Path(__file__).parent / "assets"
LOGO_FULL = ASSETS / "logo.png"
LOGO_SMALL = ASSETS / "logo_small.png"

# ------------------------------------------------------------------ project details (edit these)
COLLEGE = "Mount Zion College of Engineering and Technology"
DEPARTMENT = "Department of Artificial Intelligence and Data Science"
TEAM = ["Rakshini S", "Pavithra B"]
BATCH = "III AI&DS - Mini Project - Batch 6 (2026-2027)"

NAVY, TEAL = "#1f4e8c", "#14937d"
BAND_COLORS = {"Low": "#2e8b57", "Medium": "#e0a100", "High": "#c0392b"}
SHOW_DEMO_LOGINS = True   # set to False before a real deployment to hide the demo credentials hint

# which pages each role can open
PAGES_BY_ROLE = {
    "staff": ["Dashboard", "Student Explorer", "Early Warning", "Dataset", "My Account", "About"],
    "management": ["Dashboard", "Management Summary", "Student Explorer", "Early Warning", "Dataset",
                   "Model Report", "User Management", "My Account", "About"],
}


def _logo_b64():
    try:
        return base64.b64encode(LOGO_SMALL.read_bytes()).decode()
    except OSError:
        return ""


def _page_icon():
    try:
        return Image.open(LOGO_SMALL)
    except OSError:
        return "🎓"


st.set_page_config(page_title="StudentPulse AI", page_icon=_page_icon(), layout="wide")

st.markdown(f"""
<style>
#MainMenu, footer {{visibility: hidden;}}
.block-container {{padding-top: 1.2rem; max-width: 1250px;}}
.hero {{background: linear-gradient(120deg, {NAVY} 0%, {TEAL} 100%); color: white;
        padding: 22px 28px; border-radius: 14px; margin-bottom: 18px;}}
.hero h1 {{margin: 0; font-size: 1.9rem; color: white;}}
.hero p {{margin: 4px 0 0 0; opacity: .92; font-size: .98rem;}}
.kpi {{background: white; border-radius: 12px; padding: 16px 18px; border: 1px solid #e3e8ef;
       border-left: 5px solid {NAVY}; box-shadow: 0 1px 3px rgba(20,40,80,.06);}}
.kpi .v {{font-size: 1.9rem; font-weight: 700; color: #1b2a3a; line-height: 1.1;}}
.kpi .l {{font-size: .82rem; color: #5b6b7c; text-transform: uppercase; letter-spacing: .04em;}}
.card {{background: white; border-radius: 12px; padding: 14px 18px; border: 1px solid #e3e8ef; margin-bottom: 10px;}}
.pill {{display: inline-block; padding: 2px 12px; border-radius: 999px; color: white; font-weight: 600; font-size: .85rem;}}
.action {{background: white; border: 1px solid #e3e8ef; border-left: 4px solid {TEAL};
          border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;}}
.hero-row {{display: flex; align-items: center; gap: 20px;}}
.hero-logo {{background: white; border-radius: 14px; padding: 6px; height: 84px; width: 84px;
             display: flex; align-items: center; justify-content: center; flex-shrink: 0;}}
.hero-logo img {{max-height: 72px; max-width: 72px;}}
.role-badge {{display: inline-block; padding: 2px 12px; border-radius: 999px; font-size: .78rem; font-weight: 700;
              color: white; letter-spacing: .04em; text-transform: uppercase;}}
.login-title {{text-align: center; margin: 6px 0 0 0; color: {NAVY}; font-size: 1.9rem; font-weight: 800;}}
.login-sub {{text-align: center; color: #5b6b7c; margin: 2px 0 14px 0; font-size: .95rem;}}
.login-college {{text-align: center; color: #1b2a3a; font-weight: 600; margin-top: 4px;}}
.motto {{text-align: center; color: {TEAL}; font-style: italic; margin-bottom: 8px;}}
</style>
""", unsafe_allow_html=True)


def hero(title, subtitle):
    logo = _logo_b64()
    logo_html = f'<div class="hero-logo"><img src="data:image/png;base64,{logo}"></div>' if logo else ""
    st.markdown(f'<div class="hero"><div class="hero-row">{logo_html}<div><h1>{title}</h1><p>{subtitle}</p></div>'
                f'</div></div>', unsafe_allow_html=True)


def kpi(col, value, label, color=NAVY):
    col.markdown(f'<div class="kpi" style="border-left-color:{color}"><div class="v">{value}</div>'
                 f'<div class="l">{label}</div></div>', unsafe_allow_html=True)


def style_fig(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10),
                      paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#1b2a3a"),
                      title_font_size=16)
    return fig


# ------------------------------------------------------------------ data + model
@st.cache_resource
def get_bundle():
    return core.load_model()


@st.cache_data
def get_default_data():
    return core.ensure_data()


bundle = get_bundle()


# ------------------------------------------------------------------ login gate
def login_page():
    try:
        full = base64.b64encode(LOGO_FULL.read_bytes()).decode()
    except OSError:
        full = ""
    _, mid, _ = st.columns([1, 1.25, 1])
    with mid:
        if full:
            st.markdown(f'<div style="text-align:center;margin-top:10px"><img src="data:image/png;base64,{full}" '
                        f'width="190"></div>', unsafe_allow_html=True)
        st.markdown('<p class="login-title">StudentPulse AI</p>'
                    '<p class="login-sub">Attendance analytics · Performance prediction · Early intervention</p>'
                    f'<p class="login-college">{COLLEGE}</p>'
                    '<p class="motto">To Make Man Whole</p>', unsafe_allow_html=True)

        wait = auth.lock_seconds_left(st.session_state)
        with st.form("login_form"):
            role_label = st.radio("Sign in as", ["Staff", "Management"], horizontal=True)
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)

        if submitted:
            role = role_label.lower()
            if wait:
                st.error(f"Too many failed attempts. Please wait {wait} seconds and try again.")
            elif not username or not password:
                st.warning("Please enter your username and password.")
            else:
                user = auth.verify_login(username, password, role)
                if user:
                    auth.register_success(st.session_state)
                    auth.log_event(user["username"], user["role"], "login")
                    st.session_state["user"] = user
                    st.rerun()
                else:
                    auth.register_failure(st.session_state)
                    auth.log_event(username.strip().lower()[:30], role, "failed login")
                    st.error("Incorrect username, password or role.")
        elif wait:
            st.error(f"Too many failed attempts. Please wait {wait} seconds and try again.")

        if SHOW_DEMO_LOGINS:
            with st.expander("Demo accounts (for the project review)"):
                st.markdown("| Role | Username | Password |\n|---|---|---|\n"
                            "| Staff | `staff` | `Staff@123` |\n"
                            "| Management | `hod` | `Hod@123` |\n"
                            "| Management | `principal` | `Principal@123` |")
                st.caption("Management can change passwords and add staff accounts under User Management.")
        st.caption("Authorised use only. Sign-ins are recorded.")


user = st.session_state.get("user")
if not user:
    login_page()
    st.stop()

is_mgmt = user["role"] == "management"
PAGES = PAGES_BY_ROLE[user["role"]]

# ------------------------------------------------------------------ sidebar
if LOGO_SMALL.exists():
    st.sidebar.image(str(LOGO_SMALL), width=90)
st.sidebar.markdown("## StudentPulse AI")
st.sidebar.caption(f"{COLLEGE}\n\n{DEPARTMENT}")
badge_color = TEAL if is_mgmt else NAVY
st.sidebar.markdown(f'**{user["name"]}**<br><span class="role-badge" style="background:{badge_color}">'
                    f'{auth.ROLE_LABEL[user["role"]]}</span>', unsafe_allow_html=True)
if st.sidebar.button("Sign out", use_container_width=True):
    auth.log_event(user["username"], user["role"], "logout")
    st.session_state.pop("user", None)
    st.rerun()
st.sidebar.divider()
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.divider()
if is_mgmt:
    source = st.sidebar.radio("Data source", ["Sample dataset (400 students)", "Upload my own CSV"])
else:
    source = "Sample dataset (400 students)"

if source.startswith("Upload"):
    up = st.sidebar.file_uploader("Upload CSV", type="csv",
                                  help="Needs columns: student_id, name, " + ", ".join(core.FEATURES))
    if up is None:
        hero("StudentPulse AI", "Upload a CSV in the sidebar, or switch back to the sample dataset.")
        st.stop()
    raw = pd.read_csv(up)
    missing = [c for c in ["student_id", "name"] + core.FEATURES if c not in raw.columns]
    if missing:
        st.error(f"Your CSV is missing these columns: {missing}")
        st.stop()
else:
    raw = get_default_data()

data = core.score_all(raw, bundle)
st.sidebar.divider()
st.sidebar.caption(f"Model in use: **{bundle['name']}**")
st.sidebar.caption("Predictions support mentors; they are not final decisions.")


# ================================================================== DASHBOARD
if page == "Dashboard":
    hero("Attendance & Risk Dashboard", "Class-wide view of attendance, performance and predicted risk")
    c = st.columns(5)
    kpi(c[0], len(data), "Students")
    kpi(c[1], int((data.risk_band == "High").sum()), "High risk", BAND_COLORS["High"])
    kpi(c[2], int((data.risk_band == "Medium").sum()), "Medium risk", BAND_COLORS["Medium"])
    kpi(c[3], int((data.attendance_pct < 75).sum()), "Below 75% attendance", "#8e44ad")
    kpi(c[4], f"{data.attendance_pct.mean():.1f}%", "Average attendance", TEAL)
    st.write("")

    l, r = st.columns(2)
    with l:
        counts = data.risk_band.value_counts().reindex(["Low", "Medium", "High"]).fillna(0).reset_index()
        counts.columns = ["Risk band", "Students"]
        fig = px.pie(counts, names="Risk band", values="Students", hole=0.55, color="Risk band",
                     color_discrete_map=BAND_COLORS, title="Students by predicted risk band")
        st.plotly_chart(style_fig(fig))
    with r:
        fig = px.histogram(data, x="attendance_pct", nbins=25, title="Attendance distribution",
                           labels={"attendance_pct": "Attendance %"}, color_discrete_sequence=[NAVY])
        fig.add_vline(x=75, line_dash="dash", line_color="#c0392b", annotation_text="75% limit")
        st.plotly_chart(style_fig(fig))

    l, r = st.columns(2)
    with l:
        bins = pd.cut(data.attendance_pct, [0, 60, 70, 75, 85, 100],
                      labels=["<60", "60-70", "70-75", "75-85", "85-100"])
        grp = data.groupby(bins, observed=True).agg(avg_marks=("internal2", "mean"),
                                                   risk=("risk_prob", "mean")).reset_index()
        grp.columns = ["Attendance range (%)", "Avg Internal 2", "Avg risk"]
        fig = px.bar(grp, x="Attendance range (%)", y="Avg Internal 2", text_auto=".0f",
                     title="Higher attendance, higher marks", color_discrete_sequence=[TEAL])
        st.plotly_chart(style_fig(fig))
    with r:
        fig = px.scatter(data, x="attendance_pct", y="internal2", color="risk_band",
                         color_discrete_map=BAND_COLORS, hover_data=["student_id", "name"],
                         title="Attendance vs Internal 2 (colour = predicted risk)",
                         labels={"attendance_pct": "Attendance %", "internal2": "Internal 2 marks"})
        st.plotly_chart(style_fig(fig))

    corr = data[core.FEATURES + ["risk_prob"]].corr().round(2)
    names = [core.FEATURE_LABELS.get(c, "Predicted risk") for c in corr.columns]
    fig = px.imshow(corr.values, x=names, y=names, text_auto=True, zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r", title="How the factors relate to each other and to risk")
    st.plotly_chart(style_fig(fig, 480))


# ================================================================== MANAGEMENT SUMMARY
elif page == "Management Summary" and is_mgmt:
    hero("Management Summary", "Institution-level picture for HOD and Principal: risk, workload and priorities")
    n = len(data)
    high = int((data.risk_band == "High").sum())
    med = int((data.risk_band == "Medium").sum())
    low_att = int((data.attendance_pct < 75).sum())
    c = st.columns(5)
    kpi(c[0], n, "Students analysed")
    kpi(c[1], f"{high / n:.0%}", "High-risk share", BAND_COLORS["High"])
    kpi(c[2], f"{(high + med) / n:.0%}", "Need mentor attention", BAND_COLORS["Medium"])
    kpi(c[3], f"{low_att / n:.0%}", "Below 75% attendance", "#8e44ad")
    kpi(c[4], f"{data.attendance_pct.mean():.1f}%", "Average attendance", TEAL)
    st.write("")

    l, r = st.columns(2)
    with l:
        bins = pd.cut(data.attendance_pct, [0, 60, 70, 75, 85, 100],
                      labels=["<60", "60-70", "70-75", "75-85", "85-100"])
        mix = data.groupby([bins, "risk_band"], observed=True).size().reset_index(name="Students")
        mix.columns = ["Attendance range (%)", "Risk band", "Students"]
        fig = px.bar(mix, x="Attendance range (%)", y="Students", color="Risk band",
                     color_discrete_map=BAND_COLORS, title="Risk mix by attendance range",
                     category_orders={"Risk band": ["Low", "Medium", "High"]})
        st.plotly_chart(style_fig(fig, 380))
    with r:
        need = data[data.risk_band.isin(["High", "Medium"])]
        if len(need):
            top = need.apply(lambda x: core.recommend(x, x.risk_prob)[0].split(":")[0], axis=1)
            work = top.value_counts().reset_index()
            work.columns = ["Action needed", "Students"]
            fig = px.bar(work.sort_values("Students"), x="Students", y="Action needed", orientation="h",
                         title="Mentor workload: top action for at-risk students", color_discrete_sequence=[TEAL])
            st.plotly_chart(style_fig(fig, 380))
        else:
            st.success("No students currently need mentor attention.")

    st.markdown("#### Ten students with the highest predicted risk")
    top10 = data.sort_values("risk_prob", ascending=False).head(10)[
        ["student_id", "name", "risk_prob", "attendance_pct", "internal2", "assignment_pct"]].copy()
    top10["risk_prob"] = (top10.risk_prob * 100).round(0)
    top10.columns = ["ID", "Name", "Risk %", "Attendance %", "Internal 2", "Assignments %"]
    st.dataframe(top10, hide_index=True)

    summary = (f"StudentPulse AI - Executive Summary\n{COLLEGE}\n{DEPARTMENT}\n\n"
               f"Students analysed: {n}\nHigh risk: {high} ({high / n:.0%})\nMedium risk: {med} ({med / n:.0%})\n"
               f"Below 75% attendance: {low_att} ({low_att / n:.0%})\n"
               f"Average attendance: {data.attendance_pct.mean():.1f}%\n"
               f"Prepared for: {user['name']} ({auth.ROLE_LABEL[user['role']]})\n")
    st.download_button("⬇ Download executive summary", summary.encode(), "executive_summary.txt")


# ================================================================== STUDENT EXPLORER
elif page == "Student Explorer":
    hero("Student Explorer", "Individual risk, the reasons behind it, and what would help")
    options = data.apply(lambda r: f"{r.student_id} - {r['name']}", axis=1).tolist()
    pick = st.selectbox("Search / select a student", options)
    sid = pick.split(" - ")[0]
    base_row = data[data.student_id == sid].iloc[0]
    row = base_row.copy()
    old_p = float(base_row.risk_prob)

    st.markdown("#### What-if simulator")
    st.caption("Move the sliders to see how the risk would change if this student improved.")
    s1, s2, s3, s4 = st.columns(4)
    row["attendance_pct"] = s1.slider("Attendance %", 30.0, 100.0, float(base_row.attendance_pct), 0.5, key=f"a{sid}")
    row["internal2"] = s2.slider("Internal 2 marks", 0.0, 100.0, float(base_row.internal2), 1.0, key=f"i{sid}")
    row["assignment_pct"] = s3.slider("Assignments on time %", 0.0, 100.0, float(base_row.assignment_pct), 1.0, key=f"s{sid}")
    row["study_hours"] = s4.slider("Study hours / week", 0.0, 14.0, float(base_row.study_hours), 0.5, key=f"h{sid}")

    new_p = float(core.predict_risk(pd.DataFrame([row]), bundle)[0])
    band = core.risk_band(new_p)

    g, info = st.columns([1, 1])
    with g:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=new_p * 100,
            number={"suffix": "%"}, delta={"reference": old_p * 100, "increasing": {"color": "#c0392b"},
                                           "decreasing": {"color": "#2e8b57"}},
            title={"text": "Predicted risk"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": BAND_COLORS[band]},
                   "steps": [{"range": [0, 35], "color": "#e3f4ea"},
                             {"range": [35, 65], "color": "#fdf2d0"},
                             {"range": [65, 100], "color": "#f9dcd8"}]}))
        st.plotly_chart(style_fig(fig, 300))
    with info:
        st.markdown(f'<div class="card"><b>{base_row["name"]}</b> &nbsp; ({sid})<br><br>'
                    f'Risk band: <span class="pill" style="background:{BAND_COLORS[band]}">{band}</span><br><br>'
                    f'Original risk: <b>{old_p:.0%}</b> &nbsp;→&nbsp; With changes: <b>{new_p:.0%}</b><br>'
                    f'Attendance: <b>{row.attendance_pct:.1f}%</b> (trend {base_row.attendance_trend:+.1f}) &nbsp;|&nbsp; '
                    f'Internals: <b>{base_row.internal1:.0f} → {row.internal2:.0f}</b><br>'
                    f'Previous GPA: <b>{base_row.prev_gpa:.2f}</b> &nbsp;|&nbsp; '
                    f'Mentor meetings: <b>{int(base_row.mentor_meetings)}</b></div>', unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        imp = core.explain_student(row, bundle)
        imp_df = pd.DataFrame([(core.FEATURE_LABELS[f], v) for f, v in imp], columns=["Factor", "Effect"])
        imp_df["Direction"] = imp_df["Effect"].apply(lambda v: "Raises risk" if v > 0 else "Lowers risk")
        fig = px.bar(imp_df.sort_values("Effect"), x="Effect", y="Factor", orientation="h", color="Direction",
                     color_discrete_map={"Raises risk": "#c0392b", "Lowers risk": "#2e8b57"},
                     title="Why this risk? (right = pushes risk up)")
        st.plotly_chart(style_fig(fig, 380))
    with b:
        st.markdown("#### Recommended interventions")
        actions = core.recommend(row, new_p)
        for i, act in enumerate(actions, 1):
            st.markdown(f'<div class="action"><b>{i}.</b> {act}</div>', unsafe_allow_html=True)
        report = (f"StudentPulse AI - Mentor Note\n{COLLEGE}\n\nStudent: {base_row['name']} ({sid})\n"
                  f"Predicted risk: {new_p:.0%} ({band})\n\nRecommended actions:\n"
                  + "\n".join(f"{i}. {x}" for i, x in enumerate(actions, 1)))
        st.download_button("⬇ Download mentor note", report.encode(), f"mentor_note_{sid}.txt")


# ================================================================== EARLY WARNING
elif page == "Early Warning":
    hero("Early Warning List", "Students who need attention first, with the most important action for each")
    f1, f2 = st.columns([2, 1])
    bands = f1.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium"])
    att_limit = f2.slider("Attendance below (%)", 40, 100, 100)
    warn = data[data.risk_band.isin(bands) & (data.attendance_pct <= att_limit)] \
        .sort_values("risk_prob", ascending=False).copy()
    st.write(f"**{len(warn)} students** match the filters")
    if len(warn):
        warn["Risk %"] = (warn.risk_prob * 100).round(0)
        warn["Top action"] = warn.apply(lambda r: core.recommend(r, r.risk_prob)[0], axis=1)
        show = warn[["student_id", "name", "Risk %", "risk_band", "attendance_pct", "attendance_trend",
                     "internal2", "assignment_pct", "Top action"]]
        show = show.rename(columns={"student_id": "ID", "name": "Name", "risk_band": "Band",
                                    "attendance_pct": "Attendance %", "attendance_trend": "Trend",
                                    "internal2": "Internal 2", "assignment_pct": "Assignments %"})
        st.dataframe(show, hide_index=True, column_config={
            "Risk %": st.column_config.ProgressColumn("Risk %", min_value=0, max_value=100, format="%.0f")})
        st.download_button("⬇ Download mentor report (CSV)", show.to_csv(index=False).encode(),
                           "early_warning_report.csv", "text/csv")


# ================================================================== DATASET
elif page == "Dataset":
    hero("Dataset", "The data behind the analytics: browse, filter, understand and download")
    if source.startswith("Sample"):
        st.info("This sample dataset is **synthetic** (computer-generated with realistic relationships) "
                "because real student records are private. Names are fictional.")
    t1, t2, t3 = st.tabs(["Browse & filter", "Summary statistics", "Data dictionary"])

    with t1:
        c1, c2, c3 = st.columns(3)
        q = c1.text_input("Search by ID or name")
        rng = c2.slider("Attendance range (%)", 30, 100, (30, 100))
        bnd = c3.multiselect("Predicted risk band", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
        view = data[data.attendance_pct.between(*rng) & data.risk_band.isin(bnd)]
        if q:
            view = view[view.student_id.str.contains(q, case=False) | view["name"].str.contains(q, case=False)]
        st.write(f"Showing **{len(view)}** of {len(data)} records")
        st.dataframe(view, hide_index=True, height=420)
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Download original dataset (CSV)", raw.to_csv(index=False).encode(), "students.csv", "text/csv")
        d2.download_button("⬇ Download with predictions (CSV)", data.to_csv(index=False).encode(),
                           "students_with_predictions.csv", "text/csv")

    with t2:
        c = st.columns(4)
        kpi(c[0], len(raw), "Records")
        kpi(c[1], len(core.FEATURES), "Input features", TEAL)
        if "at_risk" in raw.columns:
            kpi(c[2], f"{raw.at_risk.mean():.0%}", "Labelled at-risk", BAND_COLORS["High"])
        kpi(c[3], int(raw[core.FEATURES].isna().sum().sum()), "Missing values", "#8e44ad")
        st.write("")
        desc = raw[core.FEATURES].describe().T.round(2)
        desc.index = [core.FEATURE_LABELS[i] for i in desc.index]
        st.dataframe(desc)
        feat = st.selectbox("Distribution of", core.FEATURES, format_func=lambda f: core.FEATURE_LABELS[f])
        color = "at_risk" if "at_risk" in raw.columns else None
        fig = px.histogram(raw.assign(at_risk=raw["at_risk"].map({0: "On track", 1: "At risk"})) if color else raw,
                           x=feat, color=color, nbins=30, barmode="overlay", opacity=0.7,
                           labels={feat: core.FEATURE_LABELS[feat]},
                           color_discrete_map={"On track": TEAL, "At risk": "#c0392b"})
        st.plotly_chart(style_fig(fig, 340))

    with t3:
        st.dataframe(pd.DataFrame(core.DATA_DICTIONARY, columns=["Column", "Meaning", "Range"]), hide_index=True)


# ================================================================== MODEL REPORT
elif page == "Model Report":
    hero("Model Report", "How the prediction models were trained, compared and evaluated")
    met = core.load_metrics()
    st.markdown(f"Three models were trained on **{met['n_train']}** students and tested on **{met['n_test']}** "
                f"unseen students. The best model by F1-score on the at-risk class is **{met['best']}**.")
    res = pd.DataFrame(met["results"]).T
    res.index.name = "Model"
    st.dataframe(res)

    long = res.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group", title="Model comparison",
                 color_discrete_sequence=[NAVY, TEAL, "#e0a100"])
    st.plotly_chart(style_fig(fig, 380))

    a, b = st.columns(2)
    with a:
        cm = met["confusion"]
        fig = px.imshow([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], text_auto=True,
                        x=["Predicted on track", "Predicted at-risk"], y=["Actually on track", "Actually at-risk"],
                        color_continuous_scale="Blues", title=f"Confusion matrix ({met['best']})")
        st.plotly_chart(style_fig(fig, 360))
        st.caption("Bottom-left = at-risk students the model missed. Mentors care most about keeping this small.")
    with b:
        fig = go.Figure()
        fig.add_scatter(x=met["roc"]["fpr"], y=met["roc"]["tpr"], mode="lines", name="Model", line=dict(color=NAVY, width=3))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess", line=dict(dash="dash", color="grey"))
        fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(style_fig(fig, 360))

    if hasattr(bundle["model"], "feature_importances_"):
        fi = pd.DataFrame({"Factor": [core.FEATURE_LABELS[f] for f in core.FEATURES],
                           "Importance": bundle["model"].feature_importances_}).sort_values("Importance")
        fig = px.bar(fi, x="Importance", y="Factor", orientation="h", title="Which factors matter most",
                     color_discrete_sequence=[TEAL])
        st.plotly_chart(style_fig(fig, 360))

    if st.button("Retrain models on the current dataset"):
        if "at_risk" in raw.columns:
            core.train_all(raw)
            st.cache_resource.clear()
            st.success("Retrained successfully. Refresh the page to see the updated results.")
        else:
            st.warning("The current dataset has no 'at_risk' column, so it cannot be used for training.")


# ================================================================== USER MANAGEMENT
elif page == "User Management" and is_mgmt:
    hero("User Management", "Create staff accounts, reset passwords and review who has signed in")
    t1, t2 = st.tabs(["Accounts", "Access log"])

    with t1:
        users = auth.list_users()
        st.dataframe(pd.DataFrame(users).rename(columns={"username": "Username", "name": "Name", "role": "Role"}),
                     hide_index=True)
        a, b = st.columns(2)
        with a:
            st.markdown("#### Add an account")
            with st.form("add_user", clear_on_submit=True):
                nu = st.text_input("Username (letters and numbers)")
                nn = st.text_input("Full name")
                nr = st.selectbox("Role", list(auth.ROLES), format_func=lambda r: auth.ROLE_LABEL[r])
                npw = st.text_input("Temporary password", type="password")
                if st.form_submit_button("Create account"):
                    err = auth.add_user(nu, nn, nr, npw)
                    if err:
                        st.error(err)
                    else:
                        auth.log_event(user["username"], user["role"], f"created account {nu.strip().lower()}")
                        st.success(f"Account '{nu.strip().lower()}' created.")
        with b:
            st.markdown("#### Reset password or remove an account")
            names = [u["username"] for u in users]
            with st.form("reset_user", clear_on_submit=True):
                target = st.selectbox("Account", names)
                rpw = st.text_input("New password", type="password")
                c1, c2 = st.columns(2)
                do_reset = c1.form_submit_button("Reset password")
                do_remove = c2.form_submit_button("Remove account")
                if do_reset:
                    err = auth.reset_password(target, rpw)
                    if err:
                        st.error(err)
                    else:
                        auth.log_event(user["username"], user["role"], f"reset password for {target}")
                        st.success(f"Password reset for '{target}'.")
                if do_remove:
                    err = auth.remove_user(target, user["username"])
                    if err:
                        st.error(err)
                    else:
                        auth.log_event(user["username"], user["role"], f"removed account {target}")
                        st.success(f"Account '{target}' removed. Refresh to update the list.")

    with t2:
        log = pd.DataFrame(auth.read_log())
        if log.empty:
            st.info("No activity recorded yet.")
        else:
            st.write(f"Showing the latest {min(len(log), 200)} of {len(log)} events (newest first)")
            st.dataframe(log.head(200), hide_index=True)
            st.download_button("⬇ Download full access log (CSV)", log.to_csv(index=False).encode(),
                               "access_log.csv", "text/csv")


# ================================================================== MY ACCOUNT
elif page == "My Account":
    hero("My Account", "Your profile and password")
    st.markdown(f'**Name:** {user["name"]}  \n**Username:** {user["username"]}  \n'
                f'**Role:** {auth.ROLE_LABEL[user["role"]]}')
    st.markdown("#### Change password")
    with st.form("change_pw", clear_on_submit=True):
        old = st.text_input("Current password", type="password")
        new1 = st.text_input("New password", type="password")
        new2 = st.text_input("Repeat new password", type="password")
        if st.form_submit_button("Update password"):
            if new1 != new2:
                st.error("The two new passwords do not match.")
            else:
                err = auth.change_password(user["username"], old, new1)
                if err:
                    st.error(err)
                else:
                    auth.log_event(user["username"], user["role"], "changed own password")
                    st.success("Password updated.")
    st.caption("Use at least 8 characters with upper and lower case letters and a digit.")


# ================================================================== ABOUT
else:
    hero("About the Project", "Artificial Intelligence-Based Student Attendance Analytics, Academic Performance "
                              "Prediction, and Early Intervention Recommendation System")
    a, b = st.columns(2)
    with a:
        st.markdown(f"""
**Institution:** {COLLEGE}
**Department:** {DEPARTMENT}
**Programme:** {BATCH}
**Team:** {", ".join(TEAM)}

#### Problem
Attendance and marks are recorded, but struggling students are usually noticed only after the results.
StudentPulse AI detects risk early and tells the mentor what to do about it.

#### How it works
1. Student data (attendance, trend, marks, assignments, GPA, study hours, mentor meetings)
2. Three machine-learning models are trained and the best one is selected
3. Each student gets a risk probability and a Low / Medium / High band
4. An explanation shows which factors raise or lower the risk
5. A rule engine converts the risk into prioritised mentor actions
""")
    with b:
        st.markdown("""
#### Technology
Python · pandas · scikit-learn · Plotly · Streamlit

#### Key features
- Attendance trend analysis, not just totals
- Explainable predictions
- What-if simulator for mentors
- Downloadable mentor reports
- Works with the sample data or your own CSV (Management)
- Role-based login for Staff and Management with an access log

#### Limitations
- Sample data is synthetic
- Eight input factors only
- Predictions guide mentors; they are not final decisions
- With real data, avoid sensitive attributes (gender, caste, religion)
""")
