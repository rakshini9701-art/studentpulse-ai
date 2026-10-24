"""
StudentPulse AI - core logic (data, model, explanations, interventions).
Kept separate from the UI so it is easy to test and easy to explain in a viva.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
DATA_FILE = BASE / "data" / "students.csv"
MODEL_FILE = BASE / "models" / "best_model.joblib"
METRICS_FILE = BASE / "models" / "metrics.json"

FEATURES = [
    "attendance_pct",       # overall attendance %
    "attendance_trend",     # last 4 weeks minus first 4 weeks (negative = dropping)
    "internal1",            # internal test 1 (out of 100)
    "internal2",            # internal test 2 (out of 100)
    "assignment_pct",       # assignments submitted on time %
    "prev_gpa",             # previous semester GPA (out of 10)
    "study_hours",          # self-study hours per week
    "mentor_meetings",      # mentor meetings attended this semester
]

FEATURE_LABELS = {
    "attendance_pct": "Attendance %",
    "attendance_trend": "Attendance trend",
    "internal1": "Internal test 1",
    "internal2": "Internal test 2",
    "assignment_pct": "Assignment submission %",
    "prev_gpa": "Previous GPA",
    "study_hours": "Weekly study hours",
    "mentor_meetings": "Mentor meetings",
}

DATA_DICTIONARY = [
    ("student_id", "Unique student ID", "text"),
    ("name", "Student name (sample names are fictional)", "text"),
    ("attendance_pct", "Overall attendance percentage this semester", "30-100 %"),
    ("attendance_trend", "Average attendance of last 4 weeks minus first 4 weeks (negative = falling)", "-30 to +20"),
    ("internal1", "Internal assessment test 1 marks", "0-100"),
    ("internal2", "Internal assessment test 2 marks", "0-100"),
    ("assignment_pct", "Percentage of assignments submitted on time", "0-100 %"),
    ("prev_gpa", "Previous semester GPA", "0-10"),
    ("study_hours", "Self-study hours per week", "0-14"),
    ("mentor_meetings", "Mentor meetings attended this semester", "0-6"),
    ("at_risk", "TARGET: 1 = student at risk of poor result, 0 = on track", "0 / 1"),
]

FIRST = ["Aarav", "Ananya", "Arjun", "Divya", "Karthik", "Lakshmi", "Meena", "Naveen", "Priya", "Rahul",
         "Sanjay", "Sneha", "Surya", "Tamil", "Vignesh", "Yamini", "Deepak", "Harini", "Kavin", "Nithya"]
LAST = ["Kumar", "Raj", "S", "M", "R", "K", "V", "P", "G", "A"]


# ---------------------------------------------------------------- data
def generate_data(n=400, seed=42):
    """Synthetic but realistic student data driven by a hidden 'engagement' and 'ability' factor."""
    rng = np.random.default_rng(seed)
    engagement = rng.beta(5, 2, n)          # how involved the student is (0-1)
    ability = rng.normal(0, 1, n)

    attendance = np.clip(55 + 45 * engagement + rng.normal(0, 5, n), 35, 100)
    trend = np.clip(rng.normal(0, 6, n) + (engagement - 0.7) * 25, -30, 20)
    study = np.clip(1 + 9 * engagement + rng.normal(0, 1.5, n), 0, 14)
    assign = np.clip(40 + 60 * engagement + rng.normal(0, 8, n), 10, 100)
    gpa = np.clip(6.3 + 1.2 * ability + 1.2 * engagement + rng.normal(0, 0.5, n), 3, 10)
    i1 = np.clip(35 + 8 * ability + 35 * engagement + rng.normal(0, 8, n), 5, 100)
    i2 = np.clip(i1 + 0.4 * trend + rng.normal(0, 7, n), 5, 100)
    mentor = np.clip(np.round(rng.poisson(1 + 2 * engagement)), 0, 6)

    # risk score used only to create the label (the model never sees this formula)
    score = (0.35 * (attendance - 75) / 12 + 0.25 * (i2 - 50) / 18 + 0.15 * (assign - 60) / 20
             + 0.15 * (gpa - 6.5) / 1.3 + 0.10 * trend / 8 + 0.05 * (study - 5) / 3
             + rng.normal(0, 0.6, n))
    at_risk = (score < -0.05).astype(int)

    names = [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n)]
    df = pd.DataFrame({
        "student_id": [f"AIDS{1000 + i}" for i in range(n)],
        "name": names,
        "attendance_pct": attendance.round(1),
        "attendance_trend": trend.round(1),
        "internal1": i1.round(0),
        "internal2": i2.round(0),
        "assignment_pct": assign.round(0),
        "prev_gpa": gpa.round(2),
        "study_hours": study.round(1),
        "mentor_meetings": mentor.astype(int),
        "at_risk": at_risk,
    })
    return df


def ensure_data():
    if not DATA_FILE.exists():
        DATA_FILE.parent.mkdir(exist_ok=True)
        generate_data().to_csv(DATA_FILE, index=False)
    return pd.read_csv(DATA_FILE)


# ---------------------------------------------------------------- model
def train_all(df=None, seed=42):
    """Train 3 models, compare them, save the best (by F1 on the at-risk class)."""
    df = ensure_data() if df is None else df
    X, y = df[FEATURES], df["at_risk"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=seed),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }
    results, fitted = {}, {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        pred, proba = m.predict(X_te), m.predict_proba(X_te)[:, 1]
        results[name] = {
            "accuracy": round(accuracy_score(y_te, pred), 3),
            "precision": round(precision_score(y_te, pred), 3),
            "recall": round(recall_score(y_te, pred), 3),
            "f1": round(f1_score(y_te, pred), 3),
            "roc_auc": round(roc_auc_score(y_te, proba), 3),
        }
        fitted[name] = m
    best = max(results, key=lambda k: results[k]["f1"])
    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump({"model": fitted[best], "name": best, "medians": X.median().to_dict()}, MODEL_FILE)
    # confusion matrix + ROC curve of the best model on the test set (for the Model Report page)
    best_pred = fitted[best].predict(X_te)
    best_proba = fitted[best].predict_proba(X_te)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_te, best_pred).ravel()
    fpr, tpr, _ = roc_curve(y_te, best_proba)
    out = {"best": best, "results": results, "n_train": len(X_tr), "n_test": len(X_te),
           "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
           "roc": {"fpr": [round(float(v), 4) for v in fpr], "tpr": [round(float(v), 4) for v in tpr]}}
    METRICS_FILE.write_text(json.dumps(out, indent=2))
    return out


def load_model():
    if not MODEL_FILE.exists():
        train_all()
    return joblib.load(MODEL_FILE)


def load_metrics():
    if not METRICS_FILE.exists():
        train_all()
    return json.loads(METRICS_FILE.read_text())


def predict_risk(df, bundle=None):
    bundle = bundle or load_model()
    return bundle["model"].predict_proba(df[FEATURES])[:, 1]


def risk_band(p):
    return "High" if p >= 0.65 else "Medium" if p >= 0.35 else "Low"


# ---------------------------------------------------------------- explanation
def explain_student(row, bundle=None):
    """
    Model-agnostic explanation: for each feature, replace the student's value with the
    cohort median and see how much the risk drops. Big drop = that feature is pushing risk up.
    Impact is measured on the log-odds scale so it stays meaningful even when risk is near 0% or 100%.
    Returns list of (feature, impact) sorted by impact, where impact > 0 raises risk.
    """
    bundle = bundle or load_model()
    model, med = bundle["model"], bundle["medians"]
    x = pd.DataFrame([{f: row[f] for f in FEATURES}])

    def logit(p):
        p = min(max(p, 0.001), 0.999)
        return float(np.log(p / (1 - p)))

    base = logit(model.predict_proba(x)[0, 1])
    impacts = []
    for f in FEATURES:
        x2 = x.copy()
        x2[f] = med[f]
        impacts.append((f, base - logit(model.predict_proba(x2)[0, 1])))
    return sorted(impacts, key=lambda t: -t[1])


# ---------------------------------------------------------------- interventions
def recommend(row, risk):
    """Rule engine -> prioritised, specific actions (this is the 'early intervention' part)."""
    acts = []  # (priority, action)
    if row["attendance_pct"] < 65:
        acts.append((1, "Critical attendance: call parent/guardian and hold a one-to-one counselling session this week."))
    elif row["attendance_pct"] < 75:
        acts.append((2, "Attendance below 75% (eligibility risk): send an alert and agree an attendance recovery plan."))
    if row["attendance_trend"] < -8:
        acts.append((2, "Attendance is falling sharply over recent weeks: ask the mentor to find out the reason (health, transport, motivation)."))
    if row["internal2"] < 45 or row["internal1"] < 45:
        acts.append((1, "Weak internal marks: enrol in remedial classes and give a topic-wise revision plan."))
    elif row["internal2"] + 5 < row["internal1"]:
        acts.append((3, "Marks dropped between Internal 1 and 2: run a short doubt-clearing session."))
    if row["assignment_pct"] < 60:
        acts.append((2, "Low assignment submission: set weekly mini-deadlines and track them with the class advisor."))
    if row["study_hours"] < 3:
        acts.append((3, "Very low self-study time: suggest a fixed daily study slot or peer study group."))
    if row["prev_gpa"] < 6:
        acts.append((2, "Weak academic history: assign a peer tutor / senior buddy for core subjects."))
    if row["mentor_meetings"] < 2 and risk >= 0.35:
        acts.append((2, "Few mentor meetings so far: schedule a mentor meeting within 7 days."))
    if not acts:
        acts.append((4, "On track. Keep encouraging; consider a peer-mentor role."))
    return [a for _, a in sorted(acts)]


def score_all(df, bundle=None):
    bundle = bundle or load_model()
    out = df.copy()
    out["risk_prob"] = predict_risk(out, bundle).round(3)
    out["risk_band"] = out["risk_prob"].apply(risk_band)
    return out


if __name__ == "__main__":
    print(json.dumps(train_all(), indent=2))
