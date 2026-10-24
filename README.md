# StudentPulse AI

**Artificial Intelligence-Based Student Attendance Analytics, Academic Performance Prediction, and Early Intervention Recommendation System**

Mini Project (2026-2027) · III AI&DS · Mount Zion College of Engineering and Technology · Batch 6

---

## 1. Problem statement
Colleges record attendance and internal marks but usually notice a struggling student only after the exam result.
StudentPulse AI uses attendance trends, marks, assignment behaviour and study habits to
1. analyse attendance and performance,
2. predict which students are at risk, and
3. tell the mentor exactly what action to take, and show what would change if the student improved.

## 2. What makes it unique
| Feature | Why it matters |
|---|---|
| **Attendance trend** (last 4 weeks vs first 4 weeks) | Catches a student who is *falling*, even if the overall % still looks fine |
| **"Why this risk?" explanation** | Shows which factors push risk up or down (log-odds method), so predictions are not a black box |
| **What-if simulator** | Mentor moves sliders (attendance, marks, assignments, study hours) and sees new risk instantly |
| **Intervention engine** | Converts prediction into prioritised, specific actions (counselling, remedial class, peer tutor...) |
| **Mentor report download** | One-click CSV of at-risk students with the top recommended action |
| **Upload your own CSV** | Works with real college data, not only the sample dataset |

## 3. Architecture
```
 CSV data ──► Feature set (8 factors) ──► 3 models trained & compared ──► Best model saved
                                                                              │
              Streamlit UI  ◄── risk score ── explanation ── intervention rules ┘
   Overview | Student Risk Explorer + What-if | Early Warning List | Model Report
```
Files: `core.py` (data, model, explanation, rules) · `app.py` (Streamlit UI) · `data/students.csv` · `models/`

## 4. How to run
**Windows (easiest):** extract the zip, open the `studentpulse` folder and double-click **`run_app.bat`**.
It installs the libraries the first time and opens the app in your browser at http://localhost:8501.

**Or from the terminal (Windows / Linux / Mac):**
```bash
cd studentpulse
pip install -r requirements.txt
streamlit run app.py
```
The dataset (`data/students.csv`) and trained model are already included. To retrain from the terminal: `python core.py`.

## 4a. App pages
Dashboard · Student Explorer (risk gauge, explanation, what-if sliders, mentor note) · Early Warning list ·
**Dataset** (browse/filter/search, summary statistics, data dictionary, CSV download) · Model Report (comparison, confusion matrix, ROC) · About.
Edit the college/team details at the top of `app.py`.

## 4b. Login and roles
| Role | Can do |
|---|---|
| **Staff** (faculty / mentors) | Dashboard, Student Explorer (risk, reasons, what-if, mentor note), Early Warning list, view/download dataset, change own password |
| **Management** (HOD / Principal) | Everything Staff can, plus Management Summary, Model Report and retraining, uploading own CSV, User Management (add accounts, reset passwords) and the Access Log |

Demo accounts (created on first run): `staff / Staff@123`, `hod / Hod@123`, `principal / Principal@123`.
**Change these passwords** under *My Account* / *User Management*, and set `SHOW_DEMO_LOGINS = False` in `app.py` to hide the hint on the login page.

How it is secured: passwords are salted and hashed (PBKDF2-SHA256, never stored as text), 5 wrong attempts lock sign-in for 60 seconds,
every sign-in / failed attempt / account change is written to `data/access_log.csv`, and a Staff account cannot sign in as Management.
This is suitable for a college demo or internal pilot; a real deployment should use the college single sign-on (SSO).

## 5. Dataset
400 synthetic students with: attendance %, attendance trend, internal 1 & 2, assignment %, previous GPA,
study hours, mentor meetings, and an `at_risk` label (generated from a hidden engagement/ability factor plus noise).
**Be upfront about this in your viva**: the sample data is synthetic because real student data is private.
To use real data, upload a CSV with the same columns (and an `at_risk` column if you want to retrain).

## 6. Models and results (held-out 25% test set)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.78 | 0.43 | 0.72 | 0.54 | 0.87 |
| **Random Forest (best by F1)** | 0.85 | 0.57 | 0.72 | 0.63 | 0.86 |
| Gradient Boosting | 0.84 | 0.56 | 0.56 | 0.56 | 0.87 |

Your numbers may differ slightly after regenerating data. We pick the model by **F1 on the at-risk class**
(and prefer high recall) because missing a struggling student is worse than a false alarm.

## 7. Viva questions you should be ready for
**Why not just use accuracy?** Only ~18% of students are at risk, so a model that says "nobody is at risk" would still score ~82%. Precision, recall and F1 on the at-risk class are more honest.

**Why did you choose recall-focused selection?** A false alarm costs a mentor 10 minutes. A missed at-risk student can fail a semester.

**How does the explanation work?** For each factor we replace the student's value with the class median and see how much the risk (log-odds) drops. A big drop means that factor is pushing the risk up. It works with any model.

**Is the dataset real?** No, it is synthetic, generated with realistic relationships. The system is built so a college can upload real data in the same format and retrain.

**What are the limitations?** Synthetic data, only 8 factors, no subject-wise breakdown, and predictions are advice for mentors, not final decisions. Privacy and fairness need care with real data (do not use gender, caste or religion as inputs).

**Future scope:** subject-wise attendance, LLM-written mentor emails, SMS/WhatsApp alerts to parents, integration with the college ERP, and model retraining every semester.

## 8. Report outline (for your documentation)
1. Abstract · 2. Introduction & problem statement · 3. Literature survey (educational data mining, student dropout prediction)
4. Proposed system & architecture · 5. Dataset & features · 6. Algorithms (Logistic Regression, Random Forest, Gradient Boosting)
7. Implementation · 8. Results & screenshots · 9. Conclusion & future scope · 10. References
