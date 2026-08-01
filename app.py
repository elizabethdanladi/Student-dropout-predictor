"""
Student Dropout Risk Predictor - Streamlit Demo App (AI-08)

Run with:
    pip install streamlit joblib pandas scikit-learn
    streamlit run app.py

Expects "student_dropout_model.pkl" (saved from the notebook) in the same folder.
"""

import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Dropout Risk Predictor", page_icon="🎓", layout="centered")

# Human-readable names for the raw model columns (used in the input form,
# tied to plain-language labels here so charts/results don't show raw code names)
FRIENDLY_NAMES = {
    "state": "State",
    "school_type": "School type",
    "gender": "Gender",
    "age": "Age",
    "class_level": "Class level",
    "distance_to_school_km": "Distance to school (km)",
    "household_income_band": "Household income",
    "parent_education_level": "Parent's education level",
    "num_siblings": "Number of siblings",
    "part_time_work": "Part-time work / trading",
    "has_scholarship": "Has scholarship",
    "family_support": "Family support at home",
    "health_issues": "Health issues",
    "attendance_rate_pct": "Attendance rate",
    "term1_avg_score": "Term 1 average score",
    "term2_avg_score": "Term 2 average score",
    "term3_avg_score": "Term 3 average score",
    "overall_avg_score": "Overall average score",
    "extracurricular_involvement": "Extracurricular involvement",
}


def prettify_feature_name(raw_name: str) -> str:
    """
    Turn a raw pipeline feature name like 'cat__household_income_band_Low'
    or 'num__attendance_rate_pct' into something readable like
    'Household income: Low' or 'Attendance rate'.
    """
    name = raw_name.replace("cat__", "").replace("num__", "")

    # Try to match against known base columns, longest match first,
    # so e.g. "household_income_band_Low" splits into base + category value.
    for base_col in sorted(FRIENDLY_NAMES, key=len, reverse=True):
        if name == base_col:
            return FRIENDLY_NAMES[base_col]
        prefix = base_col + "_"
        if name.startswith(prefix):
            category_value = name[len(prefix):].replace("_", " ")
            return f"{FRIENDLY_NAMES[base_col]}: {category_value}"

    # Fallback: just clean it up
    return name.replace("_", " ").title()

# ---------- Load model ----------
MODEL_PATH = "student_dropout_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🎓 Student Dropout Risk Predictor")
st.caption("AI-08 · Nigerian Secondary School Context")

if model is None:
    st.error(
        f"Couldn't find `{MODEL_PATH}`. Run the notebook first — it saves the trained "
        f"model in Step 9 — then place the .pkl file in this same folder."
    )
    st.stop()

st.write(
    "Fill in a student's details below. The tool will estimate how likely they "
    "are to drop out of school, and explain **why** - so a teacher or school "
    "admin can decide whether to step in early."
)

with st.expander("ℹ️ What do the risk levels mean?"):
    st.markdown(
        """
- 🟢 **Low risk** - this student shows no strong warning signs right now.
- 🟠 **Medium risk** - some warning signs are present (e.g. falling attendance
  or grades). Worth keeping an eye on, or a light check-in.
- 🔴 **High risk** - several strong warning signs are present together
  (e.g. low attendance *and* low income *and* poor grades). Recommend a
  direct conversation or support plan for this student.

This is a **decision-support tool, not a final judgement** - it's meant to help
a human decide who to check in on first, not to label a student on its own.
        """
    )

# ---------- Options (must match training data categories) ----------
STATES = ["Lagos","Kano","Rivers","Oyo","Kaduna","Enugu","Anambra","Benue",
          "Sokoto","Borno","Delta","Edo","Plateau","Ogun","Imo","Katsina"]
CLASS_LEVELS = ["JSS1","JSS2","JSS3","SS1","SS2","SS3"]
INCOME_BANDS = ["Low","Middle","High"]
PARENT_EDU = ["None","Primary","Secondary","Tertiary"]

# ---------- Input form ----------
with st.form("student_form"):
    st.markdown("###  Key factors")
    st.caption(
        "These fields drive the vast majority of the prediction (~88% of the model's "
        "decision-making) — fill these in carefully."
    )

    col1, col2 = st.columns(2)
    with col1:
        state = st.selectbox("State", STATES)
        income_band = st.selectbox("Household income band", INCOME_BANDS, index=0)
        distance_km = st.number_input("Distance to school (km)", 0.0, 30.0, 3.0, step=0.5)
        family_support = st.radio("Family support at home?", ["Yes", "No"], horizontal=True, index=0)
    with col2:
        part_time_work = st.radio("Does student do part-time work/trading?", ["Yes", "No"], horizontal=True, index=1)
        attendance_rate = st.slider("Attendance rate (%)", 0.0, 100.0, 80.0, step=0.5)

    st.write("**Term grades**")
    c1, c2, c3 = st.columns(3)
    with c1:
        term1 = st.number_input("Term 1 avg score", 0.0, 100.0, 60.0)
    with c2:
        term2 = st.number_input("Term 2 avg score", 0.0, 100.0, 60.0)
    with c3:
        term3 = st.number_input("Term 3 avg score", 0.0, 100.0, 60.0)

    st.markdown("---")
    with st.expander("➕ Additional details (optional — together these affect less than 12% of the result)"):
        st.caption(
            "Feel free to skip these — individually each one barely moves the "
            "prediction. Sensible defaults are already set."
        )
        col3, col4 = st.columns(2)
        with col3:
            school_type = st.radio("School type", ["Public", "Private"], horizontal=True)
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
            age = st.slider("Age", 10, 18, 15)
            class_level = st.selectbox("Class level", CLASS_LEVELS, index=3)
        with col4:
            num_siblings = st.slider("Number of siblings", 0, 10, 3)
            parent_edu = st.selectbox("Parent education level", PARENT_EDU, index=1)
            has_scholarship = st.radio("Has scholarship?", ["Yes", "No"], horizontal=True, index=1)
            health_issues = st.radio("Health issues?", ["Yes", "No"], horizontal=True, index=1)
            extracurricular = st.radio("Involved in extracurriculars?", ["Yes", "No"], horizontal=True, index=1)

    submitted = st.form_submit_button("Predict Risk", use_container_width=True)

# ---------- Predict ----------
if submitted:
    overall_avg = round((term1 + term2 + term3) / 3, 1)

    student = {
        "state": state,
        "school_type": school_type,
        "gender": gender,
        "age": age,
        "class_level": class_level,
        "distance_to_school_km": distance_km,
        "household_income_band": income_band,
        "parent_education_level": parent_edu,
        "num_siblings": num_siblings,
        "part_time_work": part_time_work,
        "has_scholarship": has_scholarship,
        "family_support": family_support,
        "health_issues": health_issues,
        "attendance_rate_pct": attendance_rate,
        "term1_avg_score": term1,
        "term2_avg_score": term2,
        "term3_avg_score": term3,
        "overall_avg_score": overall_avg,
        "extracurricular_involvement": extracurricular,
    }

    input_df = pd.DataFrame([student])
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    classes = model.named_steps["model"].classes_
    proba_dict = dict(zip(classes, proba))

    st.markdown("---")
    st.subheader("Result")

    color_map = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}

    outcome_word = "retention" if pred == "Low" else "dropout"
    confidence_pct = proba_dict.get(pred, 0) * 100

    st.markdown(
        f"### {color_map.get(pred, '')} Risk of dropout is **{pred}**, "
        f"with a **{confidence_pct:.0f}% chance** of **{outcome_word}**."
    )

    with st.expander("See full probability breakdown"):
        cols = st.columns(3)
        for col, level in zip(cols, ["Low", "Medium", "High"]):
            pct = proba_dict.get(level, 0) * 100
            col.metric(f"{color_map[level]} {level}", f"{pct:.0f}%")

    # ---------- Contributing factors (rule-based flags + global importance) ----------
    st.subheader(" Why this result?")
    st.caption("Here's what's driving this specific student's result, explained in plain terms.")

    # (icon, plain explanation) for each possible flag
    flag_explanations = []
    if attendance_rate < 70:
        flag_explanations.append((
             "**Misses school often** ({attendance_rate:.0f}% attendance) — "
            "students who are frequently absent are far more likely to fall behind and eventually drop out."
        ))
    if overall_avg < 50:
        flag_explanations.append((
             f"**Struggling academically** (average score {overall_avg:.0f}/100) — "
            "falling grades are often an early sign a student is losing motivation to continue."
        ))
    if income_band == "Low":
        flag_explanations.append((
            "**Comes from a lower-income household** — financial pressure at home can "
            "force a student to prioritize work or family needs over school."
        ))
    if part_time_work == "Yes":
        flag_explanations.append((
             "**Works or trades alongside school** — balancing a job with schoolwork "
            "often eats into study time and attendance."
        ))
    if family_support == "No":
        flag_explanations.append((
             "**Limited support at home** — students without encouragement or help at "
            "home are more likely to disengage from school."
        ))
    if distance_km > 5:
        flag_explanations.append((
            "🚸", f"**Long commute to school** ({distance_km:.1f} km) — a longer, harder journey "
            "makes it easier to skip school, especially in bad weather or when transport costs rise."
        ))
    if health_issues == "Yes":
        flag_explanations.append((
            "🩺", "**Has ongoing health issues** — health problems can disrupt attendance and "
            "make it harder to keep up with schoolwork."
        ))

    if flag_explanations:
        for icon, text in flag_explanations:
            st.markdown(f"{icon} {text}")
    else:
        st.success("✅ No major warning signs detected for this student.")

    st.markdown("---")

    with st.expander("📚 What matters most, in general (across all students)?"):
        st.write(
            "This isn't about the student above — it's what the model has learned "
            "tends to matter most for **any** student, based on everyone it was trained on."
        )

        # Plain-language description for each underlying factor (used regardless of student)
        PLAIN_EXPLANATIONS = {
            "Attendance rate": "How often a student shows up to school.",
            "Overall average score": "A student's average grades across the school year.",
            "Term 1 average score": "Grades in the first term.",
            "Term 2 average score": "Grades in the second term.",
            "Term 3 average score": "Grades in the third term.",
            "Household income": "How much financial pressure the family is under.",
            "Part-time work / trading": "Whether the student works or trades alongside school.",
            "Family support at home": "Whether the student has support and encouragement at home.",
            "Distance to school (km)": "How far the student has to travel to get to school.",
            "Health issues": "Whether the student has ongoing health problems.",
            "Has scholarship": "Whether the student receives financial help for school.",
            "Parent's education level": "How much formal education the student's parents received.",
        }

        feature_names = model.named_steps["preprocess"].get_feature_names_out()
        importances = model.named_steps["model"].feature_importances_

        fi_df = pd.DataFrame({
            "feature": [prettify_feature_name(f) for f in feature_names],
            "importance": importances
        })
        fi_df["group"] = fi_df["feature"].str.split(":").str[0]
        grouped = fi_df.groupby("group")["importance"].sum().sort_values(ascending=False).head(5)

        # Convert to a percentage share (of these top 5) so numbers are intuitive —
        # e.g. "32%" instead of a raw, meaningless importance score like "0.14".
        percentages = (grouped / grouped.sum() * 100).round(0)

        # Simple, clean horizontal bar chart: no axis numbers/gridlines to interpret,
        # just the factor name and its % printed directly on the bar.
        chart_order = percentages.sort_values(ascending=True)  # so #1 lands at the top
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.barh(chart_order.index, chart_order.values, color="#4C78A8", height=0.6)
        for bar, pct in zip(bars, chart_order.values):
            ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.0f}%", va="center", fontsize=11, fontweight="bold")
        ax.set_xlim(0, max(chart_order.values) * 1.3)
        ax.set_xticks([])  # hide confusing axis numbers entirely
        ax.set_yticks(range(len(chart_order)))
        ax.set_yticklabels(chart_order.index, fontsize=11)
        for spine in ["top", "right", "bottom"]:
            ax.spines[spine].set_visible(False)
        ax.set_title("Top 5 factors, by how much they matter", fontsize=12, pad=10)
        plt.tight_layout()
        st.pyplot(fig)

        st.write("")  # small spacing

        for rank, (factor, score) in enumerate(grouped.items(), start=1):
            description = PLAIN_EXPLANATIONS.get(factor, "")
            st.markdown(f"**{rank}. {factor}** ({percentages[factor]:.0f}%) — {description}")

st.markdown("---")
st.caption(
    "Model trained on a synthetic dataset engineered to reflect Nigerian secondary-school "
    "dropout patterns (attendance, income, distance, part-time work, etc.). Not based on "
    "real student records."
)
