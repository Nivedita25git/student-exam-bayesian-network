"""Streamlit interface for the Student Exam Performance Predictor.

Run with:  streamlit run app.py
Keep this file in the same folder as student_exam_bayesnet.py.
"""

import pandas as pd
import streamlit as st

from student_exam_bayesnet import (
    DOMAINS,
    BayesianNetwork,
    generate_dataset,
    risk_label,
    save_dataset,
)

st.set_page_config(page_title="Exam Performance Predictor", page_icon="🎓", layout="centered")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 760px;}
    .result-card {border-radius: 16px; padding: 1.6rem; text-align: center; color: white;}
    .result-card h1 {font-size: 3.2rem; margin: 0; color: white;}
    .result-card p {margin: 0.2rem 0 0; font-size: 1.15rem; letter-spacing: 0.05em;}
    </style>
    """,
    unsafe_allow_html=True,
)

LABELS = {
    "Study": ("📚 Study hours", ["Low", "High"]),
    "Assign": ("📝 Assignments", ["Incomplete", "Complete"]),
    "Prev": ("📊 Previous marks", ["Low", "Good"]),
    "Attend": ("🕐 Attendance", ["Low", "High"]),
}

NETWORK_DOT = """
digraph {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fillcolor="#EEF2FF", color="#6366F1", fontname="Helvetica"];
    Study -> Prep; Assign -> Prep;
    Prep -> Perf; Prev -> Perf; Attend -> Perf;
    Perf -> Result;
}
"""


@st.cache_resource
def load_model():
    data = generate_dataset()
    save_dataset(data)
    network = BayesianNetwork()
    network.fit(data)
    return network, data


network, dataset = load_model()

st.title("🎓 Exam Performance Predictor")
st.caption("A Bayesian Network that estimates the chance of passing, using exact and approximate inference.")

st.subheader("Student details")
st.write("Choose what you know. Leave a factor as *Unknown* if you don't have the information.")

evidence = {}
columns = st.columns(2)
for i, (var, (label, options)) in enumerate(LABELS.items()):
    choice = columns[i % 2].selectbox(label, ["Unknown"] + options, key=var)
    if choice != "Unknown":
        evidence[var] = choice

if st.button("Predict result", type="primary", use_container_width=True):
    exact = network.query("Result", evidence)
    approx = network.likelihood_weighting("Result", evidence)
    prep, prep_prob = network.map_query(["Prep"], evidence)

    p_pass = exact["Pass"]
    colour = "#16A34A" if p_pass >= 0.75 else "#F59E0B" if p_pass >= 0.45 else "#DC2626"

    st.markdown(
        f"""
        <div class="result-card" style="background:{colour}">
            <h1>{p_pass:.1%}</h1>
            <p>chance of passing &nbsp;·&nbsp; {risk_label(p_pass)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(p_pass)

    left, right = st.columns(2)
    left.metric("Exact (Variable Elimination)", f"{p_pass:.1%}")
    right.metric("Approximate (Likelihood Weighting)", f"{approx['Pass']:.1%}")

    st.bar_chart(
        pd.DataFrame({"Probability": [exact["Pass"], exact["Fail"]]}, index=["Pass", "Fail"])
    )
    st.info(f"Most likely exam preparation (MAP): **{prep['Prep']}** ({prep_prob:.1%})")

st.divider()

with st.expander("🔗 Bayesian network structure"):
    st.graphviz_chart(NETWORK_DOT)

with st.expander("📋 Learned probability table: Result given Performance"):
    rows = {f"Perf = {k[0]}": v for k, v in network.cpt["Result"].items()}
    st.dataframe(pd.DataFrame(rows).T.style.format("{:.3f}"))

with st.expander("🗂️ Training dataset (synthetic, first 10 of 1000 rows)"):
    st.dataframe(pd.DataFrame(dataset).head(10))
