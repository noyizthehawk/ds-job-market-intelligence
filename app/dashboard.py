import streamlit as st
import pandas as pd
import pickle
import ast
from collections import Counter

# page config
st.set_page_config(
    page_title="Job Market Intelligence",
    page_icon="📊",
    layout="wide"
)

# title
st.title(" Data Science Job Market Intelligence")
st.markdown("Real insights from 1,000+ live job postings")

# load data
df = pd.read_csv("data/jobs_clean.csv")
df["skills"] = df["skills"].apply(ast.literal_eval)
df["skills"] = df["skills"].apply(lambda x: [s.lower() for s in x])

# load model
with open("models/salary_predictor.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# overview stats
col1, col2, col3 = st.columns(3)

all_skills = [skill for skills in df["skills"] for skill in skills]
skill_counts = Counter(all_skills)
top_skill = skill_counts.most_common(1)[0][0]

col1.metric("Job Postings Analyzed", f"{len(df):,}")
col2.metric("Most In-Demand Skill", top_skill.title())
col3.metric("Avg Minimum Salary", f"${df['salary_min'].mean():,.0f}")

# skill demand chart
st.subheader("Top 10 Most In-Demand Skills")

top_10 = skill_counts.most_common(10)
skills_df = pd.DataFrame(top_10, columns=["skill", "count"])
skills_df["skill"] = skills_df["skill"].str.title()

st.bar_chart(skills_df.set_index("skill"))

st.subheader("salary predictor")
st.markdown("Select your profile to get a salary estimate")

col1, col2 = st.columns(2)

top_companies = df["company"].value_counts().head(20).index.tolist()

with col1:
    seniority = st.selectbox("Seniority Level", [
        "junior", "mid-level", "senior", "lead", "experienced"
    ])
    company = st.selectbox("Company", ["unknown"] + top_companies)
with col2:
    selected_skills = st.multiselect("Your Skills", [
        "python", "machine learning", "sql", "deep learning",
        "tensorflow", "pytorch", "aws", "azure", "data analysis",
        "statistics", "nlp", "computer vision"
    ])

if st.button("Predict Salary"):
    st.write("Predicting...")
    # converting seniority to something readable for the model
    seniority_map = {
        "junior": 1, "mid-level": 2, "mid to senior level": 3,
        "senior": 4, "lead": 5, "experienced": 6
    }
    # build input row with all zeros
    input_data = pd.DataFrame([{col: 0 for col in feature_columns}])

    # fill in seniority
    # encode company using training mean
    company_mean = df[df["company"].str.lower() == company.lower()]["salary_min"].mean()
    if pd.isna(company_mean):
        company_mean = df["salary_min"].mean()
    input_data["company_encoded"] = company_mean
    input_data["seniority_encoded"] = seniority_map.get(seniority, 2)
    for skill in selected_skills:
        if skill in input_data.columns:
            input_data[skill] = 1

        # make prediction
    predicted = model.predict(input_data)[0]

    st.success(f"Estimated Salary: ${predicted:,.0f} per year")
st.subheader("Top Hiring Companies")
top_companies_chart = df["company"].value_counts().head(10)
st.bar_chart(top_companies_chart)


