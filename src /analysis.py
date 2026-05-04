import pandas as pd
import json
import ast
from collections import Counter

# load csv
df = pd.read_csv("../data/jobs_clean.csv") # you dont load svcs like json
df["seniority"] = df["seniority"].str.lower().str.strip()
df["skills"] = df["skills"].apply(ast.literal_eval)
df["skills"] = df["skills"].apply(lambda x: [s.lower() for s in x])
skills = df["skills"]
print(skills)

all_skills = [skill for skills in df["skills"] for skill in skills]
# count how many times each skill appears
skill_counts = Counter(all_skills)
# show top 10
print(skill_counts.most_common(10))

# average salary
avg_salary = df["salary_min"].mean()
print(f"Average minimum salary: ${avg_salary:,.0f}")

# salary by seniority
salary_by_seniority = df.groupby("seniority")["salary_min"].mean()
print("\nAverage salary by seniority:")
print(salary_by_seniority)

print("\nTop hiring companies:")
print(df["company"].value_counts().head(10))
