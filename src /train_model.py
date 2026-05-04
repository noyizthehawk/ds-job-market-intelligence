import pandas as pd
import ast
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ===================== LOAD DATA =====================
df = pd.read_csv("../data/jobs_clean.csv")

# Drop rows without salary
df = df.dropna(subset=["salary_min"])
print(f"Rows with salary data: {len(df)}")

# ===================== SENIORITY =====================
seniority_map = {
    "junior": 1,
    "mid-level": 2,
    "mid to senior level": 3,
    "senior": 4,
    "lead": 5,
    "experienced": 6
}

df["seniority"] = df["seniority"].str.lower().str.strip()
df["seniority_encoded"] = df["seniority"].map(seniority_map).fillna(2)

# ===================== SKILLS =====================
df["skills"] = df["skills"].apply(ast.literal_eval)
df["skills"] = df["skills"].apply(lambda x: [s.lower() for s in x])

mlb = MultiLabelBinarizer()
skills_encoded = pd.DataFrame(
    mlb.fit_transform(df["skills"]),
    columns=mlb.classes_,
    index=df.index
)

# reduce sparsity
skills_encoded = skills_encoded.loc[:, skills_encoded.sum() >= 10]

print(f"Skills kept: {skills_encoded.shape[1]}")

# ===================== CLEAN TEXT FEATURES =====================
df["company"] = df["company"].fillna("unknown").str.lower().str.strip()
df["location"] = df["location"].fillna("unknown").str.lower().str.strip()

# ===================== SKILL GROUP FEATURES =====================
def has_skill(skill_list, keywords):
    return int(any(k in skill_list for k in keywords))

df["has_ml"] = df["skills"].apply(lambda x: has_skill(x, ["tensorflow", "pytorch", "scikit-learn", "machine learning"]))
df["has_data"] = df["skills"].apply(lambda x: has_skill(x, ["pandas", "numpy", "data analysis"]))
df["has_cloud"] = df["skills"].apply(lambda x: has_skill(x, ["aws", "azure", "gcp"]))
df["has_backend"] = df["skills"].apply(lambda x: has_skill(x, ["node", "django", "flask", "spring"]))
df["has_sql"] = df["skills"].apply(lambda x: "sql" in x)

# interaction
df["seniority_x_ml"] = df["seniority_encoded"] * df["has_ml"]

# ===================== TRAIN TEST SPLIT =====================
y = df["salary_min"]

X_base = df[[
    "seniority_encoded",
    "company",
    "location",
    "has_ml",
    "has_data",
    "has_cloud",
    "has_backend",
    "has_sql",
    "seniority_x_ml"
]]

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_base, y, test_size=0.2, random_state=42
)

# ===================== TARGET ENCODING (NO LEAKAGE) =====================
company_salary_mean = pd.concat([X_train_df, y_train], axis=1).groupby("company")["salary_min"].mean()
location_salary_mean = pd.concat([X_train_df, y_train], axis=1).groupby("location")["salary_min"].mean()

# map encoding
X_train_df["company_encoded"] = X_train_df["company"].map(company_salary_mean)
X_test_df["company_encoded"] = X_test_df["company"].map(company_salary_mean)

X_train_df["location_encoded"] = X_train_df["location"].map(location_salary_mean)
X_test_df["location_encoded"] = X_test_df["location"].map(location_salary_mean)

# fill unseen categories
global_mean = y_train.mean()

X_train_df["company_encoded"] = X_train_df["company_encoded"].fillna(global_mean)
X_test_df["company_encoded"] = X_test_df["company_encoded"].fillna(global_mean)

X_train_df["location_encoded"] = X_train_df["location_encoded"].fillna(global_mean)
X_test_df["location_encoded"] = X_test_df["location_encoded"].fillna(global_mean)

# ===================== FINAL FEATURES =====================
X_train = pd.concat([
    X_train_df[[
        "seniority_encoded",
        "company_encoded",
        "location_encoded",
        "has_ml",
        "has_data",
        "has_cloud",
        "has_backend",
        "has_sql",
        "seniority_x_ml"
    ]],
    skills_encoded.loc[X_train_df.index]
], axis=1)

X_test = pd.concat([
    X_test_df[[
        "seniority_encoded",
        "company_encoded",
        "location_encoded",
        "has_ml",
        "has_data",
        "has_cloud",
        "has_backend",
        "has_sql",
        "seniority_x_ml"
    ]],
    skills_encoded.loc[X_test_df.index]
], axis=1)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# ===================== MODEL =====================
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

# ===================== EVALUATION =====================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: ${mae:,.0f}")
print(f"R2 Score: {r2:.2f}")


# save model
with open("../models/salary_predictor.pkl", "wb") as f:
    pickle.dump(model, f)

# save mlb so dashboard can encode skills the same way
with open("../models/mlb.pkl", "wb") as f:
    pickle.dump(mlb, f)

# save feature column names
with open("../models/feature_columns.pkl", "wb") as f:
    pickle.dump(X_train.columns.tolist(), f)

print("Model and artifacts saved!")

# see which features matter most
importances = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("Top 10 most important features:")
print(importances.head(10))