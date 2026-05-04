import anthropic
import pandas as pd
import os
import json
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

with open("../data/raw_postings.json") as f:
    data = json.load(f)
postings = data["results"]
results = []
for posting in postings:
    prompt = (f"You are a job posting analyzer. For this job posting: {posting['description']} "
              "extract the following details: job title, seniority, skills, salary min, salary max, location. "
              "Return ONLY a JSON object with no extra text before or after it.")

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    raw = response.content[0].text
    raw = raw.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    extracted = json.loads(raw)
    print(f"Extracted {len(results)}/1000: {extracted.get('job_title', 'unknown')}")
    extracted["salary_min"] = posting.get("salary_min")
    extracted["salary_max"] = posting.get("salary_max")
    extracted["company"] = posting.get("company", {}).get("display_name")
    results.append(extracted)

df = pd.DataFrame(results)
df.to_csv("../data/jobs_clean.csv", index=False)
print(f"Saved {len(df)} job postings to jobs_clean.csv")
print(f"Done! Extracted {len(results)} job postings")
print(df.head())
print(df.columns.tolist())



