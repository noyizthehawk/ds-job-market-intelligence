import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

app_id = os.getenv("ADZUNA_APP_ID")
app_key = os.getenv("ADZUNA_APP_KEY")

all_postings = []

for page in range(1, 21):
    url = f"https://api.adzuna.com/v1/api/jobs/us/search/{page}"
    response = requests.get(url, params={
        "app_id": app_id,
        "app_key": app_key,
        "what": "data scientist",
        "results_per_page": 50
    })
    data = response.json()
    postings = data["results"]
    all_postings.extend(postings)
    print(f"Page {page} collected: {len(postings)} postings")

print(f"Total postings collected: {len(all_postings)}")

with open("../data/raw_postings.json", "w") as f:
    json.dump({"results": all_postings}, f)

