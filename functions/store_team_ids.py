import requests
from dotenv import load_dotenv
import json
import os

# Load environment variables
load_dotenv()
FOOTBALL_API_TOKEN = os.getenv("FOOTBALL_API_TOKEN")

# API endpoint for all competitions
uri = 'http://api.football-data.org/v4/competitions/'
headers = {'X-Auth-Token': FOOTBALL_API_TOKEN}

# Make the API request
response = requests.get(uri, headers=headers)
data = response.json()

# Build the dictionary: competition name -> code
competitions_dict = {}
for comp in data.get("competitions", []):
    name = comp.get("name")
    code = comp.get("code")
    if name and code:
        competitions_dict[name] = code

# Save the dictionary to JSON
with open("./data/competitions.json", "w") as fp:
    json.dump(competitions_dict, fp, indent=2)

print("Saved competitions dictionary to competitions.json")