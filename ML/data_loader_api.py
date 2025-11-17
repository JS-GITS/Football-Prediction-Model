import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

def load_data(url="https://free-api-live-football-data.p.rapidapi.com/matches?team_id=65") -> None:

    headers = {
	"x-rapidapi-key": RAPIDAPI_KEY,
	"x-rapidapi-host": "free-api-live-football-data.p.rapidapi.com"
}

    params = {
        "eventid": "1988822"
    }

    response = requests.get(url=url, headers=headers)
    data = response.json()

    with open("./data/fixtures.json","w") as f:
        json.dump(data, f, indent=4)