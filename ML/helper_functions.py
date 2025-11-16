import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

FOOTBALL_API_TOKEN = os.getenv("FOOTBALL_API_TOKEN")

def standings(url="http://api.football-data.org/v4/competitions/2021/standings", year=2022) -> None:

    headers = {
	"X-Auth-Token": FOOTBALL_API_TOKEN
    }

    params = {
        "season": year
    }

    response = requests.get(url=url, headers=headers, params=params)
    data = response.json()
    comp_name = data["competition"]["name"]
    standings = standings = [team["team"]["name"] for team in data["standings"][0]["table"]]

    return comp_name, standings