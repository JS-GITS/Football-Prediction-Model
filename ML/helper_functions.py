import requests
import datetime
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
import os

load_dotenv()

FOOTBALL_API_TOKEN = os.getenv("FOOTBALL_API_TOKEN")

def standings(year=2025, comp="PL") -> None:

    headers = {
	"X-Auth-Token": FOOTBALL_API_TOKEN
    }

    params = {
        "season": year
    }

    url=f"http://api.football-data.org/v4/competitions/{comp}/standings"
    response = requests.get(url=url, headers=headers, params=params)
    data = response.json()
    comp_name = data["competition"]["name"]
    standings = standings = [team["team"]["name"] for team in data["standings"][0]["table"]]

    return comp_name, standings

def get_elo(team_name):
    date = datetime.date.today()
    date.strftime(f"%Y-%m-%d")
    url = f"http://api.clubelo.com/{date}"

    response = requests.get(url)

    df = pd.read_csv(StringIO(response.text))

    team_elo = float(df[df["Club"] == team_name]["Elo"].iloc[0])
    return team_elo