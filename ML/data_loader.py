import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def data_loader():
    """
    This function reads the file "Matches.csv" from the data folder and modifies the data. It first maps the FT(Full Time) results to numbers 0, 1, 2 (class labels).
    'H (Home)' maps to 0, 'D (Draw)' maps to 1, 'A (Away)' maps to 2.
    For example, the result between Bayern Munich vs Galatasaray is 'H', indicating that Bayern Munich won, this maps the result to 0.
    All of the mapped FT are added as a new key in the DataFrame and all the NaN values are dropped.
    Afterwards, the data are sorted by MatchDate and the data are split into training and validation sets with a 4:1 ratio.
    Since the Division and Teams are text-based, all of them are encoded using LabelEncoder from the scikit-learn library.
    The data are processed to extract all features, producing the training and validation DataFrames along with the encoders.

    Parameters:
    None

    Returns:
    train_df -> pandas.DataFrame
    valid_df -> pandas.DataFrame
    encoders -> dict[str, sklearn.preprocessing.LabelEncoder]
    """
    df = pd.read_csv("./data/Matches.csv", dtype={"MatchTime":"string"})

    target_map = {"H": 0, "D": 1, "A": 2}
    df["Target"] = df["FTResult"].map(target_map)

    df = df.dropna(subset=["Target"])

    # Sort by date (time-based split)
    if "MatchDate" in df.columns:
        df["MatchDate"] = pd.to_datetime(df["MatchDate"])
        df = df.sort_values("MatchDate").reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    valid_df = df.iloc[split_idx:].copy()

    # Encoding the Division and Home/Away Teams
    col_enc = ["Division", "HomeTeam", "AwayTeam"]
    encoders = {}

    for col in col_enc:
        le = LabelEncoder()

        # Fit on train only
        train_vals = train_df[col].astype(str)
        le.fit(train_vals)

        # Encode train
        train_df[col + "_enc"] = le.transform(train_vals)

        # Encode valid with unknown handling
        valid_vals = valid_df[col].astype(str)

        known_mask = valid_vals.isin(le.classes_)
        valid_enc = np.full(len(valid_vals), -1, dtype=int)

        # only transform known labels
        valid_enc[known_mask.to_numpy()] = le.transform(valid_vals[known_mask])

        valid_df[col + "_enc"] = valid_enc

        encoders[col] = le

    # Placing flags for NaN valued Handicaps
    handi_cols = ["HandiSize", "HandiHome", "HandiAway"]
    for col in handi_cols:
        train_df[f"{col}_missing"] = train_df[col].isna().astype(int)
        valid_df[f"{col}_missing"] = valid_df[col].isna().astype(int)
    
    # Replace the NaN values in the handicap column with median values
    # I used median here so that there are no skips without major repercussions
    median_handis = train_df[handi_cols].median()
    train_df[handi_cols] = train_df[handi_cols].fillna(median_handis)
    valid_df[handi_cols] = valid_df[handi_cols].fillna(median_handis)

    # Change NaNs in the forms with 0
    form_cols = ["Form3Home", "Form5Home", "Form3Away", "Form5Away"]
    train_df[form_cols] = train_df[form_cols].fillna(0)
    valid_df[form_cols] = valid_df[form_cols].fillna(0)

    # Placing flags for the NaN valued odds
    core_odds = ["OddHome", "OddDraw", "OddAway"]
    for col in core_odds:
        train_df[f"{col}_missing"] = train_df[col].isna().astype(int)
        valid_df[f"{col}_missing"] = valid_df[col].isna().astype(int)

    # Replace the NaN values with median values
    median_odds = train_df[core_odds].median()
    train_df[core_odds] = train_df[core_odds].fillna(median_odds)
    valid_df[core_odds] = valid_df[core_odds].fillna(median_odds)

    for col in core_odds:
        train_df[col] = train_df[col].clip(lower=1e-6)
        valid_df[col] = valid_df[col].clip(lower=1e-6)

    # Elo + form differences
    dfs = [train_df, valid_df]
    prob_cols = [
            "ProbHome_off", "ProbDraw_off", "ProbAway_off",
            "ProbHome", "ProbDraw", "ProbAway", "BookerMargin",
        ]
    for each_df in dfs:
        each_df ["EloDiff"] = each_df["HomeElo"] - each_df["AwayElo"]
        each_df["EloSum"] = each_df["HomeElo"] + each_df["AwayElo"]
        each_df["Form3Diff"] = each_df["Form3Home"] - each_df["Form3Away"]
        each_df["Form5Diff"] = each_df["Form5Home"] - each_df["Form5Away"]
        each_df["EloFormRatio"] = (each_df["EloDiff"] + each_df["Form5Diff"]) / 2

        # Probability of both teams and their normalized values
        each_df["ProbHome_off"] = 1.0/each_df["OddHome"]
        each_df["ProbDraw_off"] = 1.0/each_df["OddDraw"]
        each_df["ProbAway_off"] = 1.0/each_df["OddAway"]
        prob_sum = each_df["ProbHome_off"] + each_df["ProbDraw_off"] + each_df["ProbAway_off"]
        each_df["ProbHome"] = each_df["ProbHome_off"]/prob_sum
        each_df["ProbDraw"] = each_df["ProbDraw_off"]/prob_sum
        each_df["ProbAway"] = each_df["ProbAway_off"]/prob_sum

        each_df["ProbDiff"] = each_df["ProbHome"] - each_df["ProbAway"]
        
        each_df["BookerMargin"] = prob_sum - 1.0

        each_df[prob_cols] = each_df[prob_cols].replace([np.inf, -np.inf], np.nan)
        each_df[prob_cols] = each_df[prob_cols].fillna(0.0)

    return train_df, valid_df, encoders