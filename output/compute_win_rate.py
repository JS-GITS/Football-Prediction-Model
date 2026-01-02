from ML.helper_functions import get_elo

def compute_win_rate(encoders, Division, HomeTeam, AwayTeam):
    encoder_types = ["Division", "HomeTeam", "AwayTeam"]
    Division_enc = encoders[encoder_types=="Division"]
    HomeTeam_enc = encoders[encoder_types=="HomeTeam"]
    AwayTeam_enc = encoders[encoder_types=="AwayTeam"]
    HomeElo = get_elo(HomeTeam)
    AwayElo = get_elo(AwayTeam)

