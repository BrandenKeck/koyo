import pandas as pd
from dataclasses import dataclass, field

@dataclass
class Team():
    id: str
    games: dict = field(default_factory=dict, repr=False)
    def add_game(self, data, homeaway):
        opp = "away" if homeaway=="home" else "home"
        id = data["id"]
        gf = data[f"{homeaway}Team"]["score"]
        ga = data[f"{opp}Team"]["score"]
        self.games[id] = {"GF": gf, "GA": ga}
    def roll_stats(self, win=10):
        dat = pd.DataFrame.from_dict(self.games, orient='index')
        dat[f'GF{win}'] = dat['GF'].rolling(win).mean()
        dat[f'GA{win}'] = dat['GA'].rolling(win).mean()
        self.games = dat.to_dict(orient='index')

@dataclass
class Skater():
    id: str
    name: str
    position: str
    games: dict = field(default_factory=dict, repr=False)
    def add_game(self, game, ga, gf10, ogf10):
        gameid = game["gameId"]
        points = game["points"]
        toi = self.toi_to_minutes(game["toi"])
        self.games[id] = {"P": points, "TOI": toi, "GA": ga, "GF10": gf10, "OGF10": ogf10, "O": points/(toi+oga10), "D": ga/(toi+ogf10)}
    def toi_to_minutes(self, toi):
        min, sec = toi.split(":")
        real_toi = float(min) + float(sec)/60
        return real_toi

@dataclass
class Goalie():
    id: str
    name: str
    games: dict = field(default_factory=dict, repr=False)
    def add_game(self, game):
        toi = self.toi_to_minutes(game["toi"])
        self.games[id] = {"GA": game["goalsAgainst"], "TOI": toi}
    def toi_to_minutes(self, toi):
        min, sec = toi.split(":")
        real_toi = float(min) + float(sec)/60
        return real_toi
