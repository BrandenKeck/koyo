import pandas as pd
from dataclasses import dataclass, field

# General Use TOI -> Float function
def toi_to_minutes(toi: str) -> float:
        min, sec = toi.split(":")
        real_toi = float(min) + float(sec)/60
        return real_toi

@dataclass
class Team():
    id: str
    games: dict = field(default_factory=dict, repr=False)
    def add_game(self, game, homeaway):
        opp = "away" if homeaway=="home" else "home"
        gameid = game["id"]
        gf = game[f"{homeaway}Team"]["score"]
        ga = game[f"{opp}Team"]["score"]
        self.games[gameid] = {"GF": gf, "GA": ga}
    def roll_stats(self, win=10):
        if bool(self.games):
            dat = pd.DataFrame.from_dict(self.games, orient='index')
            dat[f'GF{win}'] = dat['GF'].rolling(win).mean()
            dat[f'GA{win}'] = dat['GA'].rolling(win).mean()
            self.games = dat.to_dict(orient='index')

@dataclass
class Skater():
    id: int
    name: str
    position: str
    games: dict = field(default_factory=dict, repr=False)
    def add_game(self, game, ga, ogf10, oga10):
        gameid = game["gameId"]
        points = game["points"]
        toi = toi_to_minutes(game["toi"])
        if toi > 0:
            self.games[gameid] = {
                "P": points, 
                "TOI": toi, 
                "O": points/(toi+oga10), 
                "D": ga/(toi+ogf10)
            }
    def roll_stats(self, win=10):
        if bool(self.games):
            dat = pd.DataFrame.from_dict(self.games, orient='index')
            dat[f'O{win}'] = dat['O'].rolling(win).mean()
            dat[f'D{win}'] = dat['D'].rolling(win).mean()
            self.games = dat.to_dict(orient='index')

@dataclass
class Goalie():
    id: int
    name: str
    games: dict = field(default_factory=dict, repr=False)
    def add_game(self, game, ogf10):
        gameid = game["gameId"]
        ga = game["goalsAgainst"]
        toi = toi_to_minutes(game["toi"])
        if toi > 0:
            self.games[gameid] = {
                "GA": ga, 
                "TOI": toi, 
                "G": ga/(toi+ogf10)
            }
    def roll_stats(self, win=10):
        if bool(self.games):
            dat = pd.DataFrame.from_dict(self.games, orient='index')
            dat[f'G{win}'] = dat['G'].rolling(win).mean()
            self.games = dat.to_dict(orient='index')

@dataclass
class KoyoData():
    game: int
    s: list = field(default_factory=list, repr=False)
    d: list = field(default_factory=list, repr=False)
    g: float


