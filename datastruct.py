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
    name: str
    roster: dict = field(default_factory=dict, repr=False)
    games: dict = field(default_factory=dict, repr=False)
    def add_player(self, id, name):
        self.roster[name] = id
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
                "O": (toi/60)*(points/oga10), 
                "D": (toi/60)*(ga/ogf10)
            }
    def roll_stats(self, win=10):
        if bool(self.games):
            dat = pd.DataFrame.from_dict(self.games, orient='index')
            dat[f'O{win}'] = dat['O'].rolling(win).mean()
            dat[f'D{win}'] = dat['D'].rolling(win).mean()
            self.games = dat.to_dict(orient='index')
    @property
    def last_game(self):
        gameids = list(self.games.keys())
        if len(gameids)>0:
            lastid = gameids[-1]
            return self.games[lastid]
        return {"O10": 0, "D10": 0}
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
                "G": (60/toi)*(ga/ogf10)
            }
    def roll_stats(self, win=10):
        if bool(self.games):
            dat = pd.DataFrame.from_dict(self.games, orient='index')
            dat[f'G{win}'] = dat['G'].rolling(win).mean()
            self.games = dat.to_dict(orient='index')
    @property
    def last_game(self):
        gameids = list(self.games.keys())
        if len(gameids)>0:
            lastid = gameids[-1]
            return self.games[lastid]
        return {"G10": 0}

@dataclass
class KoyoData():
    game: int
    y: int = None
    o: list = field(default_factory=list, repr=False)
    d: list = field(default_factory=list, repr=False)
    g: float = field(default=None, repr=False)

    @property
    def true_y(self):
        y = min(self.y, 7)
        if y==0: return [1, 0, 0, 0, 0, 0, 0, 0]
        return [i/y if i<=y else 0 for i in range(8)]
    
    @property
    def true_g(self):
        return [self.g for _ in range(4)]
