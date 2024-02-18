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
    def add_game(self):
        pass

@dataclass
class Goalie():
    id: str
    name: str
    games: dict = field(default_factory=dict, repr=False)
    def add_game(self):
        pass
