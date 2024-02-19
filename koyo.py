# Import Libraries
import json, math, requests
import numpy as np
import pandas as pd
import pprint as pp
from datetime import date
from difflib import get_close_matches
from datastruct import Team, Skater, Goalie, KoyoData, toi_to_minutes

import os
os.environ["CURL_CA_BUNDLE"]=""

# Load Components
with open('teams.json', 'r') as f:
    teamdict = json.load(f)
teams = {t:Team(t) for t in teamdict.values()}
skaters = {}
goalies = {}
data = {}

# 
def skater_lookup(name):
    invdict = {skaters[s].name:skaters[s].id for s in skaters}
    invlist = list(invdict.keys())
    skname = get_close_matches(name, invlist, n=1)[0]
    skid = invdict[skname]
    return skid

def primary_goalie(goalies):
    tois = len(goalies)*[0]
    for i, g in enumerate(goalies): tois[i] = toi_to_minutes(g['toi'])
    maxg = np.argmax(tois)
    return goalies[maxg]["playerId"]

# Build Games - Team Statistics
curr = "2023-09-01"
end = "2024-01-31"
while True:
    resp = requests.get(f"https://api-web.nhle.com/v1/score/{curr}").json()
    for game in resp["games"]:
        if game['gameType']!=2: continue
        for homeaway in ["home", "away"]:
            teams[game[f"{homeaway}Team"]["abbrev"]].add_game(game, homeaway)
    curr = resp["nextDate"]
    print(curr)
    if curr == end:
        break

# Construct Rosters
for t in teams:
    teams[t].roll_stats()
    resp = requests.get(f"https://api-web.nhle.com/v1/roster/{t}/current").json()
    for sk8r in resp["forwards"]+resp["defensemen"]:
        sid = sk8r["id"]
        name = f'{sk8r["firstName"]["default"]} {sk8r["lastName"]["default"]}'
        pos = sk8r["positionCode"]
        skaters[sid] = Skater(sid, name, pos)
    for g0ly in resp["goalies"]:
        gid = g0ly["id"]
        name = f'{g0ly["firstName"]["default"]} {g0ly["lastName"]["default"]}'
        goalies[gid] = Goalie(gid, name)

# Build Games - Player Statistics
for szn in ["20232024"]:#["20212022", "20222023", "20232024"]:
    for skid in skaters:
        resp = requests.get(f"https://api-web.nhle.com/v1/player/{skid}/game-log/{szn}/2").json()
        for game in resp["gameLog"]:
            gameid = game["gameId"]
            teamid = game["teamAbbrev"]
            oppid = game["opponentAbbrev"]
            if gameid in teams[oppid].games and gameid in teams[oppid].games:
                ga = teams[teamid].games[gameid]["GA"]
                ogf10 = teams[oppid].games[gameid]["GF10"]
                oga10 = teams[oppid].games[gameid]["GA10"]
                if not math.isnan(ogf10) and not math.isnan(oga10):
                    skaters[skid].add_game(game, ga, ogf10, oga10)
        skaters[skid].roll_stats()
        
# Build Games - Goalie Statistics
for szn in ["20232024"]:#["20212022", "20222023", "20232024"]:
    for gid in goalies:
        resp = requests.get(f"https://api-web.nhle.com/v1/player/{gid}/game-log/{szn}/2").json()
        for game in resp["gameLog"]:
            gameid = game["gameId"]
            oppid = game["opponentAbbrev"]
            if gameid in teams[oppid].games:
                ogf10 = teams[oppid].games[gameid]["GF10"]
                if not math.isnan(ogf10):
                    goalies[gid].add_game(game, ogf10)
        goalies[gid].roll_stats()

# Construct a list of games in the system
games = []
for t in teams:
    games = games + list(teams[t].games.keys())
games = list(set(games))

# Construct Dataset
for game in games:
    sk8rs, g0lys = {}, {}
    resp = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game}/boxscore").json()
    for homeaway in ["home", "away"]:
        sk8rs[homeaway] = resp["boxscore"]["playerByGameStats"][f"{homeaway}Team"]["forwards"] 
        sk8rs[homeaway] = sk8rs[homeaway] + resp["boxscore"]["playerByGameStats"][f"{homeaway}Team"]["defense"]
        sk8rs[homeaway] = [sk["playerId"] for sk in sk8rs[homeaway]]         
        g0lys[homeaway] = primary_goalie(resp["boxscore"]["playerByGameStats"][f"{homeaway}Team"]["goalies"])
    for homeaway in ["home", "away"]:
        awayhome = "away" if homeaway=="home" else "home"
        y = resp[f"{homeaway}Team"]["score"]
        try:
            o = [skaters[sk].games[game]["O10"] for sk in sk8rs[homeaway]]
            if any(each!=each for each in o): continue
            d = [skaters[sk].games[game]["D10"] for sk in sk8rs[awayhome]]
            if any(each!=each for each in d): continue
            g = goalies[g0lys[awayhome]].games[game]["G10"]
            if math.isnan(g): continue
        except: continue
        data[f"{game}_{homeaway}"] = KoyoData(game, y, o, d, g)

# Construct data vectors
O = [data[dat].o for dat in data]
D = [data[dat].d for dat in data]
G = [[data[dat].g for _ in range(4)] for dat in data]
Y = [data[dat].true_y for dat in data]

def true_y(y):
    y = min(y, 7)
    if y==0: return [1, 0, 0, 0, 0, 0, 0, 0]
    return [i/y if i<=y else 0 for i in range(8)]
Y = [true_y(data[dat].y) for dat in data]

# Train model
from net import KoyoModel
mod = KoyoModel(bsize=1, epochs=400)
mod.fit(O, D, G, Y)
mod.save("./model/", "model.h5")

# Check model
mod.predict(O[0:2], D[0:2], G[0:2])
Y [0:2]

# Get New Games
sch = requests.get(f"https://api-web.nhle.com/v1/schedule/{date.today()}").json()
sch = requests.get(f"https://api-web.nhle.com/v1/schedule/2024-02-19").json()

resp = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{2023020866}/boxscore").json()
xx = resp["boxscore"]["playerByGameStats"][f"homeTeam"]["forwards"] 
xx = xx + resp["boxscore"]["playerByGameStats"][f"homeTeam"]["defense"]
xx = [sk["playerId"] for sk in xx] 

pp.pprint(sch["gameWeek"][0]["games"][0])
len(sch["gameWeek"][0]["games"])
sch.keys()

# Parters:
    # 9: DraftKings
    # 7: FanDuel
# sch = requests.get(f"https://api-web.nhle.com/v1/schedule/{date.today()}").json()
# games = sch["gameWeek"][0]["games"]
# for game in games:
#     id = game["id"]
#     home = game["homeTeam"]["abbrev"]
#     away = game["awayTeam"]["abbrev"]
#     print(f'ID: {id} |Home: {home} | Away: {away}')
# pp.pprint(games)