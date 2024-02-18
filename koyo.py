# Import Libraries
import json, requests
import pandas as pd
import pprint as pp
from datetime import date
from datastruct import Team, Skater, Goalie

# Load Components
with open('teams.json', 'r') as f:
    teamdict = json.load(f)
teams = {t:Team(t) for t in teamdict.values()}
skaters = {}
goalies = {}

# TOI Function
def toi_to_minutes(toi):
    min, sec = toi.split(":")
    real_toi = float(min) + float(sec)/60
    return real_toi

# Construct Rosters
for t in teams:
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

# Build Games - Team Statistics
curr = "2023-10-10"
end = "2023-12-31"
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
for t in teams:
    teams[t].roll_stats()


# Build Games - Player Statistics
for skid in skaters:
    print(skid)
    resp = requests.get(f"https://api-web.nhle.com/v1/player/{skid}/game-log/20232024/2").json()
    for game in resp["gameLog"]:
        gameid = game["gameId"]
        gf10 = teams[game["teamAbbrev"]].games[gameid]["GF10"]
        ga10 = teams[game["opponentAbbrev"]].games[gameid]["GA10"]
        skaters[skid]
        
        toi_to_minutes(game["toi"])
        game["goals"]
        game["assists"]
        game["points"]
        
        game["opponentAbbrev"]


pp.pprint(xx)


teams["PIT"].games


pp.pprint(game)

scores = []


xx = requests.get("https://api-web.nhle.com/v1/club-stats/TOR/20232024/2").json()
pp.pprint(xx)

















kk = koyo()
for t in teams.values():
    koyo.add_team(t)

roster = requests.get(https://api-web.nhle.com/v1/roster/{t}/current).json()
pp.pprint(roster['forwards'])
pp.pprint(roster['defensemen'])
pp.pprint(roster['goalies'])

sch = requests.get(fhttps://api-web.nhle.com/v1/schedule/{date.today()}).json()
games = sch["gameWeek"][0]["games"]
for game in games:
    id = game["id"]
    home = game["homeTeam"]["abbrev"]
    away = game["awayTeam"]["abbrev"]
    print(f'ID: {id} |Home: {home} | Away: {away}')
len(resp['defensemen'])
len(resp['forwards'])
len(resp['goalies'])

last5 = []
for game in resp['gameLog'][:5]:
    last5.append({
        'goals': game['goals'],
        'assists': game['assists'],
        'shots': game['assists'],
        'toi': game['toi']
    })
pp.pprint(last5)

resp = requests.get(fhttps://api-web.nhle.com/v1/player/8477967/game-log/20232024/2).json()
last5 = []
for game in resp['gameLog'][:5]:
    last5.append({
        'savePctg': game['savePctg'],
        'shotsAgainst': game['shotsAgainst'],
        'goalsAgainst': game['goalsAgainst'],
        'toi': game['toi']
    })
pp.pprint(last5)

# Parters:
    # 9: DraftKings
    # 7: FanDuel
pp.pprint(games)