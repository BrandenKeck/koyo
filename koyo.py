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
teams = {v:Team(v,k) for k,v in teamdict.items()}
skaters = {}
goalies = {}
data = {}

# 
def player_lookup(name, team):
    invlist = list(team.roster.keys())
    pname = get_close_matches(name.title(), invlist, n=1)[0]
    pid = team.roster[pname]
    return pid

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
        teams[t].add_player(sid, name)
    for g0ly in resp["goalies"]:
        gid = g0ly["id"]
        name = f'{g0ly["firstName"]["default"]} {g0ly["lastName"]["default"]}'
        goalies[gid] = Goalie(gid, name)
        teams[t].add_player(gid, name)

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

# Just Pickle Everything for Now
import pickle
with open('pickles/teams.pickle', 'wb') as f: 
    pickle.dump(teams, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('pickles/skaters.pickle', 'wb') as f: 
    pickle.dump(skaters, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('pickles/goalies.pickle', 'wb') as f: 
    pickle.dump(goalies, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('pickles/data.pickle', 'wb') as f: 
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

###
# CHECKPOINT
###

# Unpickle for the next part of testing
with open('pickles/teams.pickle', 'rb') as f: teams = pickle.load(f)
with open('pickles/skaters.pickle', 'rb') as f: skaters = pickle.load(f)
with open('pickles/goalies.pickle', 'rb') as f: goalies = pickle.load(f)
with open('pickles/data.pickle', 'rb') as f: data = pickle.load(f)

# Construct data vectors
O = [data[dat].o for dat in data]
D = [data[dat].d for dat in data]
G = [[data[dat].g for _ in range(4)] for dat in data]
Y = [data[dat].true_y for dat in data]

# Train model
from net import KoyoModel
mod = KoyoModel(bsize=1, epochs=400)
mod.fit(O, D, G, Y)
mod.save("./model/", "model.h5")

# Check model
mod.predict(O[0:2], D[0:2], G[0:2])
Y[0:2]

# Driver Imports
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.core.driver_cache import DriverCacheManager

# Driver function
def get_driver():
        # Driver options
        cache_manager=DriverCacheManager("./driver")
        driver = ChromeDriverManager(cache_manager=cache_manager).install()
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--log-level=3")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=options)
        return driver

# Fetch Roster
def get_roster(team, driver):

    # Clean up the team name
    team = team.replace("é", "e").replace(".", "").replace(" ", "-").lower()
    driver.get(f"https://www.dailyfaceoff.com/teams/{team}/line-combinations/")

    # Get Skaters
    f = []
    f_paths = [
        [1, "div/", 4, 1], [1, "div/", 4, 2], [1, "div/", 4, 3],
        [1, "div/", 5, 1], [1, "div/", 5, 2], [1, "div/", 5, 3],
        [1, "div/", 6, 1], [1, "div/", 6, 2], [1, "div/", 6, 3],
        [1, "div/", 7, 1], [1, "div/", 7, 2], [1, "div/", 7, 3],
    ]
    for path in f_paths:
        try: sk8r = driver.find_element("xpath", f"//section[@id='line_combos']/div[{path[0]}]/{path[1]}div[{path[2]}]/div[{path[3]}]/div/div[2]/a/span").text
        except: sk8r = "Not Found"
        f.append(sk8r)

    # Get Defensemen
    d = []
    d_paths = [
        [2, "", 2, 1], [2, "", 2, 2],
        [2, "", 3, 1], [2, "", 3, 2],
        [2, "", 4, 1], [2, "", 4, 2],
    ]
    for path in d_paths:
        try: sk8r = driver.find_element("xpath", 
                f"//section[@id='line_combos']/div[{path[0]}]/{path[1]}div[{path[2]}]/div[{path[3]}]/div/div[2]/a/span").text
        except: sk8r = "Not Found"
        d.append(sk8r)

    # Get Goaltenders
    g = []
    for num in [1,2]:
        try: g0ly = driver.find_element("xpath", 
                f"//section[@id='line_combos']/div[9]/div[2]/div[{num}]/div/div[2]/a/span").text
        except: g0ly = "Not Found"
        g.append(g0ly)
    
    # Return info
    return {"F": f, "D": d, "G": g}

def calc_goals(home_goals, away_goals):
    probs = np.zeros(len(home_goals)+len(away_goals)-2)
    for i in np.arange(len(home_goals)-1):
        for j in np.arange(len(away_goals)-1):
            if i==j:
                probs[i+j+1] = probs[i+j+1] + home_goals[i]*away_goals[j]
            else:
                probs[i+j] = probs[i+j] + home_goals[i]*away_goals[j]
    return probs

# Construct an average goalie
gave = [g.last_game["G10"] for i, g in goalies.items() 
            if not math.isnan(g.last_game["G10"])
            and not g.last_game["G10"]==0]
gave = np.mean(gave)

# Get New Games
driver = get_driver()
gsk, ggo = {}, {}
gamedatas = []
sch = requests.get(f"https://api-web.nhle.com/v1/schedule/{date.today()}").json()
for gg in sch["gameWeek"][0]["games"]:
    gdat = {"home":{}, "away":{}}
    gameid = gg['id']
    homeabb = gg["homeTeam"]["abbrev"]
    awayabb = gg["awayTeam"]["abbrev"]
    gdat["home"]["name"] = teams[homeabb].name
    gdat["away"]["name"] = teams[awayabb].name
    gdat["home"]["roster"] = get_roster(gdat["home"]["name"], driver)
    gdat["away"]["roster"] = get_roster(gdat["away"]["name"], driver)
    gsk["home"] = [player_lookup(sk, teams[homeabb]) for sk in gdat["home"]["roster"]["F"]+gdat["home"]["roster"]["D"]]
    gsk["away"] = [player_lookup(sk, teams[awayabb]) for sk in gdat["away"]["roster"]["F"]+gdat["away"]["roster"]["D"]]
    ggo["home"] = player_lookup(gdat["home"]["roster"]["G"][0], teams[homeabb])
    ggo["away"] = player_lookup(gdat["away"]["roster"]["G"][0], teams[awayabb])
    for homeaway in ["home", "away"]:
        awayhome = "away" if homeaway=="home" else "home"
        o = [skaters[sk].last_game["O10"] for sk in gsk[homeaway]]
        if any(each!=each for each in o): o = np.nan_to_num(o).tolist()
        d = [skaters[sk].last_game["D10"] for sk in gsk[awayhome]]
        if any(each!=each for each in d): d = np.nan_to_num(d).tolist()
        g = goalies[ggo[awayhome]].last_game["G10"]
        if math.isnan(g): g = gave
        gdat[f"{homeaway}_goals"] = mod.predict([o], [d], [4*[g]]).tolist()[0]
    gdat["total_goals"] = calc_goals(gdat["home_goals"], gdat["away_goals"])
    gamedatas.append(gdat)

from reporting import KoyoReport
krep = KoyoReport()
krep.generate(gamedatas)

