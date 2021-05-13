import pandas as pd

stats = ["GAME_NO","AVG_PTS","AVG_AST","AVG_OREB","AVG_DREB","cPTS","cAST","cOREB","cDREB","cFGM","cFG3M","cFGA","cTO","cFTA","cPTS_ALLOWED","OFF_EFF","DEF_EFF","EFG","ELO"]

def populate_team_stats(nba_teams):
    team_stats ={}
    print("Populating team stats...")
    for team in nba_teams:
        team_stats[team] = {
            "GAME_NO": 0,
            "AVG_PTS": 0,
            "AVG_AST": 0,
            "AVG_OREB": 0,
            "AVG_DREB": 0,
            "cPTS": 0,
            "cAST": 0,
            "cOREB": 0,
            "cDREB": 0,
            "cFGM": 0,
            "cFG3M": 0,
            "cFGA": 0,
            "cTO": 0,
            "cFTA": 0,
            "cPTS_ALLOWED": 0,
            "OFF_EFF": 0,
            "DEF_EFF": 0,
            "EFG": 0,
            "ELO": 1500
        }
    print("Successfully populated team stats.")
    return team_stats

def update_GAME_NO(team_stats, team_x, team_y):
    team_stats[team_x]['GAME_NO'] += 1
    team_stats[team_y]['GAME_NO'] += 1
    return team_stats[team_x]['GAME_NO'], team_stats[team_y]['GAME_NO']

def update_cPTS(team_stats, team_x, team_y, pts_x, pts_y):
    team_stats[team_x]['cPTS'] += pts_x
    team_stats[team_y]['cPTS'] += pts_y
    return team_stats[team_x]['cPTS'], team_stats[team_y]['cPTS']

def update_cAST(team_stats, team_x, team_y, ast_x, ast_y):
    team_stats[team_x]['cAST'] += ast_x
    team_stats[team_y]['cAST'] += ast_y
    return team_stats[team_x]['cAST'], team_stats[team_y]['cAST']

def update_cOREB(team_stats, team_x, team_y, oreb_x, oreb_y):
    team_stats[team_x]['cOREB'] += oreb_x
    team_stats[team_y]['cOREB'] += oreb_y
    return team_stats[team_x]['cOREB'], team_stats[team_y]['cOREB']

def update_cDREB(team_stats, team_x, team_y, dreb_x, dreb_y):
    team_stats[team_x]['cDREB'] += dreb_x
    team_stats[team_y]['cDREB'] += dreb_y
    return team_stats[team_x]['cDREB'], team_stats[team_y]['cDREB']

def update_cFGM(team_stats, team_x, team_y, fgm_x, fgm_y):
    team_stats[team_x]['cFGM'] += fgm_x
    team_stats[team_y]['cFGM'] += fgm_y
    return team_stats[team_x]['cFGM'], team_stats[team_y]['cFGM']

def update_cFG3M(team_stats, team_x, team_y, fg3m_x, fg3m_y):
    team_stats[team_x]['cFG3M'] += fg3m_x
    team_stats[team_y]['cFG3M'] += fg3m_y
    return team_stats[team_x]['cFG3M'], team_stats[team_y]['cFG3M']

def update_cFGA(team_stats, team_x, team_y, fga_x, fga_y):
    team_stats[team_x]['cFGA'] += fga_x
    team_stats[team_y]['cFGA'] += fga_y
    return team_stats[team_x]['cFGA'], team_stats[team_y]['cFGA']

def update_cTO(team_stats, team_x, team_y, to_x, to_y):
    team_stats[team_x]['cTO'] += to_x
    team_stats[team_y]['cTO'] += to_y
    return team_stats[team_x]['cTO'], team_stats[team_y]['cTO']

def update_cFTA(team_stats, team_x, team_y, fta_x, fta_y):
    team_stats[team_x]['cFTA'] += fta_x
    team_stats[team_y]['cFTA'] += fta_y
    return team_stats[team_x]['cFTA'], team_stats[team_y]['cFTA']

def update_cPTS_ALLOWED(team_stats, team_x, team_y, pts_y, pts_x):
    team_stats[team_x]['cPTS_ALLOWED'] += pts_y
    team_stats[team_y]['cPTS_ALLOWED'] += pts_x
    return team_stats[team_x]['cPTS_ALLOWED'], team_stats[team_y]['cPTS_ALLOWED']

def update_AVG_PTS(team_stats, team_x, team_y, GAME_NO_x, GAME_NO_y, cPTS_x, cPTS_y):
    dis = update_DIS_PTS(team_stats, team_x, team_y)
    team_stats[team_x]['AVG_PTS'] = round(cPTS_x / GAME_NO_x, 2)
    team_stats[team_y]['AVG_PTS'] = round(cPTS_y / GAME_NO_y, 2)
    return dis, team_stats[team_x]['AVG_PTS'], team_stats[team_y]['AVG_PTS']

def update_AVG_AST(team_stats, team_x, team_y, GAME_NO_x, GAME_NO_y, cAST_x, cAST_y):
    dis = update_DIS_AST(team_stats, team_x, team_y)
    team_stats[team_x]['AVG_AST'] = round(cAST_x / GAME_NO_x, 2)
    team_stats[team_y]['AVG_AST'] = round(cAST_y / GAME_NO_y, 2)
    return dis, team_stats[team_x]['AVG_AST'], team_stats[team_y]['AVG_AST']

def update_AVG_OREB(team_stats, team_x, team_y, GAME_NO_x, GAME_NO_y, cOREB_x, cOREB_y):
    dis = update_DIS_OREB(team_stats, team_x, team_y)
    team_stats[team_x]['AVG_OREB'] = round(cOREB_x/ GAME_NO_x, 2)
    team_stats[team_y]['AVG_OREB'] = round(cOREB_y/ GAME_NO_y, 2)
    return dis, team_stats[team_x]['AVG_OREB'], team_stats[team_y]['AVG_OREB']

def update_AVG_DREB(team_stats, team_x, team_y, GAME_NO_x, GAME_NO_y, cDREB_x, cDREB_y):
    dis = update_DIS_DREB(team_stats, team_x, team_y)
    team_stats[team_x]['AVG_DREB'] = round(cDREB_x / GAME_NO_x, 2)
    team_stats[team_y]['AVG_DREB'] = round(cDREB_y / GAME_NO_y, 2)
    return dis, team_stats[team_x]['AVG_DREB'], team_stats[team_y]['AVG_DREB']

def update_OFF_EFF(team_stats, team_x, team_y, cPTS_x, cPTS_y, cFGA_x, cFGA_y, cOREB_x, cOREB_y, cTO_x, cTO_y, cFTA_x, cFTA_y):
    dis = update_DIS_OFF_EFF(team_stats, team_x, team_y)
    team_stats[team_x]['OFF_EFF'] = round(cPTS_x / (cFGA_x - cOREB_x + cTO_x + (0.4 * cFTA_x)) * 100, 2)
    team_stats[team_y]['OFF_EFF'] = round(cPTS_y / (cFGA_y - cOREB_y + cTO_y + (0.4 * cFTA_y)) * 100, 2)
    return dis, team_stats[team_x]['OFF_EFF'], team_stats[team_y]['OFF_EFF']

def update_DEF_EFF(team_stats, team_x, team_y, cPTS_ALLOWED_x, cPTS_ALLOWED_y, cFGA_x, cFGA_y, cOREB_x, cOREB_y, cTO_x, cTO_y, cFTA_x, cFTA_y):
    dis = update_DIS_DEF_EFF(team_stats, team_x, team_y)
    team_stats[team_x]['DEF_EFF'] = round(cPTS_ALLOWED_x / (cFGA_x - cOREB_x + cTO_x + (0.4 * cFTA_x)) * 100, 2)
    team_stats[team_y]['DEF_EFF'] = round(cPTS_ALLOWED_y / (cFGA_y - cOREB_y + cTO_y + (0.4 * cFTA_y)) * 100, 2)
    return dis, team_stats[team_x]['DEF_EFF'], team_stats[team_y]['DEF_EFF']

def update_EFG(team_stats, team_x, team_y, cFGM_x, cFGM_y, cFG3M_x, cFG3M_y, cFGA_x, cFGA_y):
    team_stats[team_x]['EFG'] = round((cFGM_x + 0.5 * cFG3M_x) / cFGA_x, 2)
    team_stats[team_y]['EFG'] = round((cFGM_y + 0.5 * cFG3M_y) / cFGA_y, 2)
    return team_stats[team_x]['EFG'], team_stats[team_y]['EFG']

def update_ELO(team_stats, team_x, team_y, WL_x, K_FACTOR=20): 
    dis = update_DIS_ELO(team_stats, team_x, team_y)

    # K is constant value for multiplier, affects the sensitivity to recent games
    P_team = 1 / (1 + 10 ** ((team_stats[team_y]['ELO'] - team_stats[team_x]['ELO'])/400)) # probability of team winning
    if WL_x == 1:
        elo_change = round(K_FACTOR * (1 - P_team), 3) # formula for change in elo if team 1 wins
    else:
        elo_change = round(K_FACTOR * (0 - P_team), 3) # formula for change in elo if team 1 loses
    
    team_stats[team_x]['ELO'] += elo_change
    team_stats[team_y]['ELO'] -= elo_change

    return dis, team_stats[team_x]['ELO'], team_stats[team_y]['ELO']

def update_DIS_PTS(team_stats, team_x, team_y):
    dis = team_stats[team_x]['AVG_PTS'] - team_stats[team_y]['AVG_PTS']
    return dis

def update_DIS_AST(team_stats, team_x, team_y):
    dis = team_stats[team_x]['AVG_AST'] - team_stats[team_y]['AVG_AST']
    return dis

def update_DIS_OREB(team_stats, team_x, team_y):
    dis = team_stats[team_x]['AVG_OREB'] - team_stats[team_y]['AVG_OREB']
    return dis

def update_DIS_DREB(team_stats, team_x, team_y):
    dis = team_stats[team_x]['AVG_DREB'] - team_stats[team_y]['AVG_DREB']
    return dis

def update_DIS_OFF_EFF(team_stats, team_x, team_y):
    dis = team_stats[team_x]['OFF_EFF'] - team_stats[team_y]['OFF_EFF']
    return dis

def update_DIS_DEF_EFF(team_stats, team_x, team_y):
    dis = team_stats[team_x]['DEF_EFF'] - team_stats[team_y]['DEF_EFF']
    return dis

def update_DIS_ELO(team_stats, team_x, team_y):
    dis = team_stats[team_x]['ELO'] - team_stats[team_y]['ELO']
    return dis

def update_HOME_COURT(matchup):
    if '@' in matchup:
        return 0, 1
    return 1, 0