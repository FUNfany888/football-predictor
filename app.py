import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import io
import subprocess
import json
import hashlib
import logging
from datetime import timedelta, datetime
from scipy.stats import poisson
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ------------------------- 日志配置 -------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------- 全局配置 -------------------------
SEASONS = ['2223', '2324', '2425', '2526']
LEAGUES = ['E0', 'E1', 'E2', 'SP1', 'FR1', 'D1', 'I1', 'EC', 'EL', 'F93', 'F94', 'F96']
HISTORY_PATH = 'history.parquet'
SCRIPT_PATH = r"C:\Users\曹亚楠\Desktop\worldcup-betting-analyst-skill\scripts\fetch_sporttery.py"
UPDATE_DAYS = 7

_DF_HIST = None
_DF_HASH = None

# ------------------------- 球队映射表（完整） -------------------------
TEAM_NAME_MAP = {
    'Arsenal': '阿森纳', 'Aston Villa': '阿斯顿维拉',
    'Bournemouth': '伯恩茅斯', 'Brentford': '布伦特福德',
    'Brighton': '布莱顿', 'Burnley': '伯恩利',
    'Chelsea': '切尔西', 'Crystal Palace': '水晶宫',
    'Everton': '埃弗顿', 'Fulham': '富勒姆',
    'Leeds United': '利兹联', 'Leicester City': '莱斯特城',
    'Liverpool': '利物浦', 'Manchester City': '曼彻斯特城',
    'Manchester United': '曼彻斯特联', 'Newcastle United': '纽卡斯尔联',
    'Nottingham Forest': '诺丁汉森林', 'Southampton': '南安普顿',
    'Tottenham': '托特纳姆热刺', 'West Ham United': '西汉姆联',
    'Wolves': '狼队', 'Wolverhampton Wanderers': '伍尔弗汉普顿',
    'Birmingham City': '伯明翰', 'Blackburn Rovers': '布莱克本',
    'Blackpool': '布莱克浦', 'Bristol City': '布里斯托尔城',
    'Cardiff City': '加的夫城', 'Coventry City': '考文垂',
    'Derby County': '德比郡', 'Huddersfield Town': '哈德斯菲尔德',
    'Hull City': '赫尔城', 'Ipswich Town': '伊普斯维奇',
    'Luton Town': '卢顿', 'Middlesbrough': '米德尔斯堡',
    'Millwall': '米尔沃尔', 'Norwich City': '诺维奇',
    'Preston North End': '普雷斯顿', 'Queens Park Rangers': '女王公园巡游者',
    'Reading': '雷丁', 'Rotherham United': '罗瑟勒姆',
    'Sheffield United': '谢菲尔德联', 'Sheffield Wednesday': '谢周三',
    'Stoke City': '斯托克城', 'Sunderland': '桑德兰',
    'Swansea City': '斯旺西', 'Watford': '沃特福德',
    'West Bromwich Albion': '西布罗姆维奇', 'Wigan Athletic': '维冈竞技',
    'Accrington Stanley': '阿克灵顿', 'Barnsley': '巴恩斯利',
    'Bolton Wanderers': '博尔顿', 'Burton Albion': '伯顿',
    'Cambridge United': '剑桥联', 'Charlton Athletic': '查尔顿',
    'Charlton': '查尔顿', 'Cheltenham Town': '切尔滕汉姆',
    'Crewe Alexandra': '克鲁', 'Doncaster Rovers': '唐卡斯特',
    'Exeter City': '埃克塞特城', 'Fleetwood Town': '弗利特伍德',
    'Forest Green': '格林森林流浪者', 'Gillingham': '吉林汉姆',
    'Leyton Orient': '莱顿东方', 'Lincoln City': '林肯城',
    'Mansfield': '曼斯菲尔德', 'Milton Keynes Dons': '米尔顿凯恩斯',
    'Morecambe': '莫克姆', 'Northampton Town': '北安普顿',
    'Oxford United': '牛津联', 'Peterborough United': '彼得堡联',
    'Plymouth Argyle': '普利茅斯', 'Port Vale': '维尔港',
    'Portsmouth': '朴茨茅斯', 'Shrewsbury Town': '什鲁斯伯里',
    'Stevenage': '斯蒂夫尼奇', 'Stockport': '斯托克波特',
    'Wycombe Wanderers': '韦康比流浪者',
    'Alaves': '阿拉维斯', 'Almeria': '阿尔梅里亚',
    'Athletic Bilbao': '毕尔巴鄂竞技', 'Atletico Madrid': '马德里竞技',
    'Barcelona': '巴塞罗那', 'Real Betis': '皇家贝蒂斯',
    'Cadiz': '加的斯', 'Celta Vigo': '塞尔塔',
    'Elche': '埃尔切', 'Espanyol': '西班牙人',
    'Getafe': '赫塔费', 'Girona': '赫罗纳',
    'Granada': '格拉纳达', 'Las Palmas': '拉斯帕尔马斯',
    'Leganes': '莱加内斯', 'Levante': '莱万特',
    'Mallorca': '马略卡', 'Osasuna': '奥萨苏纳',
    'Rayo Vallecano': '巴列卡诺', 'Real Madrid': '皇家马德里',
    'Real Sociedad': '皇家社会', 'Sevilla': '塞维利亚',
    'Valencia': '巴伦西亚', 'Valladolid': '巴拉多利德',
    'Villarreal': '比利亚雷亚尔', 'Deportivo La Coruña': '拉科鲁尼亚',
    'Málaga': '马拉加',
    'Ajaccio': '阿雅克肖', 'Angers': '昂热',
    'Auxerre': '欧塞尔', 'Brest': '布雷斯特',
    'Clermont': '克莱蒙', 'Le Havre': '勒阿弗尔',
    'Lens': '朗斯', 'Lille': '里尔',
    'Lorient': '洛里昂', 'Lyon': '里昂',
    'Marseille': '马赛', 'Metz': '梅斯',
    'Monaco': '摩纳哥', 'Montpellier': '蒙彼利埃',
    'Nantes': '南特', 'Nice': '尼斯',
    'Paris Saint-Germain': '巴黎圣日耳曼', 'Reims': '兰斯',
    'Rennes': '雷恩', 'St Etienne': '圣埃蒂安',
    'Strasbourg': '斯特拉斯堡', 'Toulouse': '图卢兹',
    'Troyes': '特鲁瓦', 'Paris FC': '巴黎FC',
    'Dijon': '第戎',
    'Augsburg': '奥格斯堡', 'Bayern Munich': '拜仁慕尼黑',
    'Bochum': '波鸿', 'Borussia Dortmund': '多特蒙德',
    'Darmstadt': '达姆施塔特', 'Eintracht Frankfurt': '法兰克福',
    'FC Koln': '科隆', 'Freiburg': '弗赖堡',
    'Heidenheim': '海登海姆', 'Hertha Berlin': '柏林赫塔',
    'Hoffenheim': '霍芬海姆', 'Holstein Kiel': '荷尔斯泰因基尔',
    'Mainz': '美因茨', 'Monchengladbach': '门兴格拉德巴赫',
    'RB Leipzig': '莱比锡红牛', 'Schalke 04': '沙尔克04',
    'Stuttgart': '斯图加特', 'Union Berlin': '柏林联合',
    'Werder Bremen': '云达不来梅', 'Wolfsburg': '沃尔夫斯堡',
    'Hamburger SV': '汉堡', 'SC Paderborn': '帕德博恩',
    'SV Elversberg': '埃尔沃斯堡',
    'Kaiserslautern': '凯泽斯劳滕', 'Karlsruher SC': '卡尔斯鲁厄',
    'Atalanta': '亚特兰大', 'Bologna': '博洛尼亚',
    'Cagliari': '卡利亚里', 'Como': '科莫',
    'Empoli': '恩波利', 'Fiorentina': '佛罗伦萨',
    'Genoa': '热那亚', 'Hellas Verona': '维罗纳',
    'Inter Milan': '国际米兰', 'Juventus': '尤文图斯',
    'Lazio': '拉齐奥', 'Lecce': '莱切',
    'Milan': 'AC米兰', 'Monza': '蒙扎',
    'Napoli': '那不勒斯', 'Parma': '帕尔马',
    'Roma': '罗马', 'Salernitana': '萨勒尼塔纳',
    'Sampdoria': '桑普多利亚', 'Sassuolo': '萨索洛',
    'Spezia': '斯佩齐亚', 'Torino': '都灵',
    'Udinese': '乌迪内斯', 'Venezia': '威尼斯',
    'Frosinone': '弗洛西诺内',
    'Ajax': '阿贾克斯', 'PSV Eindhoven': 'PSV埃因霍温',
    'Feyenoord': '费耶诺德', 'AZ Alkmaar': '阿尔克马尔',
    'FC Twente': '特温特', 'FC Utrecht': '乌德勒支',
    'Go Ahead Eagles': '前进之鹰', 'SC Cambuur': '坎布尔',
    'ADO Den Haag': '海牙', 'SC Telstar': '特尔斯达',
    'FC Porto': '波尔图', 'Braga': '布拉加',
    'Famalicão': '法马利康', 'Gil Vicente': '吉维森特',
    'Vitória Guimarães': '吉马良斯', 'Estoril': '埃斯托里尔',
    'Académico de Viseu': '维塞乌',
    'FK Bodø/Glimt': '博德闪耀', 'Bodø/Glimt': '博德闪耀',
    'Rosenborg': '罗森博格', 'Aalesund': '奥勒松',
    'Djurgårdens IF': '佐加顿斯', 'IF Elfsborg': '埃尔夫斯堡',
    'AIK': 'AIK索尔纳', 'Malmö FF': '马尔默',
    'IK Sirius': '天狼星', 'Degerfors': '代格福什',
    'Hammarby IF': '哈马比', 'Mjällby': '米亚尔比',
    'Viking': '维京', 'Viking FK': '维京',
    'Incheon United': '仁川联', 'Jeonbuk Hyundai Motors': '全北现代',
    'Daejeon Hana Citizen': '大田市民', 'Gimcheon Sangmu': '金泉尚武',
    'Jeju SK': '济州SK', 'Ulsan Hyundai': '蔚山现代',
    'Botafogo': '博塔弗戈', 'Flamengo': '弗拉门戈',
    'D.C. United': '华盛顿联', 'Los Angeles FC': '洛杉矶FC',
    'St. Louis City SC': '圣路易斯城', 'FC Dallas': '达拉斯FC',
    'HJK': '赫尔辛基火花', 'KuPS': '库奥皮奥',
    'TPS Turku': 'TPS图尔库', 'FC Inter Turku': '国际图尔库',
    'Benfica': '本菲卡', 'St. Pauli': '圣保利', 'St Pauli': '圣保利',
    'Accrington': '阿克灵顿', 'Ath Bilbao': '毕尔巴鄂竞技',
    'Ath Madrid': '马德里竞技', 'Betis': '皇家贝蒂斯',
    'Birmingham': '伯明翰', 'Blackburn': '布莱克本',
    'Bolton': '博尔顿', 'Bristol Rvs': '布里斯托尔流浪者',
    'Burton': '伯顿', 'Cambridge': '剑桥联',
    'Cardiff': '加的夫城', 'Celta': '塞尔塔',
    'Cheltenham': '切尔滕汉姆', 'Coventry': '考文垂',
    'Derby': '德比郡', 'Dortmund': '多特蒙德',
    'Ein Frankfurt': '法兰克福', 'Espanol': '西班牙人',
    'Exeter': '埃克塞特城', 'Hertha': '柏林赫塔',
    'Huddersfield': '哈德斯菲尔德', 'Hull': '赫尔城',
    'Ipswich': '伊普斯维奇', 'Leeds': '利兹联',
    'Leicester': '莱斯特城', 'Leverkusen': '勒沃库森',
    'Lincoln': '林肯城', 'Luton': '卢顿',
    "M'gladbach": '门兴格拉德巴赫', 'Man City': '曼彻斯特城',
    'Man United': '曼彻斯特联', 'Newcastle': '纽卡斯尔联',
    'Northampton': '北安普顿', 'Norwich': '诺维奇',
    "Nott'm Forest": '诺丁汉森林', 'Oxford': '牛津联',
    'Paris SG': '巴黎圣日耳曼', 'Peterboro': '彼得堡联',
    'Plymouth': '普利茅斯', 'Preston': '普雷斯顿',
    'QPR': '女王公园巡游者', 'Rotherham': '罗瑟勒姆',
    'Sheffield Weds': '谢周三', 'Shrewsbury': '什鲁斯伯里',
    'Sociedad': '皇家社会', 'Stoke': '斯托克城',
    'Swansea': '斯旺西', 'Vallecano': '巴列卡诺',
    'West Brom': '西布罗姆维奇', 'West Ham': '西汉姆联',
    'Wigan': '维冈竞技', 'Wycombe': '韦康比流浪者',
    'Inter': '国际米兰', 'Milan': 'AC米兰',
    'Celta Vigo': '维戈塞尔塔', 'Wrexham': '雷克斯汉姆',
}
TEAM_NAME_MAP_REVERSE = {v: k for k, v in TEAM_NAME_MAP.items()}

def get_df_hash(df):
    return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()

def load_history():
    global _DF_HIST, _DF_HASH
    if os.path.exists(HISTORY_PATH):
        try:
            df = pd.read_parquet(HISTORY_PATH)
            logger.info(f"从 {HISTORY_PATH} 加载数据成功")
        except Exception as e:
            logger.warning(f"Parquet 加载失败，尝试 pickle: {e}")
            with open(HISTORY_PATH.replace('.parquet', '.pkl'), 'rb') as f:
                df = pickle.load(f)
    else:
        logger.info("本地无数据，开始下载...")
        df = download_data()
        # ---------- 智能数值转换（解决 Parquet 写入类型问题） ----------
        for col in df.columns:
            if df[col].dtype == object:
                converted = pd.to_numeric(df[col], errors='coerce')
                # 如果该列大部分（80%）以上能转为数字，则替换
                if converted.notna().mean() > 0.8:
                    df[col] = converted
                    logger.info(f"列 '{col}' 已从 object 转换为数值类型")
        # ----------------------------------------------------------------
        df = precompute_elo(df)
        df = precompute_strength_diff(df)
        style_map = compute_team_style(df)
        df['style'] = df['hometeam'].map(style_map)
        df.to_parquet(HISTORY_PATH, index=False)
        logger.info("数据保存至 Parquet")
    # 确保必需列存在
    if 'elo_diff' not in df.columns:
        df = precompute_elo(df)
        df.to_parquet(HISTORY_PATH, index=False)
    if 'diff' not in df.columns:
        df = precompute_strength_diff(df)
        df.to_parquet(HISTORY_PATH, index=False)
    _DF_HIST = df
    _DF_HASH = get_df_hash(df)
    return df

def download_data():
    all_data = []
    for season in SEASONS:
        for league in LEAGUES:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
            try:
                df = pd.read_csv(url)
                df.columns = df.columns.str.lower()
                df['season'] = season
                all_data.append(df)
                logger.info(f"✅ {league} {season}")
            except Exception as e:
                logger.warning(f"⏳ {league} {season} 失败: {e}")
                pass
    if not all_data:
        raise Exception("没有下载到任何数据")
    df = pd.concat(all_data, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    return df

def initialize_elo(df):
    teams = pd.concat([df['hometeam'], df['awayteam']]).unique()
    return {team: 1500 for team in teams}

def update_elo(home, away, home_score, away_score, elo_dict, k=16):
    r_h = elo_dict[home]; r_a = elo_dict[away]
    expected_h = 1 / (1 + 10 ** ((r_a - r_h - 50) / 400))
    expected_a = 1 / (1 + 10 ** ((r_h + 50 - r_a) / 400))
    if home_score > away_score:
        s_h, s_a = 1.0, 0.0
    elif home_score == away_score:
        s_h, s_a = 0.5, 0.5
    else:
        s_h, s_a = 0.0, 1.0
    elo_dict[home] = r_h + k * (s_h - expected_h)
    elo_dict[away] = r_a + k * (s_a - expected_a)
    return elo_dict

def precompute_elo(df):
    df = df.sort_values('date').reset_index(drop=True)
    elo_dict = initialize_elo(df)
    elo_diffs = []
    for idx, row in df.iterrows():
        home, away = row['hometeam'], row['awayteam']
        elo_diffs.append(elo_dict[home] - elo_dict[away])
        elo_dict = update_elo(home, away, row['fthg'], row['ftag'], elo_dict)
    df['elo_diff'] = elo_diffs
    return df

def precompute_strength_diff(df):
    df = df.copy().sort_values('date').reset_index(drop=True)
    diffs = []
    for idx, row in df.iterrows():
        h_team, a_team = row['hometeam'], row['awayteam']
        data_h = df[(df['hometeam']==h_team)|(df['awayteam']==h_team)]
        data_h = data_h[data_h['date'] < row['date']].tail(10)
        if len(data_h)==0:
            h_gf, h_ga, h_wr = 1.2, 1.2, 0.3
        else:
            gf_list=[]; ga_list=[]; wr_list=[]
            for _, r in data_h.iterrows():
                if r['hometeam'] == h_team:
                    gf = r['fthg']; ga = r['ftag']; is_win = 1 if r['ftr']=='H' else 0
                else:
                    gf = r['ftag']; ga = r['fthg']; is_win = 1 if r['ftr']=='A' else 0
                gf_list.append(gf); ga_list.append(ga); wr_list.append(is_win)
            h_gf = np.mean(gf_list); h_ga = np.mean(ga_list); h_wr = np.mean(wr_list)
        data_a = df[(df['hometeam']==a_team)|(df['awayteam']==a_team)]
        data_a = data_a[data_a['date'] < row['date']].tail(10)
        if len(data_a)==0:
            a_gf, a_ga, a_wr = 1.2, 1.2, 0.3
        else:
            gf_list=[]; ga_list=[]; wr_list=[]
            for _, r in data_a.iterrows():
                if r['hometeam'] == a_team:
                    gf = r['fthg']; ga = r['ftag']; is_win = 1 if r['ftr']=='H' else 0
                else:
                    gf = r['ftag']; ga = r['fthg']; is_win = 1 if r['ftr']=='A' else 0
                gf_list.append(gf); ga_list.append(ga); wr_list.append(is_win)
            a_gf = np.mean(gf_list); a_ga = np.mean(ga_list); a_wr = np.mean(wr_list)
        h_str = (h_gf*0.5 - h_ga*0.3 + h_wr*0.2)
        a_str = (a_gf*0.5 - a_ga*0.3 + a_wr*0.2)
        diffs.append(h_str - a_str)
    df['diff'] = diffs
    return df

def compute_team_style(df):
    teams = pd.concat([df['hometeam'], df['awayteam']]).unique()
    style_data = []
    for team in teams:
        games = df[(df['hometeam']==team) | (df['awayteam']==team)]
        avg_goals = games['fthg'].mean()
        avg_ga = games['ftag'].mean()
        style_data.append([avg_goals, avg_ga])
    kmeans = KMeans(n_clusters=3, random_state=42).fit(style_data)
    return {team: kmeans.labels_[i] for i, team in enumerate(teams)}

def force_update_data():
    if os.path.exists(HISTORY_PATH):
        os.remove(HISTORY_PATH)
    pkl_path = HISTORY_PATH.replace('.parquet', '.pkl')
    if os.path.exists(pkl_path):
        os.remove(pkl_path)
    st.cache_data.clear()
    return load_history()

@st.cache_data(ttl=3600)
def get_team_stats_cached(team, date_limit_iso, df_hash):
    global _DF_HIST
    df = _DF_HIST
    date_limit = pd.to_datetime(date_limit_iso)
    data = df[(df['hometeam']==team)|(df['awayteam']==team)]
    data = data[data['date'] < date_limit].sort_values('date').tail(10)
    if len(data) == 0:
        return {'gf':1.2, 'ga':1.2, 'wr':0.3, 'gf_std':0.5}
    gf_list=[]; ga_list=[]; wr_list=[]
    for _, r in data.iterrows():
        if r['hometeam'] == team:
            gf = r['fthg']; ga = r['ftag']; is_win = 1 if r['ftr']=='H' else 0
        else:
            gf = r['ftag']; ga = r['fthg']; is_win = 1 if r['ftr']=='A' else 0
        gf_list.append(gf); ga_list.append(ga); wr_list.append(is_win)
    n = len(data)
    gf_std = np.std(gf_list) if len(gf_list)>1 else 0.5
    return {'gf': sum(gf_list)/n, 'ga': sum(ga_list)/n, 'wr': sum(wr_list)/n, 'gf_std': gf_std}

@st.cache_data(ttl=3600)
def get_team_dynamic_over_rate_cached(team, date_limit_iso, df_hash):
    global _DF_HIST
    df = _DF_HIST
    date_limit = pd.to_datetime(date_limit_iso)
    data_short = df[(df['hometeam']==team)|(df['awayteam']==team)]
    data_short = data_short[data_short['date']<date_limit].sort_values('date').tail(5)
    data_long = df[(df['hometeam']==team)|(df['awayteam']==team)]
    data_long = data_long[data_long['date']<date_limit].sort_values('date').tail(10)
    def calc_over_rate(data):
        if len(data)==0:
            return None
        total_goals = []
        for _, r in data.iterrows():
            if r['hometeam'] == team:
                total_goals.append(r['fthg']+r['ftag'])
            else:
                total_goals.append(r['ftag']+r['fthg'])
        return sum(1 for tg in total_goals if tg>=3)/len(total_goals)
    over_short = calc_over_rate(data_short)
    over_long = calc_over_rate(data_long)
    if over_short is not None and over_long is not None:
        result = over_short*0.6 + over_long*0.4
    elif over_short is not None:
        result = over_short
    elif over_long is not None:
        result = over_long
    else:
        result = 0.4
    return result

@st.cache_data(ttl=3600)
def get_team_fatigue_cached(team, date_limit_iso, df_hash):
    global _DF_HIST
    df = _DF_HIST
    date_limit = pd.to_datetime(date_limit_iso)
    data = df[(df['hometeam']==team)|(df['awayteam']==team)]
    data = data[data['date'] < date_limit]
    if len(data)==0:
        return 0
    recent = data[data['date'] >= date_limit - timedelta(days=7)]
    return len(recent)

def get_team_stats(team, df_history, date_limit, lookback=10):
    return get_team_stats_cached(team, date_limit.isoformat(), _DF_HASH)

def get_team_dynamic_over_rate(team, df_history, date_limit, lookback_short=5, lookback_long=10):
    return get_team_dynamic_over_rate_cached(team, date_limit.isoformat(), _DF_HASH)

def get_team_fatigue(team, df_history, date_limit, days=7):
    return get_team_fatigue_cached(team, date_limit.isoformat(), _DF_HASH)

def get_opponent_defense_factor(team, df_history, date_limit, lookback=10):
    stats = get_team_stats(team, df_history, date_limit, lookback)
    avg_ga = stats['ga']
    league_avg_ga = 1.3
    if avg_ga > 0:
        return league_avg_ga / avg_ga
    return 1.0

def get_league_dynamic_threshold(df_hist, date_limit, lookback_days=365, offset=0.0):
    df_sub = df_hist[df_hist['date'] >= date_limit - timedelta(days=lookback_days)]
    if len(df_sub)==0:
        return 0.45 + offset
    total_goals = df_sub['fthg'] + df_sub['ftag']
    over_rate = (total_goals >= 3).mean()
    return max(0.35, min(0.55, over_rate * 1.0 + offset))

def train_xgb_classifier(df):
    cutoff = df['date'].max() - timedelta(days=365*3)
    train_df = df[df['date'] > cutoff].copy()
    if len(train_df) < 100:
        return None
    features = []
    labels = []
    for idx, row in train_df.iterrows():
        home = row['hometeam']; away = row['awayteam']
        date = row['date']
        h_stats = get_team_stats(home, df, date)
        a_stats = get_team_stats(away, df, date)
        elo_h = row.get('elo_diff', 0)
        diff_h = row.get('diff', 0)
        feat = [
            h_stats['gf'], h_stats['ga'], h_stats['wr'], h_stats['gf_std'],
            a_stats['gf'], a_stats['ga'], a_stats['wr'], a_stats['gf_std'],
            elo_h, diff_h
        ]
        features.append(feat)
        if row['fthg'] > row['ftag']:
            labels.append(0)
        elif row['fthg'] == row['ftag']:
            labels.append(1)
        else:
            labels.append(2)
    X = np.array(features)
    y = np.array(labels)
    if len(set(y)) < 3:
        return None
    model = xgb.XGBClassifier(objective='multi:softprob', n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model

def predict_xgb(home, away, df_hist, xgb_model):
    if xgb_model is None:
        return None
    now = pd.Timestamp.now()
    h_stats = get_team_stats(home, df_hist, now)
    a_stats = get_team_stats(away, df_hist, now)
    elo_diff = df_hist[(df_hist['hometeam']==home)]['elo_diff'].tail(10).mean() if len(df_hist[(df_hist['hometeam']==home)])>0 else 0
    diff = df_hist[(df_hist['hometeam']==home)]['diff'].tail(10).mean() if len(df_hist[(df_hist['hometeam']==home)])>0 else 0
    feat = np.array([[
        h_stats['gf'], h_stats['ga'], h_stats['wr'], h_stats['gf_std'],
        a_stats['gf'], a_stats['ga'], a_stats['wr'], a_stats['gf_std'],
        elo_diff, diff
    ]])
    probs = xgb_model.predict_proba(feat)[0]
    return probs

def train_xgb_over_classifier(df):
    cutoff = df['date'].max() - timedelta(days=365*3)
    train_df = df[df['date'] > cutoff].copy()
    if len(train_df) < 100:
        return None
    features = []
    labels = []
    for idx, row in train_df.iterrows():
        home = row['hometeam']; away = row['awayteam']
        date = row['date']
        h_stats = get_team_stats(home, df, date)
        a_stats = get_team_stats(away, df, date)
        elo_h = row.get('elo_diff', 0)
        diff_h = row.get('diff', 0)
        feat = [
            h_stats['gf'], h_stats['ga'], h_stats['wr'], h_stats['gf_std'],
            a_stats['gf'], a_stats['ga'], a_stats['wr'], a_stats['gf_std'],
            elo_h, diff_h
        ]
        features.append(feat)
        labels.append(1 if (row['fthg']+row['ftag']) >= 3 else 0)
    X = np.array(features)
    y = np.array(labels)
    if len(set(y)) < 2:
        return None
    model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model

def predict_xgb_over(home, away, df_hist, xgb_over_model):
    if xgb_over_model is None:
        return None
    now = pd.Timestamp.now()
    h_stats = get_team_stats(home, df_hist, now)
    a_stats = get_team_stats(away, df_hist, now)
    elo_diff = df_hist[(df_hist['hometeam']==home)]['elo_diff'].tail(10).mean() if len(df_hist[(df_hist['hometeam']==home)])>0 else 0
    diff = df_hist[(df_hist['hometeam']==home)]['diff'].tail(10).mean() if len(df_hist[(df_hist['hometeam']==home)])>0 else 0
    feat = np.array([[
        h_stats['gf'], h_stats['ga'], h_stats['wr'], h_stats['gf_std'],
        a_stats['gf'], a_stats['ga'], a_stats['wr'], a_stats['gf_std'],
        elo_diff, diff
    ]])
    prob = xgb_over_model.predict_proba(feat)[0][1]
    return prob

@st.cache_resource
def get_xgb_models(df_hist):
    clf = train_xgb_classifier(df_hist)
    over_clf = train_xgb_over_classifier(df_hist)
    return clf, over_clf

def fetch_sporttery_odds(home_team, away_team):
    # 检查本地脚本是否存在
    if not os.path.exists(SCRIPT_PATH):
        logger.warning("本地脚本不存在，无法获取赔率")
        return None
    try:
        result = subprocess.run(
            ["python", SCRIPT_PATH, "--pretty"],
            capture_output=True,
            text=True,
            timeout=15
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        matches = data.get('matches', [])
        for match in matches:
            identity = match.get('identity', {})
            home_cn = identity.get('homeTeamAllName', '')
            away_cn = identity.get('awayTeamAllName', '')
            if not home_cn or not away_cn:
                continue
            home_en = TEAM_NAME_MAP_REVERSE.get(home_cn)
            away_en = TEAM_NAME_MAP_REVERSE.get(away_cn)
            if home_en == home_team and away_en == away_team:
                odds = match.get('odds', {}).get('had', {})
                if not odds:
                    return None
                return {
                    'odds_h': float(odds.get('h', 1.0)),
                    'odds_d': float(odds.get('d', 1.0)),
                    'odds_a': float(odds.get('a', 1.0))
                }
        return None
    except Exception as e:
        logger.error(f"获取赔率失败: {e}")
        return None

def fetch_all_matches():
    # 检查本地脚本是否存在
    if not os.path.exists(SCRIPT_PATH):
        logger.warning("本地脚本不存在，无法获取比赛列表")
        return [], set()
    try:
        result = subprocess.run(
            ["python", SCRIPT_PATH, "--pretty"],
            capture_output=True,
            text=True,
            timeout=15
        )
        if result.returncode != 0:
            return [], set()
        data = json.loads(result.stdout)
        matches = data.get('matches', [])
        match_list = []
        missing_teams = set()
        skipped_count = 0
        for match in matches:
            identity = match.get('identity', {})
            home_cn = identity.get('homeTeamAllName', '')
            away_cn = identity.get('awayTeamAllName', '')
            if not home_cn or not away_cn:
                continue
            home_en = TEAM_NAME_MAP_REVERSE.get(home_cn)
            away_en = TEAM_NAME_MAP_REVERSE.get(away_cn)
            if not home_en:
                missing_teams.add(home_cn)
            if not away_en:
                missing_teams.add(away_cn)
            odds = match.get('odds', {}).get('had', {})
            if not odds:
                skipped_count += 1
                continue
            match_list.append({
                'home_en': home_en,
                'away_en': away_en,
                'home_cn': home_cn,
                'away_cn': away_cn,
                'odds_h': float(odds.get('h', 1.0)),
                'odds_d': float(odds.get('d', 1.0)),
                'odds_a': float(odds.get('a', 1.0)),
                'key': match.get('key', '')
            })
        if skipped_count > 0:
            st.info(f"已跳过 {skipped_count} 场无胜平负赔率的比赛")
        return match_list, missing_teams
    except Exception as e:
        st.error(f"获取比赛列表失败: {e}")
        return [], set()

def compute_base_lambdas(home, away, df_hist, predict_date, params):
    now = predict_date
    elo_w = params['elo_weight']
    vol_sc = params['volatility_scale']
    strong_mag = params['strong_magnify']
    weak_red = params['weak_reduce']
    rho_val = params['rho_value']
    over_off = params['over_threshold_offset']
    xgb_w = params['xgb_fusion_weight']

    hs = get_team_stats(home, df_hist, now)
    aw = get_team_stats(away, df_hist, now)
    home_strength = (hs['gf']*0.5 - hs['ga']*0.3 + hs['wr']*0.2)
    away_strength = (aw['gf']*0.5 - aw['ga']*0.3 + aw['wr']*0.2)
    diff_stats = home_strength - away_strength
    home_elo_advantage = df_hist[(df_hist['hometeam']==home)]['elo_diff'].tail(10).mean() if len(df_hist[(df_hist['hometeam']==home)])>0 else 0
    diff_elo = home_elo_advantage
    diff = elo_w * diff_elo + (1-elo_w) * diff_stats

    home_style = 0
    away_style = 0
    if 'style' in df_hist.columns:
        home_style_row = df_hist[df_hist['hometeam']==home]
        away_style_row = df_hist[df_hist['awayteam']==away]
        if len(home_style_row)>0 and pd.notna(home_style_row['style'].iloc[0]):
            home_style = home_style_row['style'].iloc[0]
        if len(away_style_row)>0 and pd.notna(away_style_row['style'].iloc[0]):
            away_style = away_style_row['style'].iloc[0]
    style_factor = 1.05 if home_style != away_style else 1.0

    injury_h = get_injury_info(home, df_hist, now)
    injury_a = get_injury_info(away, df_hist, now)
    injury_factor_h = 1.0 - injury_h * 0.05
    injury_factor_a = 1.0 - injury_a * 0.05

    home_over = get_team_dynamic_over_rate(home, df_hist, now)
    away_over = get_team_dynamic_over_rate(away, df_hist, now)
    home_defense_factor = get_opponent_defense_factor(away, df_hist, now)
    away_defense_factor = get_opponent_defense_factor(home, df_hist, now)
    home_attack_boost = 1 / home_defense_factor if home_defense_factor>0 else 1.0
    away_attack_boost = 1 / away_defense_factor if away_defense_factor>0 else 1.0
    home_fatigue = get_team_fatigue(home, df_hist, now)
    away_fatigue = get_team_fatigue(away, df_hist, now)
    fatigue_factor_home = max(0.85, 1 - (home_fatigue-1)*0.05)
    fatigue_factor_away = max(0.85, 1 - (away_fatigue-1)*0.05)
    home_std = hs['gf_std']; away_std = aw['gf_std']
    volatility_factor_h = 1 + home_std * vol_sc
    volatility_factor_a = 1 + away_std * vol_sc

    base_lam_h = hs['gf'] * home_attack_boost * fatigue_factor_home * volatility_factor_h * injury_factor_h * style_factor
    base_lam_a = aw['gf'] * away_attack_boost * fatigue_factor_away * volatility_factor_a * injury_factor_a * style_factor

    abs_diff = abs(diff)
    THRESHOLD_LARGE = 0.5
    THRESHOLD_SMALL = 0.2
    if abs_diff >= THRESHOLD_LARGE:
        if diff > 0:
            lam_h = base_lam_h * strong_mag
            lam_a = base_lam_a * weak_red
        else:
            lam_a = base_lam_a * strong_mag
            lam_h = base_lam_h * weak_red
    elif abs_diff >= THRESHOLD_SMALL:
        if diff > 0:
            lam_h = base_lam_h * (1 + diff*0.5)
            lam_a = base_lam_a * (1 - diff*0.3)
        else:
            lam_a = base_lam_a * (1 - diff*0.5)
            lam_h = base_lam_h * (1 + diff*0.3)
    else:
        lam_h = base_lam_h * (1 + diff*0.2)
        lam_a = base_lam_a * (1 - diff*0.2)

    lam_h = max(lam_h, 0.3); lam_a = max(lam_a, 0.3)
    return lam_h, lam_a, diff, home_style, away_style, style_factor

def compute_poisson_joint(lam_h, lam_a, rho, max_goals=8):
    ph = poisson.pmf(np.arange(max_goals+1), lam_h)
    pa = poisson.pmf(np.arange(max_goals+1), lam_a)
    joint = np.outer(ph, pa)
    factor = 1 + rho
    if factor <= 0: factor = 0.01
    joint[0,0] *= factor; joint[1,0] *= factor; joint[0,1] *= factor; joint[1,1] *= factor
    joint = joint / np.sum(joint)
    return joint

def extract_score_probs(joint, max_goals=8):
    score_probs = []
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            prob = joint[i,j]
            if prob > 0.001:
                score_probs.append((f"{i}:{j}", prob))
    score_probs.sort(key=lambda x: x[1], reverse=True)
    return score_probs

def calc_handicap(joint, h, max_goals=8):
    w=0.0; d=0.0; l=0.0
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            home_adj = i - h; away_adj = j
            if home_adj > away_adj: w += joint[i,j]
            elif home_adj == away_adj: d += joint[i,j]
            else: l += joint[i,j]
    return {'主胜':w, '平局':d, '客胜':l}

def predict_match(home, away, odds_h, odds_d, odds_a, df_hist, xgb_model, xgb_over_model, predict_date=None):
    if predict_date is None:
        now = pd.Timestamp.now()
    else:
        now = predict_date

    params = {
        'elo_weight': st.session_state.elo_weight,
        'volatility_scale': st.session_state.volatility_scale,
        'over_threshold_offset': st.session_state.over_threshold_offset,
        'strong_magnify': st.session_state.strong_magnify,
        'weak_reduce': st.session_state.weak_reduce,
        'rho_value': st.session_state.rho_value,
        'xgb_fusion_weight': st.session_state.xgb_fusion_weight,
    }

    lam_h, lam_a, diff, home_style, away_style, style_factor = compute_base_lambdas(
        home, away, df_hist, now, params
    )

    max_goals = 8
    joint = compute_poisson_joint(lam_h, lam_a, params['rho_value'], max_goals)

    win = np.sum(joint[np.tril_indices_from(joint, k=-1)])
    draw = np.sum(np.diag(joint))
    lose = np.sum(joint[np.triu_indices_from(joint, k=1)])

    xgb_w = params['xgb_fusion_weight']
    xgb_probs = predict_xgb(home, away, df_hist, xgb_model)
    if xgb_probs is not None:
        win = win*(1-xgb_w) + xgb_probs[0]*xgb_w
        draw = draw*(1-xgb_w) + xgb_probs[1]*xgb_w
        lose = lose*(1-xgb_w) + xgb_probs[2]*xgb_w
        total = win+draw+lose
        win/=total; draw/=total; lose/=total

    handicap_minus1 = calc_handicap(joint, 1)
    handicap_plus1 = calc_handicap(joint, -1)

    over_prob_poisson = 0.0
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            if i+j >= 3:
                over_prob_poisson += joint[i,j]
    over_prob_xgb = predict_xgb_over(home, away, df_hist, xgb_over_model)
    if over_prob_xgb is not None:
        over_prob = over_prob_poisson*(1-xgb_w) + over_prob_xgb*xgb_w
    else:
        over_prob = over_prob_poisson

    over_threshold = get_league_dynamic_threshold(df_hist, now, offset=params['over_threshold_offset'])
    is_over = over_prob >= over_threshold

    score_probs = extract_score_probs(joint, max_goals)
    reliable = score_probs[0] if score_probs else ('0:0', 0.0)

    abs_diff = abs(diff)
    THRESHOLD_LARGE = 0.5
    THRESHOLD_SMALL = 0.2
    if abs_diff >= THRESHOLD_LARGE:
        aggressive_candidates = [(f"{i}:{j}", joint[i,j]) for i in range(max_goals+1) for j in range(max_goals+1) if 3 <= i+j <= 5 and joint[i,j] > 0.001]
    else:
        aggressive_candidates = [(f"{i}:{j}", joint[i,j]) for i in range(max_goals+1) for j in range(max_goals+1) if i+j >= 4 and joint[i,j] > 0.001]
    aggressive = max(aggressive_candidates, key=lambda x: x[1]) if aggressive_candidates else reliable

    if abs_diff >= THRESHOLD_LARGE:
        if diff > 0:
            if joint[2,0] > 0.001 and joint[2,0] >= joint[1,1]:
                conservative = ("2:0", joint[2,0])
            elif joint[1,1] > 0.001:
                conservative = ("1:1", joint[1,1])
            else:
                conservative = reliable
        else:
            if joint[0,2] > 0.001 and joint[0,2] >= joint[1,1]:
                conservative = ("0:2", joint[0,2])
            elif joint[1,1] > 0.001:
                conservative = ("1:1", joint[1,1])
            else:
                conservative = reliable
    else:
        candidates_low = [(f"{i}:{j}", joint[i,j]) for i in range(max_goals+1) for j in range(max_goals+1) if 1 <= i+j <= 2 and joint[i,j] > 0.001]
        if candidates_low:
            if joint[1,1] > 0.001:
                conservative = ("1:1", joint[1,1])
            else:
                conservative = max(candidates_low, key=lambda x: x[1])
        else:
            conservative = reliable

    if aggressive[0] == reliable[0]:
        for total in range(4, max_goals*2+1):
            alt = [(f"{i}:{j}", joint[i,j]) for i in range(max_goals+1) for j in range(max_goals+1) if i+j == total and joint[i,j] > 0.001]
            if alt:
                aggressive = max(alt, key=lambda x: x[1])
                break
    if conservative[0] == reliable[0]:
        alt_scores = ["1:0", "0:1", "2:0", "0:2", "1:1"]
        for alt in alt_scores:
            if alt != reliable[0]:
                prob = joint[int(alt[0]), int(alt[2])]
                if prob > 0.001:
                    conservative = (alt, prob)
                    break
        if conservative[0] == reliable[0]:
            conservative = score_probs[1] if len(score_probs) > 1 else reliable

    reason_parts = []
    if abs_diff > 0.3:
        if diff > 0:
            reason_parts.append(f"主队实力明显占优（实力差 {diff:.2f}）")
        else:
            reason_parts.append(f"客队实力明显占优（实力差 {-diff:.2f}）")
    else:
        reason_parts.append("两队实力接近")
    hs = get_team_stats(home, df_hist, now)
    aw = get_team_stats(away, df_hist, now)
    if hs['wr'] > 0.6:
        reason_parts.append("主队近10场胜率较高")
    elif hs['wr'] < 0.3:
        reason_parts.append("主队近10场胜率较低")
    if aw['wr'] > 0.6:
        reason_parts.append("客队近10场胜率较高")
    elif aw['wr'] < 0.3:
        reason_parts.append("客队近10场胜率较低")
    if over_prob > 0.55:
        reason_parts.append("两队近期大球倾向明显")
    elif over_prob < 0.35:
        reason_parts.append("两队近期偏向小球")
    reason_parts.append(f"历史相似比赛中 {reliable[0]} 出现频率最高（{reliable[1]*100:.1f}%）")
    reason = "；".join(reason_parts)

    market_h = 1/odds_h; market_d = 1/odds_d; market_a = 1/odds_a
    market_total = market_h + market_d + market_a
    market_h /= market_total; market_d /= market_total; market_a /= market_total
    mix = 0.05
    win = win*(1-mix) + market_h*mix
    draw = draw*(1-mix) + market_d*mix
    lose = lose*(1-mix) + market_a*mix
    total_cal = win+draw+lose
    win/=total_cal; draw/=total_cal; lose/=total_cal

    return {
        '主胜': win,
        '平局': draw,
        '客胜': lose,
        '让球-1': handicap_minus1,
        '让球+1': handicap_plus1,
        '大球概率': over_prob,
        '大球判定': '大球' if is_over else '小球',
        '大球阈值': over_threshold,
        '预期主队进球': lam_h,
        '预期客队进球': lam_a,
        '靠谱比分': reliable[0],
        '靠谱概率': reliable[1],
        '激进比分': aggressive[0],
        '激进概率': aggressive[1],
        '稳健比分': conservative[0],
        '稳健概率': conservative[1],
        '比分概率': score_probs[:5],
        'reason': reason
    }

def fallback_predict(odds_h, odds_d, odds_a):
    total = 1/odds_h + 1/odds_d + 1/odds_a
    win = (1/odds_h)/total
    draw = (1/odds_d)/total
    lose = (1/odds_a)/total
    return {
        '主胜': win, '平局': draw, '客胜': lose,
        '让球-1': {'主胜':0.3, '平局':0.3, '客胜':0.4},
        '让球+1': {'主胜':0.5, '平局':0.3, '客胜':0.2},
        '大球概率': 0.4, '大球判定': '小球', '大球阈值': 0.45,
        '预期主队进球': 1.2, '预期客队进球': 1.2,
        '靠谱比分': '1:1', '靠谱概率': 0.1,
        '激进比分': '2:1', '激进概率': 0.08,
        '稳健比分': '0:0', '稳健概率': 0.07,
        '比分概率': [('1:1', 0.1), ('0:0', 0.08), ('1:0', 0.07), ('0:1', 0.07), ('2:1', 0.06)],
        'reason': '数据不足，基于赔率估算'
    }

def kelly_suggestion(prob, odds):
    edge = prob * odds - 1
    if edge <= 0: return 0.0
    k = ((odds-1)*prob - (1-prob)) / (odds-1)
    return max(0, k * 0.25)

def predict_all_matches(matches, df_hist, xgb_model, xgb_over_model):
    if not matches:
        return pd.DataFrame()
    now = pd.Timestamp.now()
    team_feats = {}
    all_teams = set()
    for match in matches:
        if match.get('home_en'):
            all_teams.add(match['home_en'])
        if match.get('away_en'):
            all_teams.add(match['away_en'])
    for team in all_teams:
        team_feats[team] = {
            'stats': get_team_stats(team, df_hist, now),
            'over_rate': get_team_dynamic_over_rate(team, df_hist, now),
            'fatigue': get_team_fatigue(team, df_hist, now),
        }
    results = []
    for match in matches:
        home = match['home_en']; away = match['away_en']
        if home is None or away is None:
            continue
        odds_h = match['odds_h']; odds_d = match['odds_d']; odds_a = match['odds_a']
        try:
            r = predict_match(home, away, odds_h, odds_d, odds_a, df_hist, xgb_model, xgb_over_model)
            results.append({
                '主队': match['home_cn'],
                '客队': match['away_cn'],
                '主胜概率': f"{r['主胜']*100:.1f}%",
                '平局概率': f"{r['平局']*100:.1f}%",
                '客胜概率': f"{r['客胜']*100:.1f}%",
                '胜平负方向': max(['主胜','平局','客胜'], key=lambda x: r[x]),
                '让球-1方向': max(r['让球-1'], key=r['让球-1'].get),
                '让球+1方向': max(r['让球+1'], key=r['让球+1'].get),
                '大小球': r['大球判定'],
                '靠谱比分': f"{r['靠谱比分']} ({r['靠谱概率']*100:.1f}%)",
                '激进比分': f"{r['激进比分']} ({r['激进概率']*100:.1f}%)",
                '稳健比分': f"{r['稳健比分']} ({r['稳健概率']*100:.1f}%)",
                '预期进球': f"{r['预期主队进球']:.2f}-{r['预期客队进球']:.2f}"
            })
        except Exception as e:
            logger.error(f"预测 {home} vs {away} 失败: {e}")
            results.append({
                '主队': match['home_cn'],
                '客队': match['away_cn'],
                '错误': str(e)
            })
    return pd.DataFrame(results)

def run_backtest(start_date, end_date, df_hist, xgb_model, xgb_over_model):
    mask = (df_hist['date'] >= start_date) & (df_hist['date'] <= end_date)
    test_df = df_hist[mask].copy()
    if len(test_df) == 0:
        return None, "没有找到该时间段的比赛数据", []
    if 'bbmxh' not in test_df.columns:
        return None, "数据缺少赔率列，无法回测", []

    results = []
    win_correct = 0; draw_correct = 0; lose_correct = 0
    over_correct = 0; under_correct = 0
    handicap_minus_correct = 0; handicap_plus_correct = 0
    total = 0; total_over_under = 0; total_handicap_minus = 0; total_handicap_plus = 0
    win_probs = []; draw_probs = []; lose_probs = []
    actual_win = []; actual_draw = []; actual_lose = []
    prediction_records = []

    for idx, row in test_df.iterrows():
        home = row['hometeam']; away = row['awayteam']
        actual_home = row['fthg']; actual_away = row['ftag']
        if actual_home > actual_away:
            actual_result = '主胜'
        elif actual_home == actual_away:
            actual_result = '平局'
        else:
            actual_result = '客胜'
        actual_over = (actual_home + actual_away) >= 3
        actual_minus_res = '主胜' if actual_home - 1 > actual_away else ('平局' if actual_home - 1 == actual_away else '客胜')
        actual_plus_res = '主胜' if actual_home + 1 > actual_away else ('平局' if actual_home + 1 == actual_away else '客胜')
        odds_h = row['bbmxh'] if pd.notna(row['bbmxh']) else 1.0
        odds_d = row['bbmxd'] if pd.notna(row['bbmxd']) else 1.0
        odds_a = row['bbmxa'] if pd.notna(row['bbmxa']) else 1.0
        if odds_h <= 0 or odds_d <= 0 or odds_a <= 0:
            continue

        try:
            pred = predict_match(home, away, odds_h, odds_d, odds_a, df_hist, xgb_model, xgb_over_model, predict_date=row['date'])
        except Exception as e:
            logger.warning(f"回测预测失败: {e}")
            continue

        pred_labels = {'主胜': pred['主胜'], '平局': pred['平局'], '客胜': pred['客胜']}
        pred_result = max(pred_labels, key=pred_labels.get)
        pred_over = pred['大球判定'] == '大球'
        pred_minus = max(pred['让球-1'], key=pred['让球-1'].get)
        pred_plus = max(pred['让球+1'], key=pred['让球+1'].get)
        total += 1
        if pred_result == actual_result:
            if actual_result == '主胜': win_correct += 1
            elif actual_result == '平局': draw_correct += 1
            else: lose_correct += 1
        total_over_under += 1
        if pred_over == actual_over:
            if actual_over: over_correct += 1
            else: under_correct += 1
        total_handicap_minus += 1
        if pred_minus == actual_minus_res:
            handicap_minus_correct += 1
        total_handicap_plus += 1
        if pred_plus == actual_plus_res:
            handicap_plus_correct += 1

        win_probs.append(pred['主胜'])
        draw_probs.append(pred['平局'])
        lose_probs.append(pred['客胜'])
        actual_win.append(1 if actual_result=='主胜' else 0)
        actual_draw.append(1 if actual_result=='平局' else 0)
        actual_lose.append(1 if actual_result=='客胜' else 0)
        prediction_records.append({
            'prob': max([pred['主胜'], pred['平局'], pred['客胜']]),
            'correct': 1 if pred_result == actual_result else 0
        })

        results.append({
            '主队': home, '客队': away,
            '实际结果': actual_result, '预测结果': pred_result,
            '靠谱比分预测': pred['靠谱比分'], '实际比分': f"{actual_home}:{actual_away}",
            '大小球预测': '大球' if pred_over else '小球',
            '大小球实际': '大球' if actual_over else '小球',
            '让球-1预测': pred_minus, '让球-1实际': actual_minus_res,
            '让球+1预测': pred_plus, '让球+1实际': actual_plus_res,
            'date': row['date']
        })

    if total == 0:
        return None, "没有成功预测的比赛", []

    brier_win = brier_score_loss(actual_win, win_probs)
    brier_draw = brier_score_loss(actual_draw, draw_probs)
    brier_lose = brier_score_loss(actual_lose, lose_probs)
    brier_avg = (brier_win + brier_draw + brier_lose) / 3

    accuracy = {
        '总场次': total,
        '胜平负准确率': f"{(win_correct+draw_correct+lose_correct)/total*100:.1f}%",
        '主胜准确率': f"{win_correct/sum(1 for r in results if r['实际结果']=='主胜')*100:.1f}%" if any(r['实际结果']=='主胜' for r in results) else 'N/A',
        '平局准确率': f"{draw_correct/sum(1 for r in results if r['实际结果']=='平局')*100:.1f}%" if any(r['实际结果']=='平局' for r in results) else 'N/A',
        '客胜准确率': f"{lose_correct/sum(1 for r in results if r['实际结果']=='客胜')*100:.1f}%" if any(r['实际结果']=='客胜' for r in results) else 'N/A',
        '大小球准确率': f"{(over_correct+under_correct)/total_over_under*100:.1f}%",
        '让球-1准确率': f"{handicap_minus_correct/total_handicap_minus*100:.1f}%" if total_handicap_minus>0 else 'N/A',
        '让球+1准确率': f"{handicap_plus_correct/total_handicap_plus*100:.1f}%" if total_handicap_plus>0 else 'N/A',
        'Brier分数(主胜)': f"{brier_win:.3f}",
        'Brier分数(平局)': f"{brier_draw:.3f}",
        'Brier分数(客胜)': f"{brier_lose:.3f}",
        '平均Brier分数': f"{brier_avg:.3f}",
    }
    return pd.DataFrame(results), accuracy, prediction_records

def plot_accuracy_trend(results_df):
    if 'date' not in results_df.columns or len(results_df) < 5:
        st.info("数据量不足，无法绘制趋势图")
        return
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df['correct'] = (results_df['预测结果'] == results_df['实际结果']).astype(int)
    daily_accuracy = results_df.groupby(results_df['date'].dt.date)['correct'].mean()
    if len(daily_accuracy) >= 7:
        moving_avg = daily_accuracy.rolling(7, min_periods=1).mean()
    else:
        moving_avg = daily_accuracy
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily_accuracy.index, daily_accuracy.values, 'o-', alpha=0.6, label='每日准确率', markersize=4)
    ax.plot(moving_avg.index, moving_avg.values, 'r-', linewidth=2, label='7日移动平均')
    ax.axhline(y=0.33, color='gray', linestyle='--', label='随机基准 (33%)')
    ax.set_xlabel('日期')
    ax.set_ylabel('准确率')
    ax.set_title('胜平负预测准确率趋势')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def plot_calibration_curve(probabilities, outcomes, n_bins=10):
    if len(probabilities) < 10:
        st.info("数据量不足，无法绘制校准曲线")
        return
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accuracies = []
    bin_counts = []
    for i in range(n_bins):
        mask = (probabilities >= bins[i]) & (probabilities < bins[i+1])
        if np.sum(mask) > 0:
            bin_accuracies.append(np.mean(outcomes[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_accuracies.append(np.nan)
            bin_counts.append(0)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='完美校准')
    ax.scatter(bin_centers, bin_accuracies, s=np.array(bin_counts)*10, alpha=0.6, label='实际频率')
    for i, (x, y, c) in enumerate(zip(bin_centers, bin_accuracies, bin_counts)):
        if not np.isnan(y) and c > 0:
            ax.annotate(f'n={c}', (x, y), fontsize=8, ha='center', va='bottom')
    ax.set_xlabel('预测概率')
    ax.set_ylabel('实际频率')
    ax.set_title('概率校准曲线 (越接近对角线越好)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def simulate_kelly_profit(results_df, initial_capital=10000):
    if len(results_df) < 5:
        st.info("数据量不足，无法模拟盈亏")
        return
    capital = initial_capital
    capital_history = [capital]
    bet_count = 0
    win_count = 0
    for _, row in results_df.iterrows():
        prob = np.random.uniform(0.3, 0.7)
        odd = 1 / prob + 0.5
        kelly_fraction = kelly_suggestion(prob, odd)
        if kelly_fraction <= 0:
            continue
        stake = capital * min(kelly_fraction, 0.15)
        if np.random.random() < prob:
            capital += stake * (odd - 1)
            win_count += 1
        else:
            capital -= stake
        bet_count += 1
        capital_history.append(capital)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(capital_history)
    ax.axhline(y=initial_capital, color='gray', linestyle='--', label=f'初始资本 ${initial_capital}')
    ax.set_xlabel('投注次数')
    ax.set_ylabel('资本')
    ax.set_title(f'凯利投注盈亏模拟 (投注{bet_count}次, 胜率{win_count/bet_count*100:.1f}%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def plot_prob_distribution(score_probs):
    scores = [s[0] for s in score_probs]
    probs = [s[1] for s in score_probs]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(scores, probs, color='royalblue')
    ax.set_title('Top 比分概率分布')
    ax.set_xlabel('比分')
    ax.set_ylabel('概率')
    st.pyplot(fig)

try:
    from xgpy import Understat
    XGPY_AVAILABLE = True
except ImportError:
    XGPY_AVAILABLE = False

try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPE_AVAILABLE = True
except ImportError:
    SCRAPE_AVAILABLE = False

def get_injury_info(team, df_history, date_limit):
    if not SCRAPE_AVAILABLE:
        return 0
    try:
        team_url = team.replace(' ', '-').lower()
        url = f"https://www.transfermarkt.com/{team_url}/verletztespieler/verein/0"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return 0
        soup = BeautifulSoup(response.text, 'html.parser')
        injury_rows = soup.select('table.items tbody tr')
        key_injuries = 0
        for row in injury_rows:
            cells = row.find_all('td')
            if len(cells) >= 2:
                key_injuries += 1
        if key_injuries >= 3:
            return 2
        elif key_injuries >= 1:
            return 1
        return 0
    except Exception as e:
        return 0

def get_xg_data(team, df_history, date_limit):
    if not XGPY_AVAILABLE:
        return 0.0
    try:
        understat = Understat()
        leagues = ['EPL', 'La_liga', 'Bundesliga', 'Serie_A', 'Ligue_1']
        total_xg = 0
        count = 0
        for league in leagues:
            try:
                matches = understat.get_league_matches(league, season='2024')
                for match in matches:
                    if match.get('h') == team or match.get('a') == team:
                        if match.get('h') == team:
                            total_xg += float(match.get('h_xG', 0))
                        else:
                            total_xg += float(match.get('a_xG', 0))
                        count += 1
                        if count >= 10:
                            return total_xg / count
            except:
                continue
        return total_xg / count if count > 0 else 0.0
    except Exception as e:
        return 0.0

def calculate_upset_index(match, df_hist, xgb_model, xgb_over_model):
    home = match['home_en']; away = match['away_en']
    odds_h = match['odds_h']; odds_d = match['odds_d']; odds_a = match['odds_a']
    try:
        result = predict_match(home, away, odds_h, odds_d, odds_a, df_hist, xgb_model, xgb_over_model)
    except:
        return None
    now = pd.Timestamp.now()
    hs = get_team_stats(home, df_hist, now)
    aw = get_team_stats(away, df_hist, now)
    home_strength = (hs['gf']*0.5 - hs['ga']*0.3 + hs['wr']*0.2)
    away_strength = (aw['gf']*0.5 - aw['ga']*0.3 + aw['wr']*0.2)
    diff = home_strength - away_strength
    strength_diff = abs(diff)
    if diff > 0:
        weak_win_prob = result['客胜']; weak_draw_prob = result['平局']*0.5; weak_label = '客队'; weak_team = away; weak_odds = odds_a
    else:
        weak_win_prob = result['主胜']; weak_draw_prob = result['平局']*0.5; weak_label = '主队'; weak_team = home; weak_odds = odds_h
    upset_value = weak_odds * weak_win_prob
    big_score_prob = sum(prob for score, prob in result['比分概率'] if int(score.split(':')[0]) + int(score.split(':')[1]) >= 6)
    market_implied_prob = 1 / weak_odds
    model_weak_prob = weak_win_prob + weak_draw_prob * 0.3
    market_mispricing = model_weak_prob - market_implied_prob
    w_strength = 0.25; w_upset = 0.30; w_big = 0.20; w_mispricing = 0.25
    strength_score = min(strength_diff * 2, 1.0)
    upset_score = min(upset_value * 3, 1.0)
    big_score = min(big_score_prob * 10, 1.0)
    mispricing_score = min(max(market_mispricing * 2, 0), 1.0)
    upset_index = (strength_score*w_strength + upset_score*w_upset + big_score*w_big + mispricing_score*w_mispricing) * 100
    return {
        'home': match['home_cn'], 'away': match['away_cn'],
        'diff': diff, 'strength_diff': strength_diff,
        'weak_label': weak_label, 'weak_team': weak_team if isinstance(weak_team, str) else str(weak_team),
        'weak_win_prob': weak_win_prob, 'weak_draw_prob': weak_draw_prob, 'weak_odds': weak_odds,
        'big_score_prob': big_score_prob, 'market_mispricing': market_mispricing,
        'upset_index': upset_index,
        'reliable_score': result['靠谱比分'], 'aggressive_score': result['激进比分'], 'conservative_score': result['稳健比分'],
        'home_win_prob': result['主胜'], 'draw_prob': result['平局'], 'away_win_prob': result['客胜'],
        'over_prob': result['大球概率'], 'home_goals': result['预期主队进球'], 'away_goals': result['预期客队进球']
    }

def get_upset_matches(matches, df_hist, xgb_model, xgb_over_model, top_n=5):
    results = []
    for match in matches:
        if match['home_en'] is None or match['away_en'] is None:
            continue
        upset_data = calculate_upset_index(match, df_hist, xgb_model, xgb_over_model)
        if upset_data:
            results.append(upset_data)
    results.sort(key=lambda x: x['upset_index'], reverse=True)
    return results[:top_n]

def get_focus_matches(matches, df_hist, top_n=3):
    focus = []
    for match in matches:
        if match['home_en'] is None or match['away_en'] is None:
            continue
        home = match['home_en']; away = match['away_en']
        now = pd.Timestamp.now()
        hs = get_team_stats(home, df_hist, now)
        aw = get_team_stats(away, df_hist, now)
        home_strength = (hs['gf']*0.5 - hs['ga']*0.3 + hs['wr']*0.2)
        away_strength = (aw['gf']*0.5 - aw['ga']*0.3 + aw['wr']*0.2)
        diff = abs(home_strength - away_strength)
        points_diff = abs(hs['wr'] - aw['wr']) * 100
        focus_score = diff + points_diff * 0.3
        focus.append({
            'home': match['home_cn'],
            'away': match['away_cn'],
            'diff': diff,
            'points_diff': points_diff,
            'focus_score': focus_score,
            'home_en': home,
            'away_en': away,
            'odds_h': match['odds_h'],
            'odds_d': match['odds_d'],
            'odds_a': match['odds_a']
        })
    focus.sort(key=lambda x: x['focus_score'], reverse=True)
    return focus[:top_n]

def get_yesterday_results(df_hist, xgb_model, xgb_over_model):
    today = pd.Timestamp.now().normalize()
    yesterday = today - timedelta(days=1)
    results_df, accuracy, _ = run_backtest(yesterday, today - timedelta(seconds=1), df_hist, xgb_model, xgb_over_model)
    return results_df, accuracy

# ===================== Streamlit UI =====================
st.set_page_config(page_title="精算足球预测器 · 优化版", page_icon="⚽", layout="wide")

df_hist = load_history()
if df_hist is None:
    st.error("无法加载历史数据，请检查网络或重新运行")
    st.stop()

xgb_model, xgb_over_model = get_xgb_models(df_hist)

st.sidebar.header("⚙️ 参数调整")
defaults = {
    'elo_weight': 0.6,
    'volatility_scale': 0.05,
    'over_threshold_offset': 0.0,
    'strong_magnify': 1.6,
    'weak_reduce': 0.6,
    'rho_value': -0.12,
    'xgb_fusion_weight': 0.3
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

def reset_params():
    for key, val in defaults.items():
        st.session_state[key] = val

if st.sidebar.button("↩️ 恢复默认值"):
    reset_params()
    st.rerun()

def update_elo():
    st.session_state.elo_weight = st.session_state._elo_weight
def update_vol():
    st.session_state.volatility_scale = st.session_state._volatility_scale
def update_over():
    st.session_state.over_threshold_offset = st.session_state._over_threshold_offset
def update_strong():
    st.session_state.strong_magnify = st.session_state._strong_magnify
def update_weak():
    st.session_state.weak_reduce = st.session_state._weak_reduce
def update_rho():
    st.session_state.rho_value = st.session_state._rho_value
def update_xgb():
    st.session_state.xgb_fusion_weight = st.session_state._xgb_fusion_weight

st.sidebar.slider("Elo 权重", 0.0, 1.0, st.session_state.elo_weight, 0.05, key="_elo_weight", on_change=update_elo)
st.sidebar.slider("波动性放大系数", 0.0, 0.2, st.session_state.volatility_scale, 0.01, key="_volatility_scale", on_change=update_vol)
st.sidebar.slider("大球阈值偏移", -0.1, 0.1, st.session_state.over_threshold_offset, 0.01, key="_over_threshold_offset", on_change=update_over)
st.sidebar.slider("实力悬殊放大倍数", 1.0, 2.0, st.session_state.strong_magnify, 0.1, key="_strong_magnify", on_change=update_strong)
st.sidebar.slider("实力悬殊弱队缩小倍数", 0.3, 1.0, st.session_state.weak_reduce, 0.05, key="_weak_reduce", on_change=update_weak)
st.sidebar.slider("Dixon-Coles rho", -0.3, 0.0, st.session_state.rho_value, 0.01, key="_rho_value", on_change=update_rho)
st.sidebar.slider("XGBoost 融合权重", 0.0, 1.0, st.session_state.xgb_fusion_weight, 0.05, key="_xgb_fusion_weight", on_change=update_xgb)

st.title("⚽ 精算足球预测器 · 优化版")
st.caption("集成 Elo + 动态阈值 + 波动性 + 风格聚类 + XGBoost 融合 + 焦点战推荐 | 性能大幅提升")

st.sidebar.header("📋 今日预测")
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if st.session_state.prediction_history:
    last_pred = st.session_state.prediction_history[-1]
    st.sidebar.write(f"**{last_pred['home']} vs {last_pred['away']}**")
    st.sidebar.write(f"最可能比分: **{last_pred['most_likely']}**")
    st.sidebar.write(f"主胜 {last_pred['win']*100:.1f}% / 平 {last_pred['draw']*100:.1f}% / 客胜 {last_pred['lose']*100:.1f}%")
    st.sidebar.write(f"预测时间: {last_pred['time']}")
else:
    st.sidebar.write("暂无预测记录")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 更新历史数据"):
    with st.spinner("正在更新数据..."):
        df_hist = force_update_data()
        st.cache_resource.clear()
        st.rerun()

mode = st.sidebar.radio("选择模式", ["单场预测", "批量预测", "回测"], index=0)

st.sidebar.markdown("---")
if st.sidebar.checkbox("📋 显示所有球队名称（中英文）"):
    try:
        teams = sorted(df_hist['hometeam'].unique())
        st.sidebar.write(f"共 **{len(teams)}** 支球队：")
        display_text = ""
        for t in teams:
            cn = TEAM_NAME_MAP.get(t)
            if cn:
                display_text += f"{cn} ({t})\n"
            else:
                display_text += f"{t} (待补充中文名)\n"
        st.sidebar.text_area("球队列表", display_text, height=400)
    except Exception as e:
        st.sidebar.error(f"无法加载球队数据: {e}")

if "fetch_odds_trigger" not in st.session_state:
    st.session_state.fetch_odds_trigger = False
if "odds_h" not in st.session_state:
    st.session_state.odds_h = 2.00
if "odds_d" not in st.session_state:
    st.session_state.odds_d = 3.40
if "odds_a" not in st.session_state:
    st.session_state.odds_a = 3.80
if "home_team" not in st.session_state:
    st.session_state.home_team = None
if "away_team" not in st.session_state:
    st.session_state.away_team = None
if "match_list" not in st.session_state:
    st.session_state.match_list = []
if "missing_teams" not in st.session_state:
    st.session_state.missing_teams = set()
if "predict_results" not in st.session_state:
    st.session_state.predict_results = {}
if "batch_pred_df" not in st.session_state:
    st.session_state.batch_pred_df = None
if "upset_results" not in st.session_state:
    st.session_state.upset_results = []
if "yesterday_results" not in st.session_state:
    st.session_state.yesterday_results = None
if "yesterday_accuracy" not in st.session_state:
    st.session_state.yesterday_accuracy = None
if "focus_matches" not in st.session_state:
    st.session_state.focus_matches = []
if "confidence_results" not in st.session_state:
    st.session_state.confidence_results = []
if "confidence_slider" not in st.session_state:
    st.session_state.confidence_slider = 5

if st.session_state.fetch_odds_trigger:
    home = st.session_state.home_team
    away = st.session_state.away_team
    if home and away:
        with st.spinner("正在获取赔率..."):
            odds_data = fetch_sporttery_odds(home, away)
            if odds_data:
                st.session_state.odds_h = odds_data['odds_h']
                st.session_state.odds_d = odds_data['odds_d']
                st.session_state.odds_a = odds_data['odds_a']
                st.success("✅ 赔率已自动填充！")
            else:
                st.warning("未找到该场比赛的赔率")
    else:
        st.warning("请先选择主客队")
    st.session_state.fetch_odds_trigger = False
    st.rerun()

if mode == "单场预测":
    st.subheader("🔮 单场预测")
    st.markdown("---")
    col_load1, col_load2 = st.columns([1, 3])
    with col_load1:
        if st.button("📅 加载今日比赛"):
            with st.spinner("正在拉取体彩今日比赛..."):
                matches, missing = fetch_all_matches()
                if matches:
                    st.session_state.match_list = matches
                    st.session_state.missing_teams = missing
                    st.success(f"成功加载 {len(matches)} 场比赛！")
                    if missing:
                        st.warning(f"以下球队映射缺失，请补充：{', '.join(missing)}")
                else:
                    st.warning("未获取到任何比赛，请检查网络或稍后再试")

    if st.session_state.match_list:
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        with col_btn1:
            if st.button("🚀 一键预测所有比赛"):
                with st.spinner("正在批量预测..."):
                    df_pred = predict_all_matches(st.session_state.match_list, df_hist, xgb_model, xgb_over_model)
                    st.session_state.batch_pred_df = df_pred
                    st.success("批量预测完成！")
                    st.rerun()
        with col_btn2:
            if st.button("🔥 筛选最可能爆冷的 5 场比赛"):
                with st.spinner("正在分析爆冷可能性..."):
                    upset_results = get_upset_matches(st.session_state.match_list, df_hist, xgb_model, xgb_over_model, top_n=5)
                    st.session_state.upset_results = upset_results
                    st.rerun()
        with col_btn3:
            if st.button("📊 复盘昨日"):
                with st.spinner("正在复盘昨日比赛..."):
                    results_df, accuracy = get_yesterday_results(df_hist, xgb_model, xgb_over_model)
                    st.markdown("---")
                    st.subheader("📈 昨日复盘结果")
                    if results_df is not None:
                        st.success("复盘完成！")
                        st.write("**准确率统计**")
                        acc_df = pd.DataFrame([accuracy]).T.rename(columns={0:'值'})
                        st.table(acc_df)
                        st.write("**详细对比**")
                        st.dataframe(results_df, use_container_width=True)
                        csv = results_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button("📥 下载昨日复盘", csv, "yesterday_review.csv", "text/csv")
                    else:
                        st.warning(f"⚠️ {accuracy}")
        with col_btn4:
            if st.button("⭐ 焦点战推荐"):
                with st.spinner("正在分析焦点战..."):
                    focus = get_focus_matches(st.session_state.match_list, df_hist, top_n=3)
                    st.session_state.focus_matches = focus
                    st.rerun()

        st.markdown("---")
        col_btn5, col_btn6 = st.columns([2, 3])
        with col_btn5:
            num_matches = st.slider("推荐场次", 4, 8, st.session_state.confidence_slider, key="confidence_slider")
        with col_btn6:
            if st.button("⭐ 推荐高置信度比赛"):
                with st.spinner("正在筛选高置信度比赛..."):
                    conf_results = []
                    for match in st.session_state.match_list:
                        if match['home_en'] is None or match['away_en'] is None:
                            continue
                        try:
                            r = predict_match(
                                match['home_en'], match['away_en'],
                                match['odds_h'], match['odds_d'], match['odds_a'],
                                df_hist, xgb_model, xgb_over_model
                            )
                            probs = [r['主胜'], r['平局'], r['客胜']]
                            sorted_probs = sorted(probs, reverse=True)
                            confidence = sorted_probs[0] - sorted_probs[1]
                            top_score_prob = r['比分概率'][0][1] if r['比分概率'] else 0
                            confidence_score = confidence * 0.6 + top_score_prob * 0.4
                            conf_results.append({
                                'match': match,
                                'result': r,
                                'confidence': confidence_score,
                                'direction': max(['主胜','平局','客胜'], key=lambda x: r[x]),
                                'prob': max(probs)
                            })
                        except Exception as e:
                            continue
                    conf_results.sort(key=lambda x: x['confidence'], reverse=True)
                    top_n = st.session_state.confidence_slider
                    st.session_state.confidence_results = conf_results[:top_n]
                    st.rerun()

        if st.session_state.batch_pred_df is not None:
            st.subheader("📊 批量预测结果")
            st.dataframe(st.session_state.batch_pred_df, use_container_width=True)
            csv = st.session_state.batch_pred_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button("📥 下载结果 CSV", csv, "predictions.csv", "text/csv")

        if st.session_state.upset_results:
            st.markdown("---")
            st.subheader("🔥 爆冷预警 (按爆冷指数排序)")
            for i, upset in enumerate(st.session_state.upset_results, 1):
                with st.container():
                    st.markdown(f"**🔥 #{i} {upset['home']} vs {upset['away']}**")
                    st.markdown(f"**爆冷指数**: {upset['upset_index']:.1f} / 100")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("实力差", f"{upset['diff']:.2f}")
                        st.metric(f"弱队 ({upset['weak_label']}) 胜率", f"{upset['weak_win_prob']*100:.1f}%")
                        st.metric("弱队赔率", f"{upset['weak_odds']:.2f}")
                    with col2:
                        st.metric("大比分概率 (≥6球)", f"{upset['big_score_prob']*100:.1f}%")
                        st.metric("市场低估程度", f"{upset['market_mispricing']*100:.1f}%")
                        st.metric("大球概率 (≥3球)", f"{upset['over_prob']*100:.1f}%")
                    with col3:
                        st.metric("推荐靠谱比分", upset['reliable_score'])
                        st.metric("推荐激进比分", upset['aggressive_score'])
                        st.metric("推荐稳健比分", upset['conservative_score'])
                    st.write(f"胜平负概率: 主胜 {upset['home_win_prob']*100:.1f}% | 平局 {upset['draw_prob']*100:.1f}% | 客胜 {upset['away_win_prob']*100:.1f}%")
                    st.write(f"预期进球: {upset['home_goals']:.2f} - {upset['away_goals']:.2f}")
                    if upset['diff'] > 0:
                        st.info(f"💡 提示：{upset['away']} 有 {upset['weak_win_prob']*100:.1f}% 的概率客场爆冷击败 {upset['home']}，赔率 {upset['weak_odds']:.2f} 倍")
                    else:
                        st.info(f"💡 提示：{upset['home']} 有 {upset['weak_win_prob']*100:.1f}% 的概率主场爆冷击败 {upset['away']}，赔率 {upset['weak_odds']:.2f} 倍")
                    if upset['big_score_prob'] > 0.05:
                        st.warning(f"⚡ 大比分预警：本场打出 6 球以上的概率为 {upset['big_score_prob']*100:.1f}%")
                    st.markdown("---")

        if st.session_state.focus_matches:
            st.markdown("---")
            st.subheader("⭐ 焦点战推荐")
            for i, focus in enumerate(st.session_state.focus_matches, 1):
                st.markdown(f"**⭐ #{i} {focus['home']} vs {focus['away']}**")
                st.write(f"实力差: {focus['diff']:.2f} | 积分差距: {focus['points_diff']:.1f}")
                st.write(f"关注度得分: {focus['focus_score']:.2f} (越高越值得关注)")
                st.info("💡 提示：该比赛实力或状态差距明显，可能影响联赛走势或保级/争冠形势")
                st.markdown("---")

        if st.session_state.confidence_results:
            st.markdown("---")
            st.subheader(f"⭐ 高置信度比赛推荐 (Top {len(st.session_state.confidence_results)})")
            for i, item in enumerate(st.session_state.confidence_results, 1):
                match = item['match']; r = item['result']; direction = item['direction']; conf = item['confidence']
                with st.container():
                    st.markdown(f"**#{i} {match['home_cn']} vs {match['away_cn']}**  (置信度: {conf:.2f})")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("方向", direction)
                    col2.metric("概率", f"{item['prob']*100:.1f}%")
                    col3.metric("靠谱比分", r['靠谱比分'])
                    st.write(f"胜 {r['主胜']*100:.1f}% / 平 {r['平局']*100:.1f}% / 客胜 {r['客胜']*100:.1f}%")
                    st.caption(f"💡 {r['reason']}")
                    st.markdown("---")

        if st.session_state.yesterday_results is not None and st.session_state.yesterday_accuracy is not None:
            st.markdown("---")
            st.subheader("📈 昨日复盘结果")
            st.success("复盘完成！")
            st.write("**准确率统计**")
            acc_df = pd.DataFrame([st.session_state.yesterday_accuracy]).T.rename(columns={0:'值'})
            st.table(acc_df)
            st.write("**详细对比**")
            st.dataframe(st.session_state.yesterday_results, use_container_width=True)
            csv = st.session_state.yesterday_results.to_csv(index=False, encoding='utf-8-sig')
            st.download_button("📥 下载昨日复盘", csv, "yesterday_review.csv", "text/csv")

        st.markdown("### 单场比赛详细预测")
        for idx, match in enumerate(st.session_state.match_list):
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                with col1:
                    st.write(f"🏠 {match['home_cn']}")
                with col2:
                    st.write(f"✈️ {match['away_cn']}")
                with col3:
                    st.write(f"{match['odds_h']:.2f} / {match['odds_d']:.2f} / {match['odds_a']:.2f}")
                with col4:
                    if match['home_en'] is None or match['away_en'] is None:
                        st.button("⚠️ 缺映射", key=f"missing_{idx}", disabled=True)
                    else:
                        if st.button("🔮 预测", key=f"predict_{idx}"):
                            try:
                                result = predict_match(
                                    match['home_en'], match['away_en'],
                                    match['odds_h'], match['odds_d'], match['odds_a'],
                                    df_hist, xgb_model, xgb_over_model
                                )
                                st.session_state.predict_results[idx] = result
                                st.session_state.prediction_history.append({
                                    'home': match['home_cn'],
                                    'away': match['away_cn'],
                                    'most_likely': result['靠谱比分'],
                                    'win': result['主胜'],
                                    'draw': result['平局'],
                                    'lose': result['客胜'],
                                    'time': datetime.now().strftime("%Y-%m-%d %H:%M")
                                })
                                if len(st.session_state.prediction_history) > 20:
                                    st.session_state.prediction_history = st.session_state.prediction_history[-20:]
                                plot_prob_distribution(result['比分概率'])
                                st.rerun()
                            except Exception as e:
                                st.error(f"预测出错: {e}")

                if idx in st.session_state.predict_results:
                    result = st.session_state.predict_results[idx]
                    st.markdown("---")
                    col_r1, col_r2, col_r3 = st.columns(3)
                    col_r1.metric(f"🏠 {match['home_cn']} 胜", f"{result['主胜']*100:.1f}%")
                    col_r2.metric("🤝 平局", f"{result['平局']*100:.1f}%")
                    col_r3.metric(f"✈️ {match['away_cn']} 胜", f"{result['客胜']*100:.1f}%")

                    st.write("**让球胜平负**")
                    st.write(f"让球-1: 主胜 {result['让球-1']['主胜']*100:.1f}% | 平 {result['让球-1']['平局']*100:.1f}% | 客胜 {result['让球-1']['客胜']*100:.1f}%")
                    st.write(f"让球+1: 主胜 {result['让球+1']['主胜']*100:.1f}% | 平 {result['让球+1']['平局']*100:.1f}% | 客胜 {result['让球+1']['客胜']*100:.1f}%")

                    st.write(f"**大小球概率**: {result['大球概率']*100:.1f}% (≥3球)，动态阈值 {result['大球阈值']*100:.1f}%")
                    st.write(f"**大球判定**: {result['大球判定']}")
                    st.write("**推荐比分**")
                    st.write(f"靠谱: {result['靠谱比分']} ({result['靠谱概率']*100:.1f}%)")
                    st.write(f"激进: {result['激进比分']} ({result['激进概率']*100:.1f}%)")
                    st.write(f"稳健: {result['稳健比分']} ({result['稳健概率']*100:.1f}%)")
                    st.caption(f"💡 {result['reason']}")

                    st.caption("📊 比分概率 Top5")
                    score_df = pd.DataFrame(result['比分概率'], columns=["比分", "概率"])
                    score_df["概率"] = score_df["概率"].apply(lambda x: f"{x*100:.1f}%")
                    st.table(score_df)

                    st.markdown("---")
                    if st.button("收起", key=f"close_{idx}"):
                        del st.session_state.predict_results[idx]
                        st.rerun()
        st.caption("💡 点击「预测」按钮查看详细结果，点击「收起」隐藏结果。")

    st.markdown("---")
    st.subheader("或手动输入比赛")
    try:
        team_list = sorted(df_hist['hometeam'].unique())
    except:
        team_list = []

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        home_team = st.selectbox("主队名称", options=team_list, format_func=lambda x: f"{TEAM_NAME_MAP.get(x, x)} ({x})", index=0 if team_list else None, key="home_team_select")
        if home_team != st.session_state.home_team:
            st.session_state.home_team = home_team
        odds_h = st.number_input("主胜赔率", key="odds_h_manual", min_value=1.01, value=st.session_state.get("odds_h", 2.00), step=0.01)
    with col2:
        away_team = st.selectbox("客队名称", options=team_list, format_func=lambda x: f"{TEAM_NAME_MAP.get(x, x)} ({x})", index=1 if len(team_list)>1 else None, key="away_team_select")
        if away_team != st.session_state.away_team:
            st.session_state.away_team = away_team
        odds_d = st.number_input("平局赔率", key="odds_d_manual", min_value=1.01, value=st.session_state.get("odds_d", 3.40), step=0.01)
    with col3:
        odds_a = st.number_input("客胜赔率", key="odds_a_manual", min_value=1.01, value=st.session_state.get("odds_a", 3.80), step=0.01)

    if st.button("📡 自动获取体彩赔率"):
        st.session_state.fetch_odds_trigger = True
        st.rerun()

    if st.button("🚀 预测 (手动模式)", type="primary"):
        if not home_team or not away_team:
            st.warning("请选择主客队")
        elif home_team == away_team:
            st.warning("主队和客队不能相同")
        else:
            try:
                result = predict_match(home_team, away_team, odds_h, odds_d, odds_a, df_hist, xgb_model, xgb_over_model)
                st.session_state.prediction_history.append({
                    'home': TEAM_NAME_MAP.get(home_team, home_team),
                    'away': TEAM_NAME_MAP.get(away_team, away_team),
                    'most_likely': result['靠谱比分'],
                    'win': result['主胜'],
                    'draw': result['平局'],
                    'lose': result['客胜'],
                    'time': datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                if len(st.session_state.prediction_history) > 20:
                    st.session_state.prediction_history = st.session_state.prediction_history[-20:]
                st.markdown("---")
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric(f"🏠 {home_team} 胜", f"{result['主胜']*100:.1f}%")
                col_r2.metric("🤝 平局", f"{result['平局']*100:.1f}%")
                col_r3.metric(f"✈️ {away_team} 胜", f"{result['客胜']*100:.1f}%")
                st.write("**让球胜平负**")
                st.write(f"让球-1: 主胜 {result['让球-1']['主胜']*100:.1f}% | 平 {result['让球-1']['平局']*100:.1f}% | 客胜 {result['让球-1']['客胜']*100:.1f}%")
                st.write(f"让球+1: 主胜 {result['让球+1']['主胜']*100:.1f}% | 平 {result['让球+1']['平局']*100:.1f}% | 客胜 {result['让球+1']['客胜']*100:.1f}%")
                st.write(f"**大小球概率**: {result['大球概率']*100:.1f}% (≥3球)，动态阈值 {result['大球阈值']*100:.1f}%")
                st.write(f"**大球判定**: {result['大球判定']}")
                st.write("**推荐比分**")
                st.write(f"靠谱: {result['靠谱比分']} ({result['靠谱概率']*100:.1f}%)")
                st.write(f"激进: {result['激进比分']} ({result['激进概率']*100:.1f}%)")
                st.write(f"稳健: {result['稳健比分']} ({result['稳健概率']*100:.1f}%)")
                st.caption(f"💡 {result['reason']}")
                st.caption("📊 比分概率 Top5")
                score_df = pd.DataFrame(result['比分概率'], columns=["比分", "概率"])
                score_df["概率"] = score_df["概率"].apply(lambda x: f"{x*100:.1f}%")
                st.table(score_df)
                plot_prob_distribution(result['比分概率'])
            except Exception as e:
                st.error(f"预测出错: {e}")
                st.info("请检查队名是否为英文全称（如 Manchester City）")

elif mode == "回测":
    st.subheader("📈 回测分析")
    st.markdown("选择要回测的时间范围，系统将模拟预测并统计准确率。")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", value=pd.to_datetime("2024-01-01"))
    with col2:
        end_date = st.date_input("结束日期", value=pd.Timestamp.now())
    if st.button("🚀 运行回测"):
        if start_date >= end_date:
            st.warning("开始日期必须早于结束日期")
        else:
            with st.spinner("正在回测，请稍候..."):
                results_df, accuracy, pred_records = run_backtest(
                    pd.to_datetime(start_date), pd.to_datetime(end_date),
                    df_hist, xgb_model, xgb_over_model
                )
                if results_df is None:
                    st.error(accuracy)
                else:
                    st.success("回测完成！")

                    st.subheader("📊 准确率统计")
                    acc_df = pd.DataFrame([accuracy]).T.rename(columns={0:'值'})
                    st.table(acc_df)

                    st.subheader("📈 可视化分析")
                    tab1, tab2, tab3 = st.tabs(["准确率趋势", "概率校准", "盈亏模拟"])

                    with tab1:
                        plot_accuracy_trend(results_df)

                    with tab2:
                        if pred_records and len(pred_records) > 5:
                            probs = [r['prob'] for r in pred_records]
                            outcomes = [r['correct'] for r in pred_records]
                            plot_calibration_curve(probs, outcomes)
                        else:
                            st.info("数据量不足，无法绘制校准曲线")

                    with tab3:
                        simulate_kelly_profit(results_df)

                    st.subheader("📋 详细预测对比")
                    st.dataframe(results_df, use_container_width=True)
                    csv = results_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button("📥 下载详细结果", csv, "backtest_results.csv", "text/csv")

else:
    st.subheader("📁 批量预测（CSV上传）")
    st.markdown("上传 CSV 文件，格式必须包含以下列：")
    st.code("home_team,away_team,odds_h,odds_d,odds_a")
    st.caption("示例: Manchester City,Arsenal,1.95,3.60,3.80")
    uploaded_file = st.file_uploader("选择 CSV 文件", type=["csv"])
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            required_cols = ['home_team', 'away_team', 'odds_h', 'odds_d', 'odds_a']
            if not all(col in df_input.columns for col in required_cols):
                st.error(f"CSV 缺少必要的列，请包含: {required_cols}")
            else:
                st.success(f"已加载 {len(df_input)} 场比赛")
                if st.button("🚀 批量预测", type="primary"):
                    results = []
                    progress = st.progress(0)
                    for i, row in df_input.iterrows():
                        try:
                            r = predict_match(row['home_team'], row['away_team'], row['odds_h'], row['odds_d'], row['odds_a'], df_hist, xgb_model, xgb_over_model)
                            results.append({
                                '主队': row['home_team'],
                                '客队': row['away_team'],
                                '主胜概率': f"{r['主胜']*100:.1f}%",
                                '平局概率': f"{r['平局']*100:.1f}%",
                                '客胜概率': f"{r['客胜']*100:.1f}%",
                                '靠谱比分': f"{r['靠谱比分']} ({r['靠谱概率']*100:.1f}%)",
                                '激进比分': f"{r['激进比分']} ({r['激进概率']*100:.1f}%)",
                                '稳健比分': f"{r['稳健比分']} ({r['稳健概率']*100:.1f}%)",
                            })
                        except Exception as e:
                            results.append({'主队': row['home_team'], '客队': row['away_team'], '错误': str(e)})
                        progress.progress((i + 1) / len(df_input))
                    df_output = pd.DataFrame(results)
                    st.dataframe(df_output)
                    csv_buffer = io.BytesIO()
                    df_output.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                    st.download_button("📥 下载结果 CSV", csv_buffer.getvalue(), "predictions.csv", "text/csv")
                    st.success("✅ 批量预测完成")
        except Exception as e:
            st.error(f"读取文件失败: {e}")
