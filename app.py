import os
os.environ['JOBLIB_TEMP_FOLDER'] = r'C:\temp_joblib'
if not os.path.exists(r'C:\temp_joblib'):
    os.makedirs(r'C:\temp_joblib')

import sys
import requests
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import subprocess
import json
import hashlib
import logging
import re
import traceback
from collections import deque
from datetime import timedelta, datetime
from scipy.stats import poisson
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEASONS = ['2223', '2324', '2425', '2526']
LEAGUES = ['E0', 'E1', 'E2', 'SP1', 'FR1', 'D1', 'I1', 'EC']
HISTORY_PATH = 'history.parquet'
import os as _os
_LOCAL_SCRIPT = r"C:\Users\曹亚楠\Desktop\worldcup-betting-analyst-skill\scripts\fetch_sporttery.py"
_CLOUD_SCRIPT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'scripts', 'fetch_sporttery.py')
SCRIPT_PATH = _LOCAL_SCRIPT if _os.path.exists(_LOCAL_SCRIPT) else _CLOUD_SCRIPT
PREDICTIONS_DIR = 'predictions'

_DF_HIST = None
_DF_HASH = None
_team_stats_precomputed = {}
_league_avg_ga = 1.3
_league_stats = {}

# ==================== 球队映射表（完整） ====================
TEAM_NAME_MAP = {
    'Arsenal': '阿森纳', 'Aston Villa': '阿斯顿维拉', 'Bournemouth': '伯恩茅斯',
    'Brentford': '布伦特福德', 'Brighton': '布莱顿', 'Burnley': '伯恩利',
    'Chelsea': '切尔西', 'Crystal Palace': '水晶宫', 'Everton': '埃弗顿',
    'Fulham': '富勒姆', 'Leeds United': '利兹联', 'Leicester City': '莱斯特城',
    'Liverpool': '利物浦', 'Manchester City': '曼彻斯特城', 'Manchester United': '曼彻斯特联',
    'Newcastle United': '纽卡斯尔联', 'Nottingham Forest': '诺丁汉森林',
    'Southampton': '南安普顿', 'Tottenham': '托特纳姆热刺',
    'West Ham United': '西汉姆联', 'Wolves': '狼队',
    'Wolverhampton Wanderers': '伍尔弗汉普顿',
    'Birmingham City': '伯明翰', 'Blackburn Rovers': '布莱克本', 'Blackpool': '布莱克浦',
    'Bristol City': '布里斯托尔城', 'Cardiff City': '加的夫城', 'Coventry City': '考文垂',
    'Derby County': '德比郡', 'Huddersfield Town': '哈德斯菲尔德', 'Hull City': '赫尔城',
    'Ipswich Town': '伊普斯维奇', 'Luton Town': '卢顿', 'Middlesbrough': '米德尔斯堡',
    'Millwall': '米尔沃尔', 'Norwich City': '诺维奇', 'Preston North End': '普雷斯顿',
    'Queens Park Rangers': '女王公园巡游者', 'Reading': '雷丁',
    'Rotherham United': '罗瑟勒姆', 'Sheffield United': '谢菲尔德联',
    'Sheffield Wednesday': '谢周三', 'Stoke City': '斯托克城', 'Sunderland': '桑德兰',
    'Swansea City': '斯旺西', 'Watford': '沃特福德',
    'West Bromwich Albion': '西布罗姆维奇', 'Wigan Athletic': '维冈竞技',
    'Accrington Stanley': '阿克灵顿', 'Barnsley': '巴恩斯利', 'Bolton Wanderers': '博尔顿',
    'Burton Albion': '伯顿', 'Cambridge United': '剑桥联', 'Charlton Athletic': '查尔顿',
    'Charlton': '查尔顿', 'Cheltenham Town': '切尔滕汉姆', 'Crewe Alexandra': '克鲁',
    'Doncaster Rovers': '唐卡斯特', 'Exeter City': '埃克塞特城', 'Fleetwood Town': '弗利特伍德',
    'Forest Green': '格林森林流浪者', 'Gillingham': '吉林汉姆', 'Leyton Orient': '莱顿东方',
    'Lincoln City': '林肯城', 'Mansfield': '曼斯菲尔德', 'Milton Keynes Dons': '米尔顿凯恩斯',
    'Morecambe': '莫克姆', 'Northampton Town': '北安普顿', 'Oxford United': '牛津联',
    'Peterborough United': '彼得堡联', 'Plymouth Argyle': '普利茅斯', 'Port Vale': '维尔港',
    'Portsmouth': '朴茨茅斯', 'Shrewsbury Town': '什鲁斯伯里', 'Stevenage': '斯蒂夫尼奇',
    'Stockport': '斯托克波特', 'Wycombe Wanderers': '韦康比流浪者',
    'Alaves': '阿拉维斯', 'Almeria': '阿尔梅里亚', 'Athletic Bilbao': '毕尔巴鄂竞技',
    'Atletico Madrid': '马德里竞技', 'Barcelona': '巴塞罗那', 'Real Betis': '皇家贝蒂斯',
    'Cadiz': '加的斯', 'Celta Vigo': '塞尔塔', 'Elche': '埃尔切', 'Espanyol': '西班牙人',
    'Getafe': '赫塔费', 'Girona': '赫罗纳', 'Granada': '格拉纳达',
    'Las Palmas': '拉斯帕尔马斯', 'Leganes': '莱加内斯', 'Levante': '莱万特',
    'Mallorca': '马略卡', 'Osasuna': '奥萨苏纳', 'Rayo Vallecano': '巴列卡诺',
    'Real Madrid': '皇家马德里', 'Real Sociedad': '皇家社会', 'Sevilla': '塞维利亚',
    'Valencia': '巴伦西亚', 'Valladolid': '巴拉多利德', 'Villarreal': '比利亚雷亚尔',
    'Deportivo La Coruña': '拉科鲁尼亚', 'Málaga': '马拉加',
    'Ajaccio': '阿雅克肖', 'Angers': '昂热', 'Auxerre': '欧塞尔', 'Brest': '布雷斯特',
    'Clermont': '克莱蒙', 'Le Havre': '勒阿弗尔', 'Lens': '朗斯', 'Lille': '里尔',
    'Lorient': '洛里昂', 'Lyon': '里昂', 'Marseille': '马赛', 'Metz': '梅斯',
    'Monaco': '摩纳哥', 'Montpellier': '蒙彼利埃', 'Nantes': '南特', 'Nice': '尼斯',
    'Paris Saint-Germain': '巴黎圣日耳曼', 'Reims': '兰斯', 'Rennes': '雷恩',
    'St Etienne': '圣埃蒂安', 'Strasbourg': '斯特拉斯堡', 'Toulouse': '图卢兹',
    'Troyes': '特鲁瓦', 'Paris FC': '巴黎FC', 'Dijon': '第戎',
    'Augsburg': '奥格斯堡', 'Bayern Munich': '拜仁慕尼黑', 'Bochum': '波鸿',
    'Borussia Dortmund': '多特蒙德', 'Darmstadt': '达姆施塔特',
    'Eintracht Frankfurt': '法兰克福', 'FC Koln': '科隆', 'Freiburg': '弗赖堡',
    'Heidenheim': '海登海姆', 'Hertha Berlin': '柏林赫塔', 'Hoffenheim': '霍芬海姆',
    'Holstein Kiel': '荷尔斯泰因基尔', 'Mainz': '美因茨',
    'Monchengladbach': '门兴格拉德巴赫', 'RB Leipzig': '莱比锡红牛',
    'Schalke 04': '沙尔克04', 'Stuttgart': '斯图加特', 'Union Berlin': '柏林联合',
    'Werder Bremen': '云达不来梅', 'Wolfsburg': '沃尔夫斯堡',
    'Hamburger SV': '汉堡', 'SC Paderborn': '帕德博恩', 'SV Elversberg': '埃尔沃斯堡',
    'Kaiserslautern': '凯泽斯劳滕', 'Karlsruher SC': '卡尔斯鲁厄',
    'Atalanta': '亚特兰大', 'Bologna': '博洛尼亚', 'Cagliari': '卡利亚里',
    'Como': '科莫', 'Empoli': '恩波利', 'Fiorentina': '佛罗伦萨', 'Genoa': '热那亚',
    'Hellas Verona': '维罗纳', 'Inter Milan': '国际米兰', 'Juventus': '尤文图斯',
    'Lazio': '拉齐奥', 'Lecce': '莱切', 'Milan': 'AC米兰', 'Monza': '蒙扎',
    'Napoli': '那不勒斯', 'Parma': '帕尔马', 'Roma': '罗马',
    'Salernitana': '萨勒尼塔纳', 'Sampdoria': '桑普多利亚', 'Sassuolo': '萨索洛',
    'Spezia': '斯佩齐亚', 'Torino': '都灵', 'Udinese': '乌迪内斯',
    'Venezia': '威尼斯', 'Frosinone': '弗洛西诺内',
    'Ajax': '阿贾克斯', 'PSV Eindhoven': 'PSV埃因霍温', 'Feyenoord': '费耶诺德',
    'AZ Alkmaar': '阿尔克马尔', 'FC Twente': '特温特', 'FC Utrecht': '乌德勒支',
    'Go Ahead Eagles': '前进之鹰', 'SC Cambuur': '坎布尔', 'ADO Den Haag': '海牙',
    'SC Telstar': '特尔斯达', 'FC Porto': '波尔图', 'Braga': '布拉加',
    'Famalicão': '法马利康', 'Gil Vicente': '吉维森特',
    'Vitória Guimarães': '吉马良斯', 'Estoril': '埃斯托里尔',
    'Académico de Viseu': '维塞乌', 'Benfica': '本菲卡',
    'Arouca': '阿罗卡', 'FC Arouca': '阿罗卡',
    'Sporting CP': '里斯本竞技', 'Sporting Lisbon': '里斯本竞技', 'Sporting': '里斯本竞技',
    'SBV Excelsior': 'SBV精英', 'Excelsior': 'SBV精英',
    'Willem II': '威廉二世',
    'PEC Zwolle': '兹沃勒', 'FC Zwolle': '兹沃勒',
    'Den Bosch': '登博思', 'FC Den Bosch': '登博思',
    'Helmond Sport': '海尔蒙特',
    'FC Groningen': '格罗宁根', 'Groningen': '格罗宁根',
    'NEC Nijmegen': '奈梅亨', 'NEC': '奈梅亨',
    'FK Bodø/Glimt': '博德闪耀', 'Bodø/Glimt': '博德闪耀', 'Rosenborg': '罗森博格',
    'Aalesund': '奥勒松', 'Djurgårdens IF': '佐加顿斯', 'IF Elfsborg': '埃尔夫斯堡',
    'AIK': 'AIK索尔纳', 'Malmö FF': '马尔默', 'IK Sirius': '天狼星',
    'Degerfors': '代格福什', 'Hammarby IF': '哈马比', 'Mjällby': '米亚尔比',
    'Viking': '维京', 'Viking FK': '维京',
    'Västerås SK': '韦斯特罗斯', 'Vasteras SK': '韦斯特罗斯',
    'GAIS': '哥德堡盖斯', 'GAIS Goteborg': '哥德堡盖斯', 'GAIS Göteborg': '哥德堡盖斯',
    'Sarpsborg 08': '萨尔普斯堡', 'Sarpsborg': '萨尔普斯堡',
    'KFUM Oslo': '奥斯陆KFUM', 'KFUM-Kameratene Oslo': '奥斯陆KFUM',
    'Kristiansund': '克里斯蒂安松', 'Kristiansund BK': '克里斯蒂安松',
    'Lillestrom': '利勒斯特罗姆', 'Lillestrøm': '利勒斯特罗姆',
    'HJK': '赫尔辛基火花', 'HJK Helsinki': '赫尔辛基', 'Helsinki': '赫尔辛基',
    'KuPS': '库奥皮奥', 'TPS Turku': 'TPS图尔库', 'FC Inter Turku': '国际图尔库',
    'Incheon United': '仁川联', 'Jeonbuk Hyundai Motors': '全北现代',
    'Daejeon Hana Citizen': '大田市民', 'Gimcheon Sangmu': '金泉尚武',
    'Jeju SK': '济州SK', 'Ulsan Hyundai': '蔚山现代',
    'FC Anyang': '安养FC', 'Anyang': '安养FC',
    'Kashiwa Reysol': '柏太阳神', 'Machida Zelvia': '町田泽维亚',
    'Vanraure Hachinohe': '八户南源', 'Kawasaki Frontale': '川崎前锋',
    'Cerezo Osaka': '大阪樱花', 'Tochigi City': '枥木城',
    'Kashima Antlers': '鹿岛鹿角', 'Mito HollyHock': '水户蜀葵',
    'Kyoto Sanga': '京都不死鸟', 'Kyoto Sanga FC': '京都不死鸟',
    'Gamba Osaka': '大阪钢巴',
    'Yokohama F. Marinos': '横滨水手', 'Yokohama F Marinos': '横滨水手',
    'Yokohama Marinos': '横滨水手',
    'V-Varen Nagasaki': '长崎航海', 'V Varen Nagasaki': '长崎航海',
    'Vissel Kobe': '神户胜利船',
    'Omiya Ardija': '大宫松鼠RB', 'RB Omiya Ardija': '大宫松鼠RB',
    'Fujieda MYFC': '藤枝MYFC',
    'Botafogo': '博塔弗戈', 'Flamengo': '弗拉门戈',
    'Mirassol': '米拉索尔', 'Santos': '桑托斯', 'Gremio': '格雷米奥',
    'Internacional': '巴西国际', 'Palmeiras': '帕尔梅拉斯',
    'Corinthians': '科林蒂安', 'Fluminense': '弗鲁米嫩塞',
    'Bahia': '巴伊亚', 'EC Bahia': '巴伊亚',
    'Athletico Paranaense': '巴拉纳竞技', 'Atletico Paranaense': '巴拉纳竞技',
    'Athletico-PR': '巴拉纳竞技',
    'LDU Quito': '基多体育大学', 'Liga Deportiva Universitaria': '基多体育大学',
    'Independiente del Valle': '德尔瓦耶独立', 'Independiente DV': '德尔瓦耶独立',
    'D.C. United': '华盛顿联', 'Los Angeles FC': '洛杉矶FC',
    'St. Louis City SC': '圣路易斯城', 'FC Dallas': '达拉斯FC',
    'New York City FC': '纽约城', 'New York City': '纽约城', 'NYCFC': '纽约城',
    'New York Red Bulls': '纽约红牛', 'NY Red Bulls': '纽约红牛',
    'Al Hilal': '利雅得新月', 'Al-Ahli': '吉达国民',
    'Al-Diriyah': '迪里耶', 'Al-Diraiyah': '迪里耶',
    'Al-Qadsiah': '胡巴尔卡德西亚', 'Al Qadsiah': '胡巴尔卡德西亚',
    'Al-Fayha': '迈季迈阿宽广', 'Al Fayha': '迈季迈阿宽广',
    'Al-Kholood': '拉斯永恒', 'Al Kholood': '拉斯永恒',
    'NEOM SC': '新未来SC', 'NEOM Sports Club': '新未来SC',
    'Al-Khaleej': '赛哈特海湾', 'Al-Nassr': '利雅得胜利', 'Al Nassr': '利雅得胜利',
    'Al Ain FC': '阿布扎比艾因', 'Al Ain': '阿布扎比艾因',
    'CA Tigre BA': '蒂格雷', 'Tigre': '蒂格雷',
    'Independiente Rivadavia': '里瓦达维亚独立',
    'Atlético Huracán': '胡拉坎', 'Atletico Huracan': '胡拉坎', 'Huracán': '胡拉坎',
    'Rosario Central': '罗萨里奥中央', 'Instituto de Córdoba': '科尔多瓦学院',
    'Instituto': '科尔多瓦学院', 'River Plate': '河床',
    'Gimnasia La Plata': '拉普拉塔体操', 'Banfield': '班菲尔德',
    'Gimnasia Mendoza': '门多萨体操', 'Lanus': '拉努斯', 'Lanús': '拉努斯',
    'Talleres': '塔勒雷斯', 'Talleres de Córdoba': '塔勒雷斯',
    'Platense': '普拉滕斯', 'Estudiantes de Río Cuarto': '里奥夸尔托学生',
    'Argentinos Juniors': '阿根廷青年', 'Independiente': '独立',
    'Newells Old Boys': '纽维尔老男孩', "Newell's Old Boys": '纽维尔老男孩',
    'Deportivo Riestra': '里斯特拉', 'Estudiantes': '拉普拉塔学生',
    'Estudiantes de La Plata': '拉普拉塔大学生',
    'Racing Club': '竞赛', 'San Lorenzo': '圣洛伦索',
    'Aldosivi Mar del Plata': '阿尔多西维', 'Aldosivi': '阿尔多西维',
    'Defensa y Justicia': '国防与司法', 'Union Santa Fe': '圣菲联合',
    'Unión Santa Fe': '圣菲联合', 'Central Córdoba': '中央科尔多瓦',
    'Central Cordoba': '中央科尔多瓦', 'Velez Sarsfield BA': '萨斯菲尔德',
    'Vélez Sarsfield': '萨斯菲尔德', 'Velez Sarsfield': '萨斯菲尔德',
    'Sarmiento de Junin': '萨米恩托', 'Sarmiento': '萨米恩托',
    'Atlético Tucuman': '图库曼竞技', 'Atletico Tucuman': '图库曼竞技',
    'Belgrano de Cordoba': '贝尔格拉诺', 'Belgrano': '贝尔格拉诺',
    'Boca Juniors': '博卡青年', 'Barracas Central': '巴拉卡斯中央',
    'Shanghai Port': '上海海港', 'Shanghai SIPG': '上海海港',
    'Shanghai Shenhua': '上海申花', 'Beijing Guoan': '北京国安',
    'Pohang Steelers': '浦项制铁',
    'South Korea U23': '韩国亚运男足', 'Korea Republic U23': '韩国亚运男足',
    'Qatar U23': '卡塔尔亚足', 'Qatar Olympic': '卡塔尔亚足',
    "Johor Darul Ta'zim": '柔佛', 'Johor DT': '柔佛', 'JDT': '柔佛',
    'Buriram United': '布里兰', 'Buriram': '布里兰',
    'Ratchaburi': '叻武里', 'Ratchaburi FC': '叻武里',
    'China U23': '中国亚运男足', 'China Olympic': '中国亚运男足',
    'China PR U23': '中国亚运男足',
    'North Korea U23': '朝鲜亚运男足', 'DPR Korea U23': '朝鲜亚运男足',
    'Korea DPR U23': '朝鲜亚运男足',
    'Hong Kong U23': '中国香港亚运男足', 'Hong Kong, China U23': '中国香港亚运男足',
    'Japan U23': '日本亚足', 'Japan Olympic': '日本亚足',
    'Kyrgyzstan U23': '吉尔吉斯斯坦亚足', 'Kyrgyzstan Olympic': '吉尔吉斯斯坦亚足',
    'Iran U23': '伊朗亚运男足', 'Iran Olympic': '伊朗亚运男足',
    'China Women': '中国女足', 'China PR Women': '中国女足', 'China W': '中国女足',
    'Uzbekistan Women': '乌兹别克斯坦女足', 'Uzbekistan W': '乌兹别克斯坦女足',
    'Tampines Rovers': '淡宾尼士流浪', 'Tampines': '淡宾尼士流浪',
    'Saudi Arabia U23': '沙特阿拉伯亚足', 'Saudi Arabia Olympic': '沙特阿拉伯亚足',
    'Anderlecht': '安德莱赫特', 'RSC Anderlecht': '安德莱赫特',
    'Celje': '采列', 'NK Celje': '采列',
    'OFI Crete': '克里特', 'OFI': '克里特',
    'Lech Poznan': '波兹南莱赫', 'Lech Poznań': '波兹南莱赫',
    'Besiktas': '贝西克塔斯', 'Beşiktaş': '贝西克塔斯',
    'Sturm Graz': '格拉茨风暴', 'SK Sturm Graz': '格拉茨风暴',
    'Omonia Nicosia': '奥莫尼亚', 'Omonia': '奥莫尼亚',
    'Racing Santander': '桑坦德竞技',
    'Torino Women': '托林斯', 'Torino Femminile': '托林斯',
    'Osnabrück': '奥斯纳布吕克', 'Osnabruck': '奥斯纳布吕克',
    'Accrington': '阿克灵顿', 'Ath Bilbao': '毕尔巴鄂竞技', 'Ath Madrid': '马德里竞技',
    'Betis': '皇家贝蒂斯', 'Birmingham': '伯明翰', 'Blackburn': '布莱克本',
    'Bolton': '博尔顿', 'Bristol Rvs': '布里斯托尔流浪者', 'Burton': '伯顿',
    'Cambridge': '剑桥联', 'Cardiff': '加的夫城', 'Celta': '塞尔塔',
    'Cheltenham': '切尔滕汉姆', 'Coventry': '考文垂', 'Derby': '德比郡',
    'Dortmund': '多特蒙德', 'Ein Frankfurt': '法兰克福', 'Espanol': '西班牙人',
    'Exeter': '埃克塞特城', 'Hertha': '柏林赫塔', 'Huddersfield': '哈德斯菲尔德',
    'Hull': '赫尔城', 'Ipswich': '伊普斯维奇', 'Leeds': '利兹联',
    'Leicester': '莱斯特城', 'Leverkusen': '勒沃库森', 'Lincoln': '林肯城',
    'Luton': '卢顿', "M'gladbach": '门兴格拉德巴赫', 'Man City': '曼彻斯特城',
    'Man United': '曼彻斯特联', 'Newcastle': '纽卡斯尔联',
    'Northampton': '北安普顿', 'Norwich': '诺维奇',
    "Nott'm Forest": '诺丁汉森林', 'Oxford': '牛津联',
    'Paris SG': '巴黎圣日耳曼', 'Peterboro': '彼得堡联', 'Plymouth': '普利茅斯',
    'Preston': '普雷斯顿', 'QPR': '女王公园巡游者', 'Rotherham': '罗瑟勒姆',
    'Sheffield Weds': '谢周三', 'Shrewsbury': '什鲁斯伯里', 'Sociedad': '皇家社会',
    'Stoke': '斯托克城', 'Swansea': '斯旺西', 'Vallecano': '巴列卡诺',
    'West Brom': '西布罗姆维奇', 'West Ham': '西汉姆联', 'Wigan': '维冈竞技',
    'Wycombe': '韦康比流浪者', 'Inter': '国际米兰', 'Milan': 'AC米兰',
    'Celta Vigo': '维戈塞尔塔', 'Wrexham': '雷克斯汉姆',
    'Paris Saint Germain': '巴黎圣日耳曼', 'Manchester Utd': '曼彻斯特联',
    'Newcastle Utd': '纽卡斯尔联', 'Nottm Forest': '诺丁汉森林',
    'Tottenham Hotspur': '托特纳姆热刺', 'Spurs': '托特纳姆热刺',
    'Brighton and Hove Albion': '布莱顿', 'Atlético Madrid': '马德里竞技',
    'Athletic Club': '毕尔巴鄂竞技', 'Bayern München': '拜仁慕尼黑',
    'Bayer Leverkusen': '勒沃库森', 'Borussia Mönchengladbach': '门兴格拉德巴赫',
    'VfB Stuttgart': '斯图加特', 'VfL Wolfsburg': '沃尔夫斯堡',
    'Internazionale': '国际米兰', 'AC Milan': 'AC Milan', 'AS Roma': '罗马',
    'SS Lazio': '拉齐奥', 'SSC Napoli': '那不勒斯', 'PSG': '巴黎圣日耳曼',
    'Wolverhampton': '狼队', 'Nottingham': '诺丁汉森林',
        # 美职联
    'Real Salt Lake': '皇家盐湖城',
    'Seattle Sounders': '西雅图海湾人',
    'Seattle Sounders FC': '西雅图海湾人',

    # 亚运男足
    'Thailand U23': '泰国亚运男足',
    'Thailand Olympic': '泰国亚运男足',

    # 国家队（欧洲）
    'Wales': '威尔士',
    'Portugal': '葡萄牙',
    'Greece': '希腊',
    'Serbia': '塞尔维亚',
    'Denmark': '丹麦',
    'Germany': '德国',
    'Norway': '挪威',
    'Kosovo': '科索沃',
    'Ireland': '爱尔兰',
    'Republic of Ireland': '爱尔兰',
    'Netherlands': '荷兰',
    'Holland': '荷兰',

    # 国家队（亚洲/美洲）
    'South Korea': '韩国',
    'Korea Republic': '韩国',
    'Korea': '韩国',
    'Japan': '日本',
    'China': '中国',
    'China PR': '中国',
    'Maldives': '马尔代夫',
    'Ecuador': '厄瓜多尔',
    'Uruguay': '乌拉圭',
}

TEAM_NAME_MAP_REVERSE = {v: k for k, v in TEAM_NAME_MAP.items()}
TEAM_NAME_MAP_REVERSE['赫尔辛基'] = 'HJK Helsinki'
TEAM_NAME_MAP_REVERSE['赫尔辛基火花'] = 'HJK'
TEAM_NAME_MAP_REVERSE['拉普拉塔大学生'] = 'Estudiantes de La Plata'
TEAM_NAME_MAP_REVERSE['巴黎圣日尔曼'] = 'Paris Saint-Germain'
TEAM_NAME_MAP_REVERSE['巴黎圣日耳曼'] = 'Paris Saint-Germain'
TEAM_NAME_MAP_REVERSE['阿罗卡'] = 'Arouca'
TEAM_NAME_MAP_REVERSE['巴伊亚'] = 'Bahia'
TEAM_NAME_MAP_REVERSE['SBV精英'] = 'SBV Excelsior'
TEAM_NAME_MAP_REVERSE['韦斯特罗斯'] = 'Västerås SK'
TEAM_NAME_MAP_REVERSE['大阪钢巴'] = 'Gamba Osaka'
TEAM_NAME_MAP_REVERSE['横滨水手'] = 'Yokohama F. Marinos'
TEAM_NAME_MAP_REVERSE['长崎航海'] = 'V-Varen Nagasaki'
TEAM_NAME_MAP_REVERSE['里斯本竞技'] = 'Sporting CP'
TEAM_NAME_MAP_REVERSE['神户胜利船'] = 'Vissel Kobe'
TEAM_NAME_MAP_REVERSE['吉尔吉斯斯坦亚足'] = 'Kyrgyzstan U23'
TEAM_NAME_MAP_REVERSE['巴拉纳竞技'] = 'Athletico Paranaense'
TEAM_NAME_MAP_REVERSE['大宫松鼠RB'] = 'Omiya Ardija'
TEAM_NAME_MAP_REVERSE['克里斯蒂安松'] = 'Kristiansund'
TEAM_NAME_MAP_REVERSE['伊朗亚运男足'] = 'Iran U23'
TEAM_NAME_MAP_REVERSE['安养FC'] = 'FC Anyang'
TEAM_NAME_MAP_REVERSE['藤枝MYFC'] = 'Fujieda MYFC'
TEAM_NAME_MAP_REVERSE['哥德堡盖斯'] = 'GAIS'
TEAM_NAME_MAP_REVERSE['皇家盐湖城'] = 'Real Salt Lake'
TEAM_NAME_MAP_REVERSE['西雅图海湾人'] = 'Seattle Sounders'
TEAM_NAME_MAP_REVERSE['泰国亚运男足'] = 'Thailand U23'
TEAM_NAME_MAP_REVERSE['威尔士'] = 'Wales'
TEAM_NAME_MAP_REVERSE['葡萄牙'] = 'Portugal'
TEAM_NAME_MAP_REVERSE['希腊'] = 'Greece'
TEAM_NAME_MAP_REVERSE['塞尔维亚'] = 'Serbia'
TEAM_NAME_MAP_REVERSE['丹麦'] = 'Denmark'
TEAM_NAME_MAP_REVERSE['德国'] = 'Germany'
TEAM_NAME_MAP_REVERSE['挪威'] = 'Norway'
TEAM_NAME_MAP_REVERSE['科索沃'] = 'Kosovo'
TEAM_NAME_MAP_REVERSE['爱尔兰'] = 'Republic of Ireland'
TEAM_NAME_MAP_REVERSE['荷兰'] = 'Netherlands'
TEAM_NAME_MAP_REVERSE['韩国'] = 'South Korea'
TEAM_NAME_MAP_REVERSE['日本'] = 'Japan'
TEAM_NAME_MAP_REVERSE['中国'] = 'China'
TEAM_NAME_MAP_REVERSE['马尔代夫'] = 'Maldives'
TEAM_NAME_MAP_REVERSE['厄瓜多尔'] = 'Ecuador'
TEAM_NAME_MAP_REVERSE['乌拉圭'] = 'Uruguay'


def translate_team_name(en_name):
    if not en_name:
        return en_name
    if en_name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[en_name]
    en_lower = en_name.lower().strip()
    for k, v in TEAM_NAME_MAP.items():
        if k.lower().strip() == en_lower:
            return v
    en_clean = re.sub(r'[^a-z]', '', en_lower)
    for k, v in TEAM_NAME_MAP.items():
        k_clean = re.sub(r'[^a-z]', '', k.lower())
        if k_clean == en_clean and len(k_clean) >= 4:
            return v
    for k, v in TEAM_NAME_MAP.items():
        k_lower = k.lower().strip()
        if len(en_lower) >= 5 and len(k_lower) >= 5:
            if en_lower in k_lower or k_lower in en_lower:
                return v
    return en_name
def safe_format_team(x):
    """安全格式化球队名，避免 NaN/None 报错"""
    if not isinstance(x, str) or not x.strip():
        return "（未知）"
    try:
        cn = translate_team_name(x)
        return f"{cn} ({x})"
    except Exception:
        return x

def get_df_hash(df):
    return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()


@st.cache_data(ttl=3600)
@st.cache_data(ttl=3600)
def load_history():
    global _DF_HIST, _DF_HASH
    QXC_HISTORY_PATH = 'qxc_history.parquet'

    if os.path.exists(HISTORY_PATH):
        try:
            df = pd.read_parquet(HISTORY_PATH)
            logger.info(f"从 {HISTORY_PATH} 加载数据成功，行数: {len(df)}")
        except Exception as e:
            logger.warning(f"Parquet 加载失败，尝试 pickle: {e}")
            with open(HISTORY_PATH.replace('.parquet', '.pkl'), 'rb') as f:
                df = pickle.load(f)

        # ========== 合并球小策扩展历史库 ==========
        if os.path.exists(QXC_HISTORY_PATH):
            try:
                qxc_df = pd.read_parquet(QXC_HISTORY_PATH)
                # 只保留主库需要的列
                common_cols = [c for c in df.columns if c in qxc_df.columns]
                qxc_sub = qxc_df[common_cols].copy()
                # 合并去重
                df = pd.concat([df, qxc_sub], ignore_index=True)
                df = df.drop_duplicates(subset=['date', 'hometeam', 'awayteam'], keep='first')
                df = df.sort_values('date').reset_index(drop=True)
                logger.info(f"已合并球小策数据，总比赛数：{len(df)}")
            except Exception as e:
                logger.warning(f"合并球小策数据失败: {e}")
        # ============================================

    else:
        logger.info("本地无数据，开始下载...")
        df = download_data()
        for col in df.columns:
            if df[col].dtype == object:
                converted = pd.to_numeric(df[col], errors='coerce')
                if converted.notna().mean() > 0.8:
                    df[col] = converted
        # 如果本地没有历史库，也把球小策数据合并进来
        if os.path.exists(QXC_HISTORY_PATH):
            try:
                qxc_df = pd.read_parquet(QXC_HISTORY_PATH)
                common_cols = [c for c in df.columns if c in qxc_df.columns]
                qxc_sub = qxc_df[common_cols].copy()
                df = pd.concat([df, qxc_sub], ignore_index=True)
                df = df.drop_duplicates(subset=['date', 'hometeam', 'awayteam'], keep='first')
                df = df.sort_values('date').reset_index(drop=True)
                logger.info(f"已合并球小策数据，总比赛数：{len(df)}")
            except Exception as e:
                logger.warning(f"合并球小策数据失败: {e}")
        df = precompute_all_features(df)
        df.to_parquet(HISTORY_PATH, index=False)
        logger.info("数据保存至 Parquet")

    if df is None or len(df) < 100:
        return None

    if ('elo_diff' not in df.columns or 'diff' not in df.columns
            or 'rank_points' not in df.columns or 'style' not in df.columns
            or 'league' not in df.columns):
        df = precompute_all_features(df)
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
                df['league'] = league
                all_data.append(df)
                logger.info(f"✅ {league} {season}")
            except Exception as e:
                logger.warning(f"⏳ {league} {season} 失败: {e}")
    if not all_data:
        raise Exception("没有下载到任何数据")
    df = pd.concat(all_data, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def initialize_elo(df):
    teams = pd.concat([df['hometeam'], df['awayteam']]).unique()
    return {team: 1500 for team in teams}


def update_elo(home, away, home_score, away_score, elo_dict, k=25):
    r_h = elo_dict[home]
    r_a = elo_dict[away]
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


def precompute_all_features(df):
    # ========== 数据清洗 ==========
    # 过滤比分缺失的比赛
    df = df.dropna(subset=['fthg', 'ftag', 'hometeam', 'awayteam']).copy()
    df['fthg'] = df['fthg'].astype(int)
    df['ftag'] = df['ftag'].astype(int)
    # 补齐 ftr（如果缺失）
    if 'ftr' not in df.columns:
        df['ftr'] = df.apply(
            lambda r: 'H' if r['fthg'] > r['ftag'] else ('D' if r['fthg'] == r['ftag'] else 'A'),
            axis=1
        )
    else:
        # 有些行的 ftr 是 NaN，用比分补
        mask = df['ftr'].isna()
        if mask.any():
            df.loc[mask, 'ftr'] = df.loc[mask].apply(
                lambda r: 'H' if r['fthg'] > r['ftag'] else ('D' if r['fthg'] == r['ftag'] else 'A'),
                axis=1
            )
    # ==============================

    df = df.sort_values('date').reset_index(drop=True)
    elo_dict = initialize_elo(df)
    elo_diffs = []
    for _, row in df.iterrows():
        home, away = row['hometeam'], row['awayteam']
        elo_diffs.append(elo_dict[home] - elo_dict[away])
        elo_dict = update_elo(home, away, row['fthg'], row['ftag'], elo_dict)
    df['elo_diff'] = elo_diffs

    team_history = {}
    strength_diffs = []
    rank_points_list = []
    points = {}
    for _, row in df.iterrows():
        home, away = row['hometeam'], row['awayteam']
        h_key = f"{home}_home"
        a_key = f"{away}_away"
        h_hist = team_history.get(h_key, deque(maxlen=10))
        a_hist = team_history.get(a_key, deque(maxlen=10))

        def calc_stats(hist):
            if len(hist) == 0:
                return 1.2, 1.2, 0.3
            return (np.mean([x[0] for x in hist]),
                    np.mean([x[1] for x in hist]),
                    np.mean([x[2] for x in hist]))

        h_gf, h_ga, h_wr = calc_stats(h_hist)
        a_gf, a_ga, a_wr = calc_stats(a_hist)
        h_str = (h_gf * 0.5 - h_ga * 0.3 + h_wr * 0.2)
        a_str = (a_gf * 0.5 - a_ga * 0.3 + a_wr * 0.2)
        strength_diffs.append(h_str - a_str)
        rank_points_list.append(points.get(home, 0) - points.get(away, 0))

        if row['fthg'] > row['ftag']:
            points[home] = points.get(home, 0) + 3
        elif row['fthg'] == row['ftag']:
            points[home] = points.get(home, 0) + 1
            points[away] = points.get(away, 0) + 1
        else:
            points[away] = points.get(away, 0) + 3

        h_hist.append((row['fthg'], row['ftag'], 1 if row['ftr'] == 'H' else 0))
        a_hist.append((row['ftag'], row['fthg'], 1 if row['ftr'] == 'A' else 0))
        team_history[h_key] = h_hist
        team_history[a_key] = a_hist

    df['diff'] = strength_diffs
    df['rank_points'] = rank_points_list
    style_map = compute_team_style(df)
    df['style'] = df['hometeam'].map(style_map)
    return df


def precompute_strength_diff(df):
    return precompute_all_features(df)


def compute_team_style(df):
    teams = pd.concat([df['hometeam'], df['awayteam']]).unique()
    style_data = []
    valid_teams = []

    for team in teams:
        games = df[(df['hometeam'] == team) | (df['awayteam'] == team)]
        # 过滤掉比分缺失的比赛
        games = games.dropna(subset=['fthg', 'ftag'])
        if len(games) == 0:
            continue
        avg_goals = games['fthg'].mean()
        avg_ga = games['ftag'].mean()
        # 再检查一次，确保不是 NaN
        if pd.isna(avg_goals) or pd.isna(avg_ga):
            continue
        style_data.append([avg_goals, avg_ga])
        valid_teams.append(team)

    if len(style_data) == 0:
        return {team: 0 for team in teams}

    style_array = np.array(style_data, dtype=float)
    # 二次防御：如果还有 NaN 或 inf，用 1.2 填充
    style_array = np.nan_to_num(style_array, nan=1.2, posinf=1.2, neginf=0.0)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(style_array)
    result = {team: 0 for team in teams}
    for i, team in enumerate(valid_teams):
        result[team] = int(kmeans.labels_[i])
    return result

def force_update_data():
    if os.path.exists(HISTORY_PATH):
        os.remove(HISTORY_PATH)
    pkl_path = HISTORY_PATH.replace('.parquet', '.pkl')
    if os.path.exists(pkl_path):
        os.remove(pkl_path)
    st.cache_data.clear()
    return load_history()


def _calc_team_stats(data, team, max_games=15):
    """计算一组比赛的统计（对手强度加权 + 近 3 场额外加权）"""
    if len(data) == 0:
        return {
            'gf': 1.2, 'ga': 1.2, 'wr': 0.3, 'gf_std': 0.5,
            'gd': 0.0, 'sample_size': 0, 'opp_elo_avg': 1500.0
        }
    data = data.tail(max_games)
    gf_list, ga_list, wr_list, weights, gd_list, opp_elos = [], [], [], [], [], []
    total_n = len(data)
    for idx, (_, r) in enumerate(data.iterrows()):
        elo_diff = r.get('elo_diff', 0)
        if pd.isna(elo_diff):
            elo_diff = 0
        if r['hometeam'] == team:
            gf, ga = r['fthg'], r['ftag']
            is_win = 1 if r['ftr'] == 'H' else 0
            opp_strength = -elo_diff
        else:
            gf, ga = r['ftag'], r['fthg']
            is_win = 1 if r['ftr'] == 'A' else 0
            opp_strength = elo_diff
        w = max(0.5, min(2.5, 1 + opp_strength / 400))
        # 近 3 场额外加权 15%
        if idx >= total_n - 3:
            w *= 1.15
        gf_list.append(gf); ga_list.append(ga); gd_list.append(gf - ga)
        wr_list.append(is_win); weights.append(w)
        opp_elos.append(1500 - opp_strength)
    weights = np.array(weights)
    wnorm = weights / weights.sum()
    return {
        'gf': float(np.sum(np.array(gf_list) * wnorm)),
        'ga': float(np.sum(np.array(ga_list) * wnorm)),
        'gd': float(np.sum(np.array(gd_list) * wnorm)),
        'wr': float(np.mean(wr_list)),
        'gf_std': float(np.std(gf_list, ddof=1)) if len(gf_list) > 1 else 0.5,
        'sample_size': total_n,
        'opp_elo_avg': float(np.mean(opp_elos))
    }


def build_team_stats_table(df):
    """跨赛季实力评估：上赛季基线 + 本赛季状态，动态加权"""
    global _team_stats_precomputed, _league_avg_ga, _league_stats
    _team_stats_precomputed = {}
    _league_stats = {}
    if df is None or len(df) == 0:
        return

    if 'league' in df.columns:
        for lg in df['league'].dropna().unique():
            sub = df[df['league'] == lg]
            _league_stats[lg] = {
                'avg_goals': (sub['fthg'].mean() + sub['ftag'].mean()) / 2,
                'avg_ga': sub['ftag'].mean()
            }
    _league_avg_ga = (df['ftag'].mean() + df['fthg'].mean()) / 2
    if _league_avg_ga <= 0:
        _league_avg_ga = 1.3

    df_sorted = df.sort_values('date')
    max_date = df_sorted['date'].max()
    current_season_start = max_date - timedelta(days=300)
    last_season_start = current_season_start - timedelta(days=400)

    all_teams = pd.concat([df['hometeam'], df['awayteam']]).unique()

    for team in all_teams:
        team_games = df_sorted[(df_sorted['hometeam'] == team) | (df_sorted['awayteam'] == team)]
        current = team_games[team_games['date'] >= current_season_start]
        last = team_games[(team_games['date'] >= last_season_start) &
                          (team_games['date'] < current_season_start)]

        for venue in ['home', 'away', 'all']:
            if venue == 'home':
                cur_data = current[current['hometeam'] == team]
                last_data = last[last['hometeam'] == team]
            elif venue == 'away':
                cur_data = current[current['awayteam'] == team]
                last_data = last[last['awayteam'] == team]
            else:
                cur_data = current
                last_data = last

            cur_stats = _calc_team_stats(cur_data, team, max_games=15)
            last_stats = _calc_team_stats(last_data, team, max_games=38)

            cur_n = cur_stats['sample_size']
            last_n = last_stats['sample_size']

            if last_n < 5:
                weight_cur = 1.0 if cur_n >= 3 else 0.0
            else:
                weight_cur = min(0.9, 0.4 + cur_n * 0.033)
            weight_last = 1.0 - weight_cur

            if cur_n == 0 and last_n == 0:
                final = {
                    'gf': 1.2, 'ga': 1.2, 'wr': 0.3, 'gf_std': 0.5,
                    'gd': 0.0, 'sample_size': 0, 'strength': 0.5,
                    'form': 0.0, 'opp_elo_avg': 1500.0,
                    'cur_sample': 0, 'last_sample': 0
                }
            else:
                gf = cur_stats['gf'] * weight_cur + last_stats['gf'] * weight_last
                ga = cur_stats['ga'] * weight_cur + last_stats['ga'] * weight_last
                wr = cur_stats['wr'] * weight_cur + last_stats['wr'] * weight_last
                gd = cur_stats['gd'] * weight_cur + last_stats['gd'] * weight_last
                gf_std = cur_stats['gf_std'] if cur_n >= 5 else last_stats['gf_std']
                opp_elo = cur_stats['opp_elo_avg'] * weight_cur + last_stats['opp_elo_avg'] * weight_last
                sample = cur_n + last_n

                gf_norm = min(gf / 2.5, 1.0)
                ga_norm = max(0, 1 - ga / 2.5)
                gd_norm = max(0, min((gd + 1.5) / 3.0, 1.0))
                opp_quality = max(0, min((opp_elo - 1400) / 300, 1.0))
                strength = (gf_norm * 0.30 + ga_norm * 0.20 + gd_norm * 0.20 +
                            wr * 0.20 + opp_quality * 0.10)

                recent5 = cur_data.tail(5) if cur_n >= 5 else last_data.tail(5)
                if len(recent5) >= 3:
                    r_gd = []
                    for _, r in recent5.iterrows():
                        if r['hometeam'] == team:
                            r_gd.append(r['fthg'] - r['ftag'])
                        else:
                            r_gd.append(r['ftag'] - r['fthg'])
                    form = float(np.mean(r_gd)) - gd
                else:
                    form = 0.0

                final = {
                    'gf': gf, 'ga': ga, 'wr': wr, 'gf_std': gf_std,
                    'gd': gd, 'sample_size': sample, 'strength': strength,
                    'form': form, 'opp_elo_avg': opp_elo,
                    'cur_sample': cur_n, 'last_sample': last_n
                }
            _team_stats_precomputed[(team, venue)] = final
    logger.info(f"已预计算 {len(all_teams)} 支球队（跨赛季加权）")


def get_team_stats(team, df_history, date_limit, lookback=10, venue='home'):
    key = (team, venue)
    stats = _team_stats_precomputed.get(key)
    if stats and stats.get('sample_size', 0) >= 5:
        return stats
    if venue != 'all':
        stats_all = _team_stats_precomputed.get((team, 'all'))
        if stats_all and stats_all.get('sample_size', 0) >= 5:
            return stats_all
    return {'gf': 1.2, 'ga': 1.2, 'wr': 0.3, 'gf_std': 0.5,
            'gd': 0.0, 'sample_size': 0, 'strength': 0.5,
            'form': 0.0, 'opp_elo_avg': 1500.0}


@st.cache_data(ttl=3600)
def get_head_to_head(home, away, date_limit_iso, df_hash):
    global _DF_HIST
    df = _DF_HIST
    if df is None:
        return 0.5
    date_limit = pd.to_datetime(date_limit_iso)
    h2h = df[((df['hometeam'] == home) & (df['awayteam'] == away)) |
             ((df['hometeam'] == away) & (df['awayteam'] == home))]
    h2h = h2h[h2h['date'] < date_limit].sort_values('date').tail(5)
    if len(h2h) == 0:
        return 0.5
    home_wins = 0
    for _, r in h2h.iterrows():
        if r['hometeam'] == home:
            if r['fthg'] > r['ftag']:
                home_wins += 1
        else:
            if r['ftag'] > r['fthg']:
                home_wins += 1
    return home_wins / len(h2h)


def get_trend(team, df_history, date_limit):
    return get_team_stats(team, df_history, date_limit).get('form', 0.0)


@st.cache_data(ttl=3600)
def get_team_dynamic_over_rate_cached(team, date_limit_iso, df_hash):
    global _DF_HIST
    df = _DF_HIST
    if df is None:
        return 0.4
    date_limit = pd.to_datetime(date_limit_iso)
    data = df[(df['hometeam'] == team) | (df['awayteam'] == team)]
    data = data[data['date'] < date_limit].sort_values('date').tail(10)
    if len(data) == 0:
        return 0.4
    total_goals = [r['fthg'] + r['ftag'] for _, r in data.iterrows()]
    return sum(1 for tg in total_goals if tg >= 3) / len(total_goals)


def get_team_dynamic_over_rate(team, df_history, date_limit, lookback_short=5, lookback_long=10):
    return get_team_dynamic_over_rate_cached(team, date_limit.isoformat(), _DF_HASH)


def days_since_last_match(team, df_hist, date_limit):
    if df_hist is None:
        return 7
    data = df_hist[(df_hist['hometeam'] == team) | (df_hist['awayteam'] == team)]
    data = data[data['date'] < date_limit].sort_values('date')
    if len(data) == 0:
        return 7
    days = (pd.Timestamp(date_limit) - data.iloc[-1]['date']).days
    return max(0, min(30, days))


def rest_factor(days):
    if days < 2:
        return 0.88
    elif days < 3:
        return 0.93
    elif days <= 5:
        return 1.00
    elif days <= 8:
        return 1.02
    elif days <= 14:
        return 0.99
    else:
        return 0.95


def get_opponent_defense_factor(team, df_history, date_limit, lookback=10):
    stats = get_team_stats(team, df_history, date_limit, venue='all')
    avg_ga = stats['ga']
    if avg_ga > 0 and _league_avg_ga > 0:
        return _league_avg_ga / avg_ga
    return 1.0


def get_league_dynamic_threshold(df_hist, date_limit, lookback_days=365, offset=0.0):
    if df_hist is None:
        return 0.45 + offset
    df_sub = df_hist[df_hist['date'] >= date_limit - timedelta(days=lookback_days)]
    if len(df_sub) == 0:
        return 0.45 + offset
    total_goals = df_sub['fthg'] + df_sub['ftag']
    over_rate = (total_goals >= 3).mean()
    return max(0.35, min(0.55, over_rate * 1.0 + offset))


# ==================== XGBoost ====================
def train_xgb_classifier(df):
    cutoff = df['date'].max() - timedelta(days=365 * 2)
    train_df = df[df['date'] > cutoff].copy()
    if len(train_df) < 100:
        return None
    if len(train_df) > 8000:
        train_df = train_df.sample(n=8000, random_state=42)
    features = []
    labels = []
    for _, row in train_df.iterrows():
        home = row['hometeam']; away = row['awayteam']; date = row['date']
        h_stats = get_team_stats(home, df, date, venue='home')
        a_stats = get_team_stats(away, df, date, venue='away')
        h2h = get_head_to_head(home, away, date.isoformat(), _DF_HASH)
        rank_diff = row.get('rank_points', 0)
        elo_h = row.get('elo_diff', 0)
        diff_h = row.get('diff', 0)
        features.append([
            h_stats['gf'], h_stats['ga'], h_stats['wr'], h_stats['gf_std'],
            a_stats['gf'], a_stats['ga'], a_stats['wr'], a_stats['gf_std'],
            elo_h, diff_h, rank_diff, h2h
        ])
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
    model = xgb.XGBClassifier(
        objective='multi:softprob', max_depth=4, learning_rate=0.1,
        n_estimators=100, random_state=42, n_jobs=1,
        tree_method='hist', subsample=0.8, colsample_bytree=0.8
    )
    model.fit(X, y)
    return model


def predict_xgb(home, away, df_hist, xgb_model):
    if xgb_model is None:
        return None
    now = pd.Timestamp.now()
    h_stats = get_team_stats(home, df_hist, now, venue='home')
    a_stats = get_team_stats(away, df_hist, now, venue='away')
    h2h = get_head_to_head(home, away, now.isoformat(), _DF_HASH)
    rank_diff = df_hist[(df_hist['hometeam'] == home)]['rank_points'].tail(10).mean() if len(df_hist[(df_hist['hometeam'] == home)]) > 0 else 0
    elo_diff = df_hist[(df_hist['hometeam'] == home)]['elo_diff'].tail(10).mean() if len(df_hist[(df_hist['hometeam'] == home)]) > 0 else 0
    diff = df_hist[(df_hist['hometeam'] == home)]['diff'].tail(10).mean() if len(df_hist[(df_hist['hometeam'] == home)]) > 0 else 0
    if pd.isna(rank_diff): rank_diff = 0
    if pd.isna(elo_diff): elo_diff = 0
    if pd.isna(diff): diff = 0
    feat = np.array([[
        h_stats['gf'], h_stats['ga'], h_stats['wr'], h_stats['gf_std'],
        a_stats['gf'], a_stats['ga'], a_stats['wr'], a_stats['gf_std'],
        elo_diff, diff, rank_diff, h2h
    ]])
    return xgb_model.predict_proba(feat)[0]


def train_xgb_over_classifier(df):
    cutoff = df['date'].max() - timedelta(days=365 * 2)
    train_df = df[df['date'] > cutoff].copy()
    if len(train_df) < 100:
        return None
    if len(train_df) > 8000:
        train_df = train_df.sample(n=8000, random_state=42)
    features = []
    labels = []
    for _, row in train_df.iterrows():
        home = row['hometeam']; away = row['awayteam']; date = row['date']
        h_stats = get_team_stats(home, df, date, venue='home')
        a_stats = get_team_stats(away, df, date, venue='away')
        h2h = get_head_to_head(home, away, date.isoformat(), _DF_HASH)
        rank_diff = row.get('rank_points', 0)
        elo_h = row.get('elo_diff', 0)
        diff_h = row.get('diff', 0)
        features.append([
            h_stats['gf'], h_stats['ga'], h_stats['wr'], h_stats['gf_std'],
            a_stats['gf'], a_stats['ga'], a_stats['wr'], a_stats['gf_std'],
            elo_h, diff_h, rank_diff, h2h
        ])
        labels.append(1 if (row['fthg'] + row['ftag']) >= 3 else 0)
    X = np.array(features)
    y = np.array(labels)
    if len(set(y)) < 2:
        return None
    model = xgb.XGBClassifier(
        objective='binary:logistic', max_depth=4, learning_rate=0.1,
        n_estimators=100, random_state=42, n_jobs=1,
        tree_method='hist', subsample=0.8, colsample_bytree=0.8
    )
    model.fit(X, y)
    return model


def predict_xgb_over(home, away, df_hist, xgb_over_model):
    if xgb_over_model is None:
        return None
    now = pd.Timestamp.now()
    h_stats = get_team_stats(home, df_hist, now, venue='home')
    a_stats = get_team_stats(away, df_hist, now, venue='away')
    h2h = get_head_to_head(home, away, now.isoformat(), _DF_HASH)
    rank_diff = df_hist[(df_hist['hometeam'] == home)]['rank_points'].tail(10).mean() if len(df_hist[(df_hist['hometeam'] == home)]) > 0 else 0
    elo_diff = df_hist[(df_hist['hometeam'] == home)]['elo_diff'].tail(10).mean() if len(df_hist[(df_hist['hometeam'] == home)]) > 0 else 0
    diff = df_hist[(df_hist['hometeam'] == home)]['diff'].tail(10).mean() if len(df_hist[(df_hist['hometeam'] == home)]) > 0 else 0
    if pd.isna(rank_diff): rank_diff = 0
    if pd.isna(elo_diff): elo_diff = 0
    if pd.isna(diff): diff = 0
    feat = np.array([[
        h_stats['gf'], h_stats['ga'], h_stats['wr'], h_stats['gf_std'],
        a_stats['gf'], a_stats['ga'], a_stats['wr'], a_stats['gf_std'],
        elo_diff, diff, rank_diff, h2h
    ]])
    return xgb_over_model.predict_proba(feat)[0][1]


@st.cache_resource
def get_xgb_models(df_hist):
    clf = train_xgb_classifier(df_hist)
    over_clf = train_xgb_over_classifier(df_hist)
    return clf, over_clf


# ==================== 体彩脚本执行 ====================
def _run_sporttery_script():
    if not os.path.exists(SCRIPT_PATH):
        return None, "体彩脚本不存在"
    try:
        cmd = [sys.executable, SCRIPT_PATH, "--pretty", "--pool-code", "hhad,had"]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if not result.stdout or len(result.stdout) == 0:
            stderr_msg = ""
            if result.stderr:
                try:
                    stderr_msg = result.stderr.decode('utf-8', errors='ignore')[:1000]
                except:
                    stderr_msg = str(result.stderr)[:1000]
            return None, f"脚本无输出。stderr: {stderr_msg}"
        raw = result.stdout
        for encoding in ['utf-8', 'gbk', 'gb18030', 'utf-8-sig', 'latin-1']:
            try:
                text = raw.decode(encoding)
                data = json.loads(text)
                return data, None
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        return None, f"无法解析脚本输出。前 500 字符：{raw[:500]}"
    except subprocess.TimeoutExpired:
        return None, "体彩脚本运行超时（30秒）"
    except Exception as e:
        return None, f"执行体彩脚本出错: {e}"


def fetch_all_matches():
    data, err = _run_sporttery_script()
    if err:
        st.error(f"❌ {err}")
        return [], set()
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
            'home_en': home_en, 'away_en': away_en,
            'home_cn': home_cn, 'away_cn': away_cn,
            'odds_h': float(odds.get('h', 1.0)),
            'odds_d': float(odds.get('d', 1.0)),
            'odds_a': float(odds.get('a', 1.0)),
            'key': match.get('key', '')
        })
    if skipped_count > 0:
        st.info(f"已跳过 {skipped_count} 场无胜平负赔率的比赛")
        st.info(f"🔍 调试：脚本返回 {len(matches)} 场比赛，其中有胜平负赔率的 {len(match_list)} 场")
    if not match_list:
        st.warning("体彩脚本返回了数据，但没有可用的胜平负赔率")
    return match_list, missing_teams


def fetch_sporttery_odds(home_team, away_team):
    data, err = _run_sporttery_script()
    if err or data is None:
        return None
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


# ==================== 核心预测 ====================
def has_team_data(team, df_hist, date_limit, min_games=3):
    if df_hist is None or len(df_hist) == 0:
        return False
    count = len(df_hist[
        ((df_hist['hometeam'] == team) | (df_hist['awayteam'] == team)) &
        (df_hist['date'] < date_limit)
    ])
    return count >= min_games


def estimate_lambdas_from_odds(odds_h, odds_d, odds_a):
    total = 1 / odds_h + 1 / odds_d + 1 / odds_a
    p_h = (1 / odds_h) / total
    p_d = (1 / odds_d) / total
    p_a = (1 / odds_a) / total
    best_h, best_a = 1.3, 1.1
    best_err = float('inf')
    for lam_h in np.arange(0.3, 4.0, 0.15):
        for lam_a in np.arange(0.3, 4.0, 0.15):
            ph = poisson.pmf(np.arange(9), lam_h)
            pa = poisson.pmf(np.arange(9), lam_a)
            joint = np.outer(ph, pa)
            joint = joint / joint.sum()
            win = np.sum(joint[np.tril_indices_from(joint, k=-1)])
            draw = np.sum(np.diag(joint))
            lose = np.sum(joint[np.triu_indices_from(joint, k=1)])
            err = (win - p_h) ** 2 * 2 + (draw - p_d) ** 2 + (lose - p_a) ** 2 * 2
            if err < best_err:
                best_err = err
                best_h, best_a = lam_h, lam_a
    return best_h, best_a, p_h, p_d, p_a


def compute_base_lambdas(home, away, df_hist, predict_date, params):
    now = predict_date
    elo_w = params['elo_weight']
    vol_sc = params['volatility_scale']
    strong_mag = params['strong_magnify']
    weak_red = params['weak_reduce']

    hs = get_team_stats(home, df_hist, now, venue='home')
    aw = get_team_stats(away, df_hist, now, venue='away')

    home_strength_norm = hs.get('strength', 0.5)
    away_strength_norm = aw.get('strength', 0.5)
    diff_stats = (home_strength_norm - away_strength_norm) * 2.0

    form_diff = hs.get('form', 0) - aw.get('form', 0)
    diff_stats += form_diff * 0.15

    home_elo_advantage = (
        df_hist[(df_hist['hometeam'] == home)]['elo_diff'].tail(10).mean()
        if len(df_hist[(df_hist['hometeam'] == home)]) > 0 else 0
    )
    if pd.isna(home_elo_advantage):
        home_elo_advantage = 0

    diff = elo_w * (home_elo_advantage / 400.0) + (1 - elo_w) * diff_stats

    home_defense_factor = get_opponent_defense_factor(away, df_hist, now)
    away_defense_factor = get_opponent_defense_factor(home, df_hist, now)
    home_attack_boost = 1 / home_defense_factor if home_defense_factor > 0 else 1.0
    away_attack_boost = 1 / away_defense_factor if away_defense_factor > 0 else 1.0

    home_rest = days_since_last_match(home, df_hist, now)
    away_rest = days_since_last_match(away, df_hist, now)
    rest_factor_home = rest_factor(home_rest)
    rest_factor_away = rest_factor(away_rest)

    volatility_factor_h = 1 + hs['gf_std'] * vol_sc
    volatility_factor_a = 1 + aw['gf_std'] * vol_sc

    base_lam_h = hs['gf'] * home_attack_boost * rest_factor_home * volatility_factor_h
    base_lam_a = aw['gf'] * away_attack_boost * rest_factor_away * volatility_factor_a

    abs_diff = abs(diff)
    if abs_diff >= 0.30:
        if diff > 0:
            lam_h = base_lam_h * strong_mag
            lam_a = base_lam_a * weak_red
        else:
            lam_a = base_lam_a * strong_mag
            lam_h = base_lam_h * weak_red
    elif abs_diff >= 0.12:
        if diff > 0:
            lam_h = base_lam_h * (1 + diff * 0.6)
            lam_a = base_lam_a * (1 - diff * 0.4)
        else:
            lam_a = base_lam_a * (1 - diff * 0.6)
            lam_h = base_lam_h * (1 + diff * 0.4)
    else:
        lam_h = base_lam_h * (1 + diff * 0.2)
        lam_a = base_lam_a * (1 - diff * 0.2)

    lam_h = max(lam_h, 0.3)
    lam_a = max(lam_a, 0.3)
    return lam_h, lam_a, diff, hs, aw


def compute_poisson_joint(lam_h, lam_a, rho, max_goals=8):
    ph = poisson.pmf(np.arange(max_goals + 1), lam_h)
    pa = poisson.pmf(np.arange(max_goals + 1), lam_a)
    joint = np.outer(ph, pa)
    factor = 1 + rho
    if factor <= 0:
        factor = 0.01
    joint[0, 0] *= factor
    joint[1, 0] *= factor
    joint[0, 1] *= factor
    joint[1, 1] *= factor
    # 平局概率提升 8%
    draw_boost = 1.08
    for i in range(max_goals + 1):
        joint[i, i] *= draw_boost
    joint = joint / np.sum(joint)
    return joint


def extract_score_probs(joint, max_goals=8):
    score_probs = []
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob = joint[i, j]
            if prob > 0.001:
                score_probs.append((f"{i}:{j}", prob))
    score_probs.sort(key=lambda x: x[1], reverse=True)
    return score_probs


def calc_handicap(joint, h, max_goals=8):
    w = 0.0; d = 0.0; l = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            home_adj = i - h
            away_adj = j
            if home_adj > away_adj:
                w += joint[i, j]
            elif home_adj == away_adj:
                d += joint[i, j]
            else:
                l += joint[i, j]
    return {'主胜': w, '平局': d, '客胜': l}


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
    lam_h, lam_a, diff, hs, aw = compute_base_lambdas(home, away, df_hist, now, params)

    home_has_data = has_team_data(home, df_hist, now)
    away_has_data = has_team_data(away, df_hist, now)
    data_source = "模型"
    if not home_has_data or not away_has_data:
        lam_h_odds, lam_a_odds, p_h, p_d, p_a = estimate_lambdas_from_odds(odds_h, odds_d, odds_a)
        if not home_has_data and not away_has_data:
            lam_h, lam_a = lam_h_odds, lam_a_odds
            data_source = "赔率反推（两队均无历史数据）"
        elif not home_has_data:
            lam_h = lam_h_odds
            data_source = "混合（主队无历史数据，用赔率）"
        else:
            lam_a = lam_a_odds
            data_source = "混合（客队无历史数据，用赔率）"

    max_goals = 8
    joint = compute_poisson_joint(lam_h, lam_a, params['rho_value'], max_goals)
    win = np.sum(joint[np.tril_indices_from(joint, k=-1)])
    draw = np.sum(np.diag(joint))
    lose = np.sum(joint[np.triu_indices_from(joint, k=1)])

    xgb_w = params['xgb_fusion_weight']
    xgb_probs = predict_xgb(home, away, df_hist, xgb_model)
    if xgb_probs is not None:
        win = win * (1 - xgb_w) + xgb_probs[0] * xgb_w
        draw = draw * (1 - xgb_w) + xgb_probs[1] * xgb_w
        lose = lose * (1 - xgb_w) + xgb_probs[2] * xgb_w
        total = win + draw + lose
        win /= total; draw /= total; lose /= total

    handicap_minus1 = calc_handicap(joint, 1)
    handicap_plus1 = calc_handicap(joint, -1)

    over_prob_poisson = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if i + j >= 3:
                over_prob_poisson += joint[i, j]
    over_prob_xgb = predict_xgb_over(home, away, df_hist, xgb_over_model)
    if over_prob_xgb is not None:
        over_prob = over_prob_poisson * (1 - xgb_w) + over_prob_xgb * xgb_w
    else:
        over_prob = over_prob_poisson

    over_threshold = get_league_dynamic_threshold(df_hist, now, offset=params['over_threshold_offset'])
    is_over = over_prob >= over_threshold

    score_probs = extract_score_probs(joint, max_goals)
    reliable = score_probs[0] if score_probs else ('0:0', 0.0)

    abs_diff = abs(diff)
    if abs_diff >= 0.30:
        aggressive_candidates = [(f"{i}:{j}", joint[i, j]) for i in range(max_goals + 1)
                                 for j in range(max_goals + 1)
                                 if 3 <= i + j <= 5 and joint[i, j] > 0.001]
    else:
        aggressive_candidates = [(f"{i}:{j}", joint[i, j]) for i in range(max_goals + 1)
                                 for j in range(max_goals + 1)
                                 if i + j >= 4 and joint[i, j] > 0.001]
    aggressive = max(aggressive_candidates, key=lambda x: x[1]) if aggressive_candidates else reliable

    if abs_diff >= 0.30:
        if diff > 0:
            if joint[2, 0] > 0.001 and joint[2, 0] >= joint[1, 1]:
                conservative = ("2:0", joint[2, 0])
            elif joint[1, 1] > 0.001:
                conservative = ("1:1", joint[1, 1])
            else:
                conservative = reliable
        else:
            if joint[0, 2] > 0.001 and joint[0, 2] >= joint[1, 1]:
                conservative = ("0:2", joint[0, 2])
            elif joint[1, 1] > 0.001:
                conservative = ("1:1", joint[1, 1])
            else:
                conservative = reliable
    else:
        candidates_low = [(f"{i}:{j}", joint[i, j]) for i in range(max_goals + 1)
                          for j in range(max_goals + 1)
                          if 1 <= i + j <= 2 and joint[i, j] > 0.001]
        if candidates_low:
            if joint[1, 1] > 0.001:
                conservative = ("1:1", joint[1, 1])
            else:
                conservative = max(candidates_low, key=lambda x: x[1])
        else:
            conservative = reliable

    if aggressive[0] == reliable[0]:
        for total in range(4, max_goals * 2 + 1):
            alt = [(f"{i}:{j}", joint[i, j]) for i in range(max_goals + 1)
                   for j in range(max_goals + 1)
                   if i + j == total and joint[i, j] > 0.001]
            if alt:
                aggressive = max(alt, key=lambda x: x[1])
                break
    if conservative[0] == reliable[0]:
        for alt in ["1:0", "0:1", "2:0", "0:2", "1:1"]:
            if alt != reliable[0]:
                prob = joint[int(alt[0]), int(alt[2])]
                if prob > 0.001:
                    conservative = (alt, prob)
                    break
        if conservative[0] == reliable[0]:
            conservative = score_probs[1] if len(score_probs) > 1 else reliable

    analysis_parts = [data_source]
       # 用赔率差作为辅助判断
    market_gap_for_analysis = market_h - market_a
    combined_diff = diff * 0.5 + market_gap_for_analysis * 1.5

    if abs(combined_diff) > 0.20:
        analysis_parts.append(f"实力{'主强客弱' if combined_diff > 0 else '客强主弱'}，差值 {abs(combined_diff):.2f}")
    elif abs(combined_diff) > 0.10:
        analysis_parts.append(f"实力{'主队略优' if combined_diff > 0 else '客队略优'}")
    else:
        analysis_parts.append("双方实力接近")
    if hs.get('sample_size', 0) >= 5:
        if hs['wr'] > 0.6:
            analysis_parts.append(f"主队近10场胜率较高（{hs['wr']*100:.0f}%）")
        elif hs['wr'] < 0.3:
            analysis_parts.append(f"主队近10场胜率较低（{hs['wr']*100:.0f}%）")
    if aw.get('sample_size', 0) >= 5:
        if aw['wr'] > 0.6:
            analysis_parts.append(f"客队近10场胜率较高（{aw['wr']*100:.0f}%）")
        elif aw['wr'] < 0.3:
            analysis_parts.append(f"客队近10场胜率较低（{aw['wr']*100:.0f}%）")
    h2h = get_head_to_head(home, away, now.isoformat(), _DF_HASH)
    if h2h > 0.6:
        analysis_parts.append(f"近5次交锋主队胜率较高（{h2h*100:.0f}%）")
    elif h2h < 0.4:
        analysis_parts.append(f"近5次交锋客队占优（主队胜率 {h2h*100:.0f}%）")
    if over_prob > 0.55:
        analysis_parts.append("大球倾向明显")
    elif over_prob < 0.35:
        analysis_parts.append("小球倾向明显")
    if (diff > 0 and lose > 0.3) or (diff < 0 and win > 0.3):
        analysis_parts.append("⚠️ 存在爆冷可能")
    analysis_parts.append(f"最可能比分 {reliable[0]} ({reliable[1]*100:.1f}%)")
    analysis_text = "；".join(analysis_parts)

    market_h = 1 / odds_h; market_d = 1 / odds_d; market_a = 1 / odds_a
    market_total = market_h + market_d + market_a
    market_h /= market_total; market_d /= market_total; market_a /= market_total

    # ========== 赔率一致性检查 ==========
    # 计算模型和市场对"主胜-客胜"的差距
    model_gap = win - lose        # 模型认为主客差距
    market_gap = market_h - market_a  # 市场认为主客差距

    # 如果市场认为差距远大于模型 → 说明模型漏掉了关键信息
    # 用差距比来决定融合权重
    if abs(market_gap) > 0.05:
        gap_ratio = abs(model_gap) / abs(market_gap)
        if gap_ratio < 0.5:
            # 模型低估了强弱差距，加大市场权重
            mix = 0.70
        elif gap_ratio < 0.8:
            # 模型有点低估，适度加市场权重
            mix = 0.55
        elif gap_ratio > 2.0:
            # 模型高估了强弱差距，降低市场权重
            mix = 0.30
        else:
            mix = 0.40
    else:
        mix = 0.40
    # ====================================

    win = win * (1 - mix) + market_h * mix
    draw = draw * (1 - mix) + market_d * mix
    lose = lose * (1 - mix) + market_a * mix
    total_cal = win + draw + lose
    win /= total_cal; draw /= total_cal; lose /= total_cal

    return {
        '主胜': win, '平局': draw, '客胜': lose,
        '让球-1': handicap_minus1, '让球+1': handicap_plus1,
        '大球概率': over_prob, '大球判定': '大球' if is_over else '小球',
        '大球阈值': over_threshold,
        '预期主队进球': lam_h, '预期客队进球': lam_a,
        '靠谱比分': reliable[0], '靠谱概率': reliable[1],
        '激进比分': aggressive[0], '激进概率': aggressive[1],
        '稳健比分': conservative[0], '稳健概率': conservative[1],
        '比分概率': score_probs[:5],
        'analysis': analysis_text,
        'data_source': data_source
    }


def fallback_predict(odds_h, odds_d, odds_a):
    total = 1 / odds_h + 1 / odds_d + 1 / odds_a
    win = (1 / odds_h) / total
    draw = (1 / odds_d) / total
    lose = (1 / odds_a) / total
    return {
        '主胜': win, '平局': draw, '客胜': lose,
        '让球-1': {'主胜': 0.3, '平局': 0.3, '客胜': 0.4},
        '让球+1': {'主胜': 0.5, '平局': 0.3, '客胜': 0.2},
        '大球概率': 0.4, '大球判定': '小球', '大球阈值': 0.45,
        '预期主队进球': 1.2, '预期客队进球': 1.2,
        '靠谱比分': '1:1', '靠谱概率': 0.1,
        '激进比分': '2:1', '激进概率': 0.08,
        '稳健比分': '0:0', '稳健概率': 0.07,
        '比分概率': [('1:1', 0.1), ('0:0', 0.08), ('1:0', 0.07), ('0:1', 0.07), ('2:1', 0.06)],
        'analysis': '数据不足，基于赔率估算',
        'data_source': '赔率反推'
    }


def kelly_suggestion(prob, odds):
    edge = prob * odds - 1
    if edge <= 0:
        return 0.0
    k = ((odds - 1) * prob - (1 - prob)) / (odds - 1)
    return max(0, k * 0.25)


def predict_all_matches(matches, df_hist, xgb_model, xgb_over_model):
    if not matches:
        return pd.DataFrame()
    results = []
    for match in matches:
        home = match['home_en']
        away = match['away_en']
        if home is None or away is None:
            continue
        odds_h = match['odds_h']; odds_d = match['odds_d']; odds_a = match['odds_a']
        try:
            r = predict_match(home, away, odds_h, odds_d, odds_a, df_hist, xgb_model, xgb_over_model)
            conf_label = ""
            probs_sorted = sorted([r['主胜'], r['平局'], r['客胜']], reverse=True)
            gap = probs_sorted[0] - probs_sorted[1]
            if gap > 0.25:
                conf_label = "🔥 高信心"
            elif gap > 0.15:
                conf_label = "✅ 中等信心"
            elif gap < 0.08:
                conf_label = "🤝 难分伯仲"
            if (r['主胜'] < 0.35 and r['客胜'] > 0.35) or (r['客胜'] < 0.35 and r['主胜'] > 0.35):
                conf_label += " ⚠️ 冷门"
            if r['大球概率'] > 0.6:
                conf_label += " 📈 大球"
            elif r['大球概率'] < 0.35:
                conf_label += " 📉 小球"
            results.append({
                '主队': match['home_cn'], '客队': match['away_cn'],
                '主胜概率': f"{r['主胜']*100:.1f}%",
                '平局概率': f"{r['平局']*100:.1f}%",
                '客胜概率': f"{r['客胜']*100:.1f}%",
                '胜平负方向': max(['主胜', '平局', '客胜'], key=lambda x: r[x]),
                '让球-1方向': max(r['让球-1'], key=r['让球-1'].get),
                '让球+1方向': max(r['让球+1'], key=r['让球+1'].get),
                '大小球': r['大球判定'],
                '靠谱比分': f"{r['靠谱比分']} ({r['靠谱概率']*100:.1f}%)",
                '激进比分': f"{r['激进比分']} ({r['激进概率']*100:.1f}%)",
                '稳健比分': f"{r['稳健比分']} ({r['稳健概率']*100:.1f}%)",
                '预期进球': f"{r['预期主队进球']:.2f}-{r['预期客队进球']:.2f}",
                '分析摘要': r.get('analysis', ''),
                '信心标签': conf_label.strip(),
                '数据来源': r.get('data_source', '模型')
            })
        except Exception as e:
            logger.error(f"预测 {home} vs {away} 失败: {e}")
            results.append({'主队': match['home_cn'], '客队': match['away_cn'], '错误': str(e)})
    return pd.DataFrame(results)


def save_predictions(df_pred):
    try:
        if not os.path.exists(PREDICTIONS_DIR):
            os.makedirs(PREDICTIONS_DIR)
        date_str = pd.Timestamp.now().strftime('%Y-%m-%d')
        path = os.path.join(PREDICTIONS_DIR, f'predictions_{date_str}.csv')
        df_pred.to_csv(path, index=False, encoding='utf-8-sig')
        return path
    except Exception as e:
        logger.error(f"保存预测失败: {e}")
        return None
    # ==================== 复盘 ====================
@st.cache_data(ttl=1800)
def fetch_actual_scores_from_odds_api():
    result = {}
    api_key = st.secrets.get("ODDS_API_KEY", "")
    if not api_key:
        return result
    try:
        url = f"https://api.the-odds-api.com/v4/sports/?apiKey={api_key}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return result
        sports = resp.json()
        soccer_keys = [s['key'] for s in sports if 'soccer' in s['key'].lower()]
        for sk in soccer_keys[:20]:
            try:
                scores_url = (
                    f"https://api.the-odds-api.com/v4/sports/{sk}/scores/"
                    f"?apiKey={api_key}&daysFrom=3"
                )
                r = requests.get(scores_url, timeout=10)
                if r.status_code != 200:
                    continue
                games = r.json()
                for g in games:
                    if not g.get('completed', False):
                        continue
                    home = g.get('home_team', '')
                    away = g.get('away_team', '')
                    scores = g.get('scores', [])
                    if not scores or len(scores) < 2:
                        continue
                    home_score = away_score = None
                    for s in scores:
                        if s.get('name') == home:
                            home_score = int(s.get('score', 0))
                        elif s.get('name') == away:
                            away_score = int(s.get('score', 0))
                    if home_score is not None and away_score is not None:
                        result[(home, away)] = (home_score, away_score)
            except Exception:
                continue
    except Exception as e:
        logger.error(f"从 The Odds API 获取比分失败: {e}")
    return result
# ==================== Football-Data.org 数据源 ====================
@st.cache_data(ttl=1800)
def fetch_results_from_football_data(date_str):
    """
    从 Football-Data.org 获取指定日期的比赛结果。
    date_str: 'YYYY-MM-DD'
    返回: {(home_en, away_en): (home_score, away_score)}
    """
    api_key = st.secrets.get("FOOTBALL_DATA_API_KEY", "")
    if not api_key:
        return {}

    competitions = [
        'PL',    # 英超
        'ELC',   # 英冠
        'PD',    # 西甲
        'BL1',   # 德甲
        'SA',    # 意甲
        'FL1',   # 法甲
        'DED',   # 荷甲
        'PPL',   # 葡超
        'BSA',   # 巴甲
        'CL',    # 欧冠
    ]

    results = {}
    headers = {'X-Auth-Token': api_key}

    for comp in competitions:
        try:
            url = f"https://api.football-data.org/v4/competitions/{comp}/matches"
            params = {
                'dateFrom': date_str,
                'dateTo': date_str,
                'status': 'FINISHED'
            }
            resp = requests.get(url, headers=headers, params=params, timeout=10)

            if resp.status_code == 429:
                logger.warning(f"Football-Data.org 限流，跳过 {comp}")
                continue
            if resp.status_code != 200:
                continue

            data = resp.json()
            for match in data.get('matches', []):
                home_en = match['homeTeam']['name']
                away_en = match['awayTeam']['name']
                home_score = match['score']['fullTime']['home']
                away_score = match['score']['fullTime']['away']

                if home_score is not None and away_score is not None:
                    results[(home_en, away_en)] = (home_score, away_score)
        except Exception as e:
            logger.warning(f"Football-Data.org 获取 {comp} 失败: {e}")
            continue

    logger.info(f"Football-Data.org 获取到 {len(results)} 场比赛结果")
    return results


# ==================== 球小策 数据源 ====================
@st.cache_data(ttl=1800)
def fetch_results_from_qiuxiaoce(date_str):
    """
    从球小策获取指定日期的比赛结果。
    date_str: 'YYYY-MM-DD'
    返回: {(home_cn, away_cn): (home_score, away_score), (home_en, away_en): (...)}
    """
    results = {}
    api_key = st.secrets.get("QIUXIAOCE_API_KEY", "")
    if not api_key:
        logger.warning("未配置 QIUXIAOCE_API_KEY")
        return results

    try:
        from qiuxiaoce import QiuXiaoCeClient
        client = QiuXiaoCeClient(api_key=api_key)
        fixtures = client.get_fixtures(date=date_str)

        for f in fixtures:
            if str(f.get('status', '')).upper() != 'FT':
                continue

            home_cn = f.get('home_team_name', '')
            away_cn = f.get('away_team_name', '')
            home_en = f.get('home_team_en', '')
            away_en = f.get('away_team_en', '')
            home_goals = f.get('home_goals')
            away_goals = f.get('away_goals')

            if home_goals is None or away_goals is None:
                continue

            if home_cn and away_cn:
                results[(home_cn, away_cn)] = (int(home_goals), int(away_goals))
            if home_en and away_en:
                results[(home_en, away_en)] = (int(home_goals), int(away_goals))

        logger.info(f"球小策获取到 {len(fixtures)} 场比赛")
    except Exception as e:
        logger.error(f"球小策获取失败: {e}")

    return results

def review_saved_predictions(df_hist):
    if not os.path.exists(PREDICTIONS_DIR):
        return None, "还没有任何预测记录，请先点「🚀 一键预测所有比赛」", 0, 0
    files = sorted(
        [f for f in os.listdir(PREDICTIONS_DIR)
         if f.startswith('predictions_') and f.endswith('.csv')],
        reverse=True
    )
    if not files:
        return None, "还没有任何预测记录", 0, 0
    latest_file = files[0]
    pred_date = latest_file.replace('predictions_', '').replace('.csv', '')
    pred_path = os.path.join(PREDICTIONS_DIR, latest_file)
    try:
        df_pred = pd.read_csv(pred_path, encoding='utf-8-sig')
    except Exception as e:
        return None, f"读取预测文件失败：{e}", 0, 0

    odds_scores = fetch_actual_scores_from_odds_api()
    odds_scores = fetch_actual_scores_from_odds_api()
    fd_scores = fetch_results_from_football_data(pred_date)
    qxc_scores = fetch_results_from_qiuxiaoce(pred_date)
    review_rows = []
    highlight_map = {}
    found_count = 0
    not_covered = []

    for i, row in df_pred.iterrows():
        home_cn = str(row.get('主队', '')).strip()
        away_cn = str(row.get('客队', '')).strip()
        home_en = TEAM_NAME_MAP_REVERSE.get(home_cn)
        away_en = TEAM_NAME_MAP_REVERSE.get(away_cn)

        actual_h = actual_a = None
        source = ""

        # 优先级 1：本地历史库
        if home_en and away_en:
            match = df_hist[(df_hist['hometeam'] == home_en) & (df_hist['awayteam'] == away_en)]
            if len(match) > 0:
                match = match.sort_values('date', ascending=False).iloc[0]
                actual_h = match['fthg']
                actual_a = match['ftag']
                source = "历史库"

        # 优先级 2：Football-Data.org
        if actual_h is None and home_en and away_en:
            key = (home_en, away_en)
            if key in fd_scores:
                actual_h, actual_a = fd_scores[key]
                source = "Football-Data"

        # 优先级 3：球小策
        if actual_h is None:
            key_cn = (home_cn, away_cn)
            key_en = (home_en, away_en)
            if key_cn in qxc_scores:
                actual_h, actual_a = qxc_scores[key_cn]
                source = "球小策"
            elif home_en and away_en and key_en in qxc_scores:
                actual_h, actual_a = qxc_scores[key_en]
                source = "球小策"

        # 优先级 4：The Odds API 兜底
        if actual_h is None and odds_scores:
            for (oh, oa), (hs, as_) in odds_scores.items():
                if translate_team_name(oh) == home_cn and translate_team_name(oa) == away_cn:
                    actual_h = hs
                    actual_a = as_
                    source = "Odds API"
                    break

        new_row = row.to_dict()
        if actual_h is not None:
            actual_score = f"{int(actual_h)}:{int(actual_a)}"
            new_row['实际比分'] = actual_score
            new_row['数据来源'] = source
            found_count += 1
            if actual_h > actual_a:
                actual_result = '主胜'
            elif actual_h == actual_a:
                actual_result = '平局'
            else:
                actual_result = '客胜'
            actual_minus = '主胜' if actual_h - 1 > actual_a else ('平局' if actual_h - 1 == actual_a else '客胜')
            actual_plus = '主胜' if actual_h + 1 > actual_a else ('平局' if actual_h + 1 == actual_a else '客胜')
            actual_over = '大球' if (actual_h + actual_a) >= 3 else '小球'
            if str(row.get('胜平负方向', '')).strip() == actual_result:
                highlight_map[(i, '胜平负方向')] = True
            if str(row.get('让球-1方向', '')).strip() == actual_minus:
                highlight_map[(i, '让球-1方向')] = True
            if str(row.get('让球+1方向', '')).strip() == actual_plus:
                highlight_map[(i, '让球+1方向')] = True
            if str(row.get('大小球', '')).strip() == actual_over:
                highlight_map[(i, '大小球')] = True
            reliable_score = str(row.get('靠谱比分', '')).split(' ')[0].strip()
            if reliable_score == actual_score:
                highlight_map[(i, '靠谱比分')] = True
            aggressive_score = str(row.get('激进比分', '')).split(' ')[0].strip()
            if aggressive_score == actual_score:
                highlight_map[(i, '激进比分')] = True
            conservative_score = str(row.get('稳健比分', '')).split(' ')[0].strip()
            if conservative_score == actual_score:
                highlight_map[(i, '稳健比分')] = True
        else:
            new_row['实际比分'] = '未找到'
            new_row['数据来源'] = '—'
            not_covered.append(f"{home_cn} vs {away_cn}")
        review_rows.append(new_row)

    review_df = pd.DataFrame(review_rows)

    def highlight_cell(row):
        styles = []
        for col in row.index:
            if highlight_map.get((row.name, col), False):
                styles.append('background-color: #c6efce; color: #006100; font-weight: bold;')
            else:
                styles.append('')
        return styles

    try:
        styled = review_df.style.apply(highlight_cell, axis=1)
    except Exception as e:
        logger.error(f"应用样式失败: {e}")
        styled = review_df

    summary = f"📅 预测日期：{pred_date} | 共 {len(df_pred)} 场 | 找到实际结果 {found_count} 场"
    if not_covered:
        summary += f" | 未覆盖 {len(not_covered)} 场"
    return styled, summary, found_count, len(df_pred)


# ==================== 周报 ====================
def load_recent_predictions(days=7):
    if not os.path.exists(PREDICTIONS_DIR):
        return {}
    files = sorted(
        [f for f in os.listdir(PREDICTIONS_DIR)
         if f.startswith('predictions_') and f.endswith('.csv')],
        reverse=True
    )
    cutoff = pd.Timestamp.now() - timedelta(days=days)
    result = {}
    for f in files:
        try:
            date_str = f.replace('predictions_', '').replace('.csv', '')
            file_date = pd.to_datetime(date_str)
            if file_date >= cutoff:
                df = pd.read_csv(os.path.join(PREDICTIONS_DIR, f), encoding='utf-8-sig')
                result[date_str] = df
        except Exception as e:
            logger.warning(f"读取 {f} 失败: {e}")
            continue
    return result


def evaluate_predictions_week(week_preds, df_hist):
    if not week_preds:
        return []

    odds_scores = fetch_actual_scores_from_odds_api()
    all_dates = set(week_preds.keys())
    fd_cache = {}
    qxc_cache = {}
    for d in all_dates:
        fd_cache[d] = fetch_results_from_football_data(d)
        qxc_cache[d] = fetch_results_from_qiuxiaoce(d)

    records = []
    for date_str, df_day in week_preds.items():
        fd_scores = fd_cache.get(date_str, {})
        qxc_scores = qxc_cache.get(date_str, {})
        for _, row in df_day.iterrows():
            home_cn = str(row.get('主队', '')).strip()
            away_cn = str(row.get('客队', '')).strip()
            home_en = TEAM_NAME_MAP_REVERSE.get(home_cn)
            away_en = TEAM_NAME_MAP_REVERSE.get(away_cn)

            actual_h = actual_a = None
            source = ""

            # 1. 本地历史库
            if home_en and away_en:
                match = df_hist[(df_hist['hometeam'] == home_en) & (df_hist['awayteam'] == away_en)]
                if len(match) > 0:
                    match = match.sort_values('date', ascending=False).iloc[0]
                    actual_h, actual_a = match['fthg'], match['ftag']
                    source = "历史库"

            # 2. Football-Data.org
            if actual_h is None and home_en and away_en:
                if (home_en, away_en) in fd_scores:
                    actual_h, actual_a = fd_scores[(home_en, away_en)]
                    source = "Football-Data"

            # 3. 球小策
            if actual_h is None:
                key_cn = (home_cn, away_cn)
                key_en = (home_en, away_en)
                if key_cn in qxc_scores:
                    actual_h, actual_a = qxc_scores[key_cn]
                    source = "球小策"
                elif home_en and away_en and key_en in qxc_scores:
                    actual_h, actual_a = qxc_scores[key_en]
                    source = "球小策"

            # 4. The Odds API 兜底
            if actual_h is None and odds_scores:
                for (oh, oa), (hs, as_) in odds_scores.items():
                    if translate_team_name(oh) == home_cn and translate_team_name(oa) == away_cn:
                        actual_h, actual_a = hs, as_
                        source = "Odds API"
                        break

            if actual_h is None:
                continue

            actual_score = f"{int(actual_h)}:{int(actual_a)}"
            if actual_h > actual_a:
                actual_result = '主胜'
            elif actual_h == actual_a:
                actual_result = '平局'
            else:
                actual_result = '客胜'
            actual_over = '大球' if (actual_h + actual_a) >= 3 else '小球'

            pred_wdl = str(row.get('胜平负方向', '')).strip()
            pred_over = str(row.get('大小球', '')).strip()
            pred_reliable = str(row.get('靠谱比分', '')).split(' ')[0].strip()
            conf_tag = str(row.get('信心标签', '')).strip()

            try:
                home_prob = float(str(row.get('主胜概率', '0%')).replace('%', '')) / 100
                draw_prob = float(str(row.get('平局概率', '0%')).replace('%', '')) / 100
                away_prob = float(str(row.get('客胜概率', '0%')).replace('%', '')) / 100
            except Exception:
                home_prob = draw_prob = away_prob = 0

            records.append({
                'date': date_str, 'home': home_cn, 'away': away_cn,
                'actual_score': actual_score, 'actual_result': actual_result,
                'actual_over': actual_over, 'pred_wdl': pred_wdl,
                'pred_over': pred_over, 'pred_reliable': pred_reliable,
                'conf_tag': conf_tag, 'source': source,
                'home_prob': home_prob, 'draw_prob': draw_prob, 'away_prob': away_prob,
                'correct_wdl': pred_wdl == actual_result,
                'correct_over': pred_over == actual_over,
                'correct_score': pred_reliable == actual_score,
            })
    return records

def analyze_week(records):
    if not records:
        return {}
    total = len(records)
    correct_wdl = sum(1 for r in records if r['correct_wdl'])
    correct_over = sum(1 for r in records if r['correct_over'])
    correct_score = sum(1 for r in records if r['correct_score'])

    analysis = {
        'total': total,
        'wdl_acc': correct_wdl / total,
        'over_acc': correct_over / total,
        'score_acc': correct_score / total,
        'wdl_correct': correct_wdl,
        'over_correct': correct_over,
        'score_correct': correct_score,
    }

    by_conf = {}
    for r in records:
        tag = r['conf_tag']
        key = "高信心" if "高信心" in tag else ("中等信心" if "中等信心" in tag else ("难分伯仲" if "难分伯仲" in tag else "其他"))
        if key not in by_conf:
            by_conf[key] = {'total': 0, 'correct': 0}
        by_conf[key]['total'] += 1
        if r['correct_wdl']:
            by_conf[key]['correct'] += 1
    for k in by_conf:
        by_conf[k]['acc'] = by_conf[k]['correct'] / by_conf[k]['total'] if by_conf[k]['total'] > 0 else 0
    analysis['by_conf'] = by_conf

    by_result = {'主胜': {'total': 0, 'correct': 0},
                 '平局': {'total': 0, 'correct': 0},
                 '客胜': {'total': 0, 'correct': 0}}
    for r in records:
        by_result[r['actual_result']]['total'] += 1
        if r['correct_wdl']:
            by_result[r['actual_result']]['correct'] += 1
    for k in by_result:
        by_result[k]['acc'] = by_result[k]['correct'] / by_result[k]['total'] if by_result[k]['total'] > 0 else 0
    analysis['by_result'] = by_result

    pred_home_count = sum(1 for r in records if r['pred_wdl'] == '主胜')
    pred_draw_count = sum(1 for r in records if r['pred_wdl'] == '平局')
    pred_away_count = sum(1 for r in records if r['pred_wdl'] == '客胜')
    analysis['pred_distribution'] = {
        '主胜': pred_home_count, '平局': pred_draw_count, '客胜': pred_away_count
    }
    analysis['actual_distribution'] = {
        '主胜': by_result['主胜']['total'],
        '平局': by_result['平局']['total'],
        '客胜': by_result['客胜']['total']
    }

    wrong_records = [r for r in records if not r['correct_wdl']]
    wrong_records.sort(key=lambda x: max(x['home_prob'], x['draw_prob'], x['away_prob']), reverse=True)
    analysis['worst_mistakes'] = wrong_records[:5]

    right_records = [r for r in records if r['correct_wdl']]
    right_records.sort(key=lambda x: max(x['home_prob'], x['draw_prob'], x['away_prob']), reverse=True)
    analysis['best_hits'] = right_records[:5]

    return analysis


def render_weekly_report(df_hist):
    st.subheader("📊 每周预测分析报告")
    st.caption("自动分析最近 7 天的预测准确率，并给出改进建议")

    days = st.slider("分析周期（天）", 3, 30, 7, key="report_days")

    with st.spinner("正在加载预测历史..."):
        week_preds = load_recent_predictions(days=days)

    if not week_preds:
        st.warning(f"⚠️ 最近 {days} 天没有预测记录。请先在一键预测中保存一些预测。")
        return

    st.success(f"✅ 已加载 {len(week_preds)} 天的预测记录")

    with st.spinner("正在获取实际结果并评估..."):
        records = evaluate_predictions_week(week_preds, df_hist)

    if not records:
        st.warning("⚠️ 无法获取任何比赛的实际结果（可能历史库未更新，或所有比赛都不在数据源覆盖范围）")
        return

    analysis = analyze_week(records)

    st.markdown("### 📈 总览")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总预测场次", analysis['total'])
    col2.metric("胜平负准确率", f"{analysis['wdl_acc']*100:.1f}%",
                delta=f"{analysis['wdl_correct']} / {analysis['total']}")
    col3.metric("大小球准确率", f"{analysis['over_acc']*100:.1f}%",
                delta=f"{analysis['over_correct']} / {analysis['total']}")
    col4.metric("比分准确率", f"{analysis['score_acc']*100:.1f}%",
                delta=f"{analysis['score_correct']} / {analysis['total']}")

    st.markdown("### 🎯 按信心等级准确率")
    conf_data = []
    for k, v in analysis['by_conf'].items():
        conf_data.append({
            '信心等级': k, '场次': v['total'], '正确': v['correct'],
            '准确率': f"{v['acc']*100:.1f}%"
        })
    if conf_data:
        st.dataframe(pd.DataFrame(conf_data), use_container_width=True)

    st.markdown("### ⚽ 按实际结果类型")
    result_data = []
    for k, v in analysis['by_result'].items():
        result_data.append({
            '实际结果': k, '场次': v['total'],
            '预测正确': v['correct'], '准确率': f"{v['acc']*100:.1f}%"
        })
    st.dataframe(pd.DataFrame(result_data), use_container_width=True)

    st.markdown("### 📊 预测分布 vs 实际分布")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**预测分布**")
        st.write(analysis['pred_distribution'])
    with col_b:
        st.write("**实际分布**")
        st.write(analysis['actual_distribution'])

    if analysis['worst_mistakes']:
        st.markdown("### ❌ 最意外的 5 场错误")
        st.caption("按模型信心度排序（越有信心却预测错的，越值得关注）")
        for r in analysis['worst_mistakes']:
            max_prob = max(r['home_prob'], r['draw_prob'], r['away_prob'])
            with st.expander(f"❌ {r['home']} vs {r['away']} | 预测 {r['pred_wdl']}，实际 {r['actual_result']} {r['actual_score']}"):
                st.write(f"**预测**：胜平负={r['pred_wdl']}（概率 {max_prob*100:.0f}%），比分={r['pred_reliable']}")
                st.write(f"**实际**：{r['actual_result']}，比分 {r['actual_score']}")
                st.write(f"**数据来源**：{r['source']}")

    if analysis['best_hits']:
        st.markdown("### ✅ 最精准的 5 场预测")
        st.caption("高信心且预测正确")
        for r in analysis['best_hits']:
            max_prob = max(r['home_prob'], r['draw_prob'], r['away_prob'])
            st.write(f"✅ **{r['home']} vs {r['away']}** | 预测 {r['pred_wdl']}，实际 {r['actual_result']} {r['actual_score']} | 概率 {max_prob*100:.0f}%")

    st.markdown("### 💡 改进建议")
    tips = []

    pred_home = analysis['pred_distribution']['主胜']
    actual_home = analysis['actual_distribution']['主胜']
    if pred_home > actual_home * 1.3 and analysis['total'] >= 10:
        tips.append(f"⚠️ 模型**过度预测主胜**（预测 {pred_home} 场，实际 {actual_home} 场），建议降低主队优势")

    pred_draw = analysis['pred_distribution']['平局']
    actual_draw = analysis['actual_distribution']['平局']
    if pred_draw < actual_draw * 0.6 and analysis['total'] >= 10:
        tips.append(f"⚠️ 模型**低估平局**（预测 {pred_draw} 场，实际 {actual_draw} 场），建议提高平局概率")

    if '高信心' in analysis['by_conf']:
        high_acc = analysis['by_conf']['高信心']['acc']
        if high_acc < 0.5:
            tips.append(f"⚠️ 高信心预测准确率仅 {high_acc*100:.0f}%，建议收紧信心阈值")
        elif high_acc > 0.7:
            tips.append(f"✅ 高信心预测准确率 {high_acc*100:.0f}%，表现优秀")

    if analysis['over_acc'] < 0.5:
        tips.append(f"⚠️ 大小球准确率 {analysis['over_acc']*100:.0f}% 偏低，可调 `大球阈值偏移` 参数")
    elif analysis['over_acc'] > 0.6:
        tips.append(f"✅ 大小球准确率 {analysis['over_acc']*100:.0f}%，表现良好")

    if not tips:
        tips.append("✅ 各项指标均在合理范围内，继续保持")

    for tip in tips:
        st.markdown(f"- {tip}")

    df_report = pd.DataFrame(records)
    csv = df_report.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        "📥 下载完整周报 CSV", csv,
        f"weekly_report_{pd.Timestamp.now().strftime('%Y-%m-%d')}.csv",
        "text/csv"
    )


# ==================== 回测 ====================
def get_odds_columns(df):
    candidates_h = ['bbmxh', 'maxh', 'avwh', 'b365h', 'pinnacleh', 'whh', 'sxh', 'h']
    candidates_d = ['bbmxd', 'maxd', 'avwd', 'b365d', 'pinnacled', 'whd', 'sxd', 'd']
    candidates_a = ['bbmxa', 'maxa', 'avwa', 'b365a', 'pinnaclea', 'wha', 'sxa', 'a']
    h = next((c for c in candidates_h if c in df.columns), None)
    d = next((c for c in candidates_d if c in df.columns), None)
    a = next((c for c in candidates_a if c in df.columns), None)
    return h, d, a


def run_backtest(start_date, end_date, df_hist, xgb_model, xgb_over_model):
    mask = (df_hist['date'] >= start_date) & (df_hist['date'] <= end_date)
    test_df = df_hist[mask].copy()
    if len(test_df) == 0:
        return None, "该时间段没有比赛数据", []
    h_col, d_col, a_col = get_odds_columns(test_df)
    results = []
    win_correct = draw_correct = lose_correct = 0
    over_correct = under_correct = 0
    handicap_minus_correct = handicap_plus_correct = 0
    total = total_over_under = total_handicap_minus = total_handicap_plus = 0
    win_probs = []; draw_probs = []; lose_probs = []
    actual_win = []; actual_draw = []; actual_lose = []
    prediction_records = []
    for _, row in test_df.iterrows():
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
        if h_col and d_col and a_col:
            odds_h = row[h_col] if pd.notna(row[h_col]) else 1.0
            odds_d = row[d_col] if pd.notna(row[d_col]) else 1.0
            odds_a = row[a_col] if pd.notna(row[a_col]) else 1.0
        else:
            odds_h = odds_d = odds_a = 1.0
        if odds_h <= 0 or odds_d <= 0 or odds_a <= 0:
            continue
        try:
            pred = predict_match(home, away, odds_h, odds_d, odds_a, df_hist, xgb_model, xgb_over_model, predict_date=row['date'])
        except:
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
        if pred_minus == actual_minus_res: handicap_minus_correct += 1
        total_handicap_plus += 1
        if pred_plus == actual_plus_res: handicap_plus_correct += 1
        win_probs.append(pred['主胜']); draw_probs.append(pred['平局']); lose_probs.append(pred['客胜'])
        actual_win.append(1 if actual_result == '主胜' else 0)
        actual_draw.append(1 if actual_result == '平局' else 0)
        actual_lose.append(1 if actual_result == '客胜' else 0)
        prediction_records.append({'prob': max(pred_labels.values()), 'correct': 1 if pred_result == actual_result else 0})
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
        return None, "该时间段内无可成功预测的比赛（可能缺少赔率）", []
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
        '让球-1准确率': f"{handicap_minus_correct/total_handicap_minus*100:.1f}%" if total_handicap_minus > 0 else 'N/A',
        '让球+1准确率': f"{handicap_plus_correct/total_handicap_plus*100:.1f}%" if total_handicap_plus > 0 else 'N/A',
        'Brier分数(主胜)': f"{brier_win:.3f}",
        'Brier分数(平局)': f"{brier_draw:.3f}",
        'Brier分数(客胜)': f"{brier_lose:.3f}",
        '平均Brier分数': f"{brier_avg:.3f}",
    }
    return pd.DataFrame(results), accuracy, prediction_records


def get_yesterday_results(df_hist, xgb_model, xgb_over_model):
    today = pd.Timestamp.now().normalize()
    for days_ago in range(1, 8):
        target = today - timedelta(days=days_ago)
        mask = (df_hist['date'] >= target) & (df_hist['date'] < target + timedelta(days=1))
        if len(df_hist[mask]) > 0:
            results_df, accuracy, _ = run_backtest(
                target, target + timedelta(days=1) - timedelta(seconds=1),
                df_hist, xgb_model, xgb_over_model
            )
            info = "昨日" if days_ago == 1 else f"{days_ago} 天前（{target.date()}）"
            return results_df, accuracy, info
    latest_date = df_hist['date'].max().date()
    return None, f"最近 7 天无比赛数据。当前最新日期为 {latest_date}", None


# ==================== 可视化 ====================
def plot_accuracy_trend(results_df):
    if 'date' not in results_df.columns or len(results_df) < 5:
        st.info("数据量不足，无法绘制趋势图")
        return
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df['correct'] = (results_df['预测结果'] == results_df['实际结果']).astype(int)
    daily_accuracy = results_df.groupby(results_df['date'].dt.date)['correct'].mean()
    moving_avg = daily_accuracy.rolling(7, min_periods=1).mean() if len(daily_accuracy) >= 7 else daily_accuracy
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily_accuracy.index, daily_accuracy.values, 'o-', alpha=0.6, label='每日准确率', markersize=4)
    ax.plot(moving_avg.index, moving_avg.values, 'r-', linewidth=2, label='7日移动平均')
    ax.axhline(y=0.33, color='gray', linestyle='--', label='随机基准 (33%)')
    ax.set_xlabel('日期'); ax.set_ylabel('准确率')
    ax.set_title('胜平负预测准确率趋势')
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def plot_calibration_curve(probabilities, outcomes, n_bins=10):
    if len(probabilities) < 10:
        st.info("数据量不足，无法绘制校准曲线")
        return
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accuracies = []; bin_counts = []
    for i in range(n_bins):
        mask = (probabilities >= bins[i]) & (probabilities < bins[i + 1])
        if np.sum(mask) > 0:
            bin_accuracies.append(np.mean(outcomes[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_accuracies.append(np.nan); bin_counts.append(0)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='完美校准')
    ax.scatter(bin_centers, bin_accuracies, s=np.array(bin_counts) * 10, alpha=0.6, label='实际频率')
    for i, (x, y, c) in enumerate(zip(bin_centers, bin_accuracies, bin_counts)):
        if not np.isnan(y) and c > 0:
            ax.annotate(f'n={c}', (x, y), fontsize=8, ha='center', va='bottom')
    ax.set_xlabel('预测概率'); ax.set_ylabel('实际频率')
    ax.set_title('概率校准曲线 (越接近对角线越好)')
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def simulate_kelly_profit(results_df, initial_capital=10000):
    if len(results_df) < 5:
        st.info("数据量不足，无法模拟盈亏")
        return
    capital = initial_capital
    capital_history = [capital]
    bet_count = 0; win_count = 0
    for _, row in results_df.iterrows():
        prob = np.random.uniform(0.3, 0.7)
        odd = 1 / prob + 0.5
        kelly_fraction = kelly_suggestion(prob, odd)
        if kelly_fraction <= 0:
            continue
        stake = capital * min(kelly_fraction, 0.15)
        if np.random.random() < prob:
            capital += stake * (odd - 1); win_count += 1
        else:
            capital -= stake
        bet_count += 1
        capital_history.append(capital)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(capital_history)
    ax.axhline(y=initial_capital, color='gray', linestyle='--', label=f'初始资本 ${initial_capital}')
    ax.set_xlabel('投注次数'); ax.set_ylabel('资本')
    ax.set_title(f'凯利投注盈亏模拟 (投注{bet_count}次, 胜率{win_count/max(bet_count,1)*100:.1f}%)')
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def plot_prob_distribution(score_probs):
    scores = [s[0] for s in score_probs]
    probs = [s[1] for s in score_probs]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(scores, probs, color='royalblue')
    ax.set_title('Top 比分概率分布')
    ax.set_xlabel('比分'); ax.set_ylabel('概率')
    st.pyplot(fig)


# ==================== 爆冷、焦点战 ====================
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
    diff = (hs.get('strength', 0.5) - aw.get('strength', 0.5)) * 2.0
    strength_diff = abs(diff)
    if diff > 0:
        weak_win_prob = result['客胜']; weak_draw_prob = result['平局'] * 0.5
        weak_label = '客队'; weak_team = away; weak_odds = odds_a
    else:
        weak_win_prob = result['主胜']; weak_draw_prob = result['平局'] * 0.5
        weak_label = '主队'; weak_team = home; weak_odds = odds_h
    upset_value = weak_odds * weak_win_prob
    big_score_prob = sum(prob for score, prob in result['比分概率']
                          if int(score.split(':')[0]) + int(score.split(':')[1]) >= 6)
    market_implied_prob = 1 / weak_odds
    model_weak_prob = weak_win_prob + weak_draw_prob * 0.3
    market_mispricing = model_weak_prob - market_implied_prob
    strength_score = min(strength_diff * 2, 1.0)
    upset_score = min(upset_value * 3, 1.0)
    big_score = min(big_score_prob * 10, 1.0)
    mispricing_score = min(max(market_mispricing * 2, 0), 1.0)
    upset_index = (strength_score * 0.25 + upset_score * 0.30 + big_score * 0.20 + mispricing_score * 0.25) * 100
    return {
        'home': match['home_cn'], 'away': match['away_cn'],
        'diff': diff, 'strength_diff': strength_diff,
        'weak_label': weak_label,
        'weak_team': weak_team if isinstance(weak_team, str) else str(weak_team),
        'weak_win_prob': weak_win_prob, 'weak_draw_prob': weak_draw_prob, 'weak_odds': weak_odds,
        'big_score_prob': big_score_prob, 'market_mispricing': market_mispricing,
        'upset_index': upset_index,
        'reliable_score': result['靠谱比分'], 'aggressive_score': result['激进比分'],
        'conservative_score': result['稳健比分'],
        'home_win_prob': result['主胜'], 'draw_prob': result['平局'], 'away_win_prob': result['客胜'],
        'over_prob': result['大球概率'], 'home_goals': result['预期主队进球'],
        'away_goals': result['预期客队进球'],
        'analysis': result.get('analysis', '')
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
        diff = abs(hs.get('strength', 0.5) - aw.get('strength', 0.5)) * 2
        points_diff = abs(hs['wr'] - aw['wr']) * 100
        focus_score = diff + points_diff * 0.3
        focus.append({
            'home': match['home_cn'], 'away': match['away_cn'],
            'diff': diff, 'points_diff': points_diff, 'focus_score': focus_score,
            'home_en': home, 'away_en': away,
            'odds_h': match['odds_h'], 'odds_d': match['odds_d'], 'odds_a': match['odds_a']
        })
    focus.sort(key=lambda x: x['focus_score'], reverse=True)
    return focus[:top_n]


# ==================== Streamlit UI ====================
st.set_page_config(page_title="精算足球预测器 · 终极稳定版", page_icon="⚽", layout="wide")

with st.spinner("正在加载历史数据，请稍候..."):
    df_hist = load_history()
if df_hist is None or len(df_hist) == 0:
    st.error("❌ 无法加载历史数据，请检查网络后点击侧边栏「更新历史数据」重试")
    st.stop()

_DF_HIST = df_hist
_DF_HASH = get_df_hash(df_hist)

with st.spinner("正在预计算球队统计..."):
    build_team_stats_table(df_hist)

with st.spinner("正在训练预测模型，请稍候..."):
    xgb_model, xgb_over_model = get_xgb_models(df_hist)

min_date = df_hist['date'].min().date()
max_date = df_hist['date'].max().date()
st.info(f"📊 历史数据范围：**{min_date}** 至 **{max_date}**，共 **{len(df_hist)}** 场比赛")

st.sidebar.header("⚙️ 参数调整")

defaults = {
    'elo_weight': 0.6, 'volatility_scale': 0.05,
    'over_threshold_offset': 0.0, 'strong_magnify': 1.6,
    'weak_reduce': 0.6, 'rho_value': -0.12, 'xgb_fusion_weight': 0.5
}

# 主参数（真正生效的值）
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# 临时参数（滑块拖动的值，不点"应用"不生效）
for key, val in defaults.items():
    tmp_key = f"_tmp_{key}"
    if tmp_key not in st.session_state:
        st.session_state[tmp_key] = st.session_state[key]

# ========== 滑块（绑定到临时参数） ==========
st.sidebar.slider("Elo 权重", 0.0, 1.0,
                  st.session_state["_tmp_elo_weight"], 0.05,
                  key="_tmp_elo_weight")
st.sidebar.caption("↑ 越大越依赖 Elo 长期实力评分；↓ 越小越依赖近期进球/失球")

st.sidebar.slider("波动性放大系数", 0.0, 0.2,
                  st.session_state["_tmp_volatility_scale"], 0.01,
                  key="_tmp_volatility_scale")
st.sidebar.caption("↑ 越大对状态起伏越敏感，进球预测波动更大；↓ 越小越保守")

st.sidebar.slider("大球阈值偏移", -0.1, 0.1,
                  st.session_state["_tmp_over_threshold_offset"], 0.01,
                  key="_tmp_over_threshold_offset")
st.sidebar.caption("↑ 正值更容易判「大球」；↓ 负值更容易判「小球」")

st.sidebar.slider("实力悬殊放大倍数", 1.0, 2.0,
                  st.session_state["_tmp_strong_magnify"], 0.1,
                  key="_tmp_strong_magnify")
st.sidebar.caption("↑ 强队进球被放大更多，比分更悬殊；↓ 强队优势被压缩")

st.sidebar.slider("实力悬殊弱队缩小倍数", 0.3, 1.0,
                  st.session_state["_tmp_weak_reduce"], 0.05,
                  key="_tmp_weak_reduce")
st.sidebar.caption("↓ 弱队进球被压缩更狠，冷门概率更低；↑ 弱队保留更多进球期望")

st.sidebar.slider("Dixon-Coles rho", -0.3, 0.0,
                  st.session_state["_tmp_rho_value"], 0.01,
                  key="_tmp_rho_value")
st.sidebar.caption("↓ 越负越抑制低比分（0:0/1:0/1:1），提高大比分概率；↑ 接近 0 则影响越小")

st.sidebar.slider("XGBoost 融合权重", 0.0, 1.0,
                  st.session_state["_tmp_xgb_fusion_weight"], 0.05,
                  key="_tmp_xgb_fusion_weight")
st.sidebar.caption("↑ 越大越依赖机器学习模型；↓ 越小越依赖泊松统计模型")

st.sidebar.markdown("---")

# ========== 应用/恢复按钮 ==========
col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("✅ 应用参数", type="primary", use_container_width=True):
        for key in defaults.keys():
            st.session_state[key] = st.session_state[f"_tmp_{key}"]
        st.sidebar.success("参数已生效")
        st.rerun()
with col_b:
    if st.button("↩️ 恢复默认", use_container_width=True):
        for key, val in defaults.items():
            st.session_state[f"_tmp_{key}"] = val
            st.session_state[key] = val
        st.sidebar.success("已恢复默认")
        st.rerun()

# 显示当前生效的参数（默认折叠）
with st.sidebar.expander("📋 当前生效参数", expanded=False):
    for key in defaults.keys():
        current = st.session_state[key]
        temp = st.session_state[f"_tmp_{key}"]
        changed = " ⚠️待应用" if abs(current - temp) > 0.001 else ""
        st.write(f"**{key}**: {current:.3f}{changed}")

st.title("⚽ 精算足球预测器 · 终极稳定版")
st.caption("Elo + 多维实力 + 主客场分离 + 对手强度加权 + 跨赛季评估 + XGBoost(hist) + 周报分析")

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
        _DF_HIST = df_hist
        _DF_HASH = get_df_hash(df_hist) if df_hist is not None else None
        if df_hist is not None:
            build_team_stats_table(df_hist)
        st.cache_resource.clear()
        st.rerun()

mode = st.sidebar.radio("选择模式", ["单场预测", "批量预测", "回测", "周报"], index=0)

st.sidebar.markdown("---")
if st.sidebar.checkbox("📋 显示所有球队名称（中英文）"):
    try:
        teams = sorted(df_hist['hometeam'].unique())
        st.sidebar.write(f"共 **{len(teams)}** 支球队：")
        display_text = ""
        for t in teams:
            cn = translate_team_name(t)
            display_text += f"{cn} ({t})\n" if cn != t else f"{t} (待补充中文名)\n"
        st.sidebar.text_area("球队列表", display_text, height=400)
    except Exception as e:
        st.sidebar.error(f"无法加载球队数据: {e}")

for k, v in {
    'fetch_odds_trigger': False, 'odds_h': 2.00, 'odds_d': 3.40, 'odds_a': 3.80,
    'home_team': None, 'away_team': None, 'match_list': [], 'missing_teams': set(),
    'predict_results': {}, 'batch_pred_df': None, 'upset_results': [],
    'yesterday_results': None, 'yesterday_accuracy': None, 'focus_matches': [],
    'confidence_results': [], 'confidence_slider': 5
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

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


# ===================== 单场预测 =====================
if mode == "单场预测":
    st.subheader("🔮 单场预测")
    st.markdown("---")
    col_load1, col_load2 = st.columns([1, 3])
    with col_load1:
        if st.button("📅 加载今日比赛"):
            with st.spinner("正在拉取体彩比赛数据..."):
                matches, missing = fetch_all_matches()
                if matches:
                    st.session_state.match_list = matches
                    st.session_state.missing_teams = missing
                    st.success(f"成功加载 {len(matches)} 场比赛！")
                    if missing:
                        st.warning(f"以下球队映射缺失，请补充：{', '.join(missing)}")
                else:
                    st.warning("未获取到任何比赛，请检查体彩脚本")
    # ========== 复盘按钮（不依赖今日比赛）==========
    st.markdown("---")
    col_review1, col_review2 = st.columns([1, 3])
    with col_review1:
        if st.button("📊 复盘昨日"):
            with st.spinner("正在加载预测记录并对比实际结果..."):
                styled, summary, found, total = review_saved_predictions(df_hist)
                st.markdown("---")
                st.subheader("📈 复盘结果")
                if styled is None:
                    st.warning(f"⚠️ {summary}")
                else:
                    if found == 0:
                        st.info(f"{summary}（历史库尚未更新，请稍后再试）")
                    else:
                        st.success(f"✅ {summary}")
                    st.caption("绿色 = 预测正确")
                    st.dataframe(styled, use_container_width=True)
                    try:
                        csv = styled.data.to_csv(index=False, encoding='utf-8-sig')
                    except:
                        csv = pd.DataFrame(styled).to_csv(index=False, encoding='utf-8-sig')
                    st.download_button("📥 下载复盘 CSV", csv,
                                       f"review_{pd.Timestamp.now().strftime('%Y-%m-%d')}.csv", "text/csv")
    if st.session_state.match_list:
        col_btn1, col_btn2, col_btn4 = st.columns(3)
        with col_btn1:
            if st.button("🚀 一键预测所有比赛"):
                with st.spinner("正在批量预测..."):
                    df_pred = predict_all_matches(st.session_state.match_list, df_hist, xgb_model, xgb_over_model)
                    st.session_state.batch_pred_df = df_pred
                    saved_path = save_predictions(df_pred)
                    if saved_path:
                        st.success(f"批量预测完成！已保存至 {saved_path}")
                    else:
                        st.warning("批量预测完成，但保存失败")
                    st.rerun()
        with col_btn2:
            if st.button("🔥 筛选最可能爆冷的 5 场比赛"):
                with st.spinner("正在分析爆冷可能性..."):
                    st.session_state.upset_results = get_upset_matches(
                        st.session_state.match_list, df_hist, xgb_model, xgb_over_model, top_n=5
                    )
                    st.rerun()
        with col_btn4:
            if st.button("⭐ 焦点战推荐"):
                with st.spinner("正在分析焦点战..."):
                    st.session_state.focus_matches = get_focus_matches(
                        st.session_state.match_list, df_hist, top_n=3
                    )
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
                            r = predict_match(match['home_en'], match['away_en'],
                                              match['odds_h'], match['odds_d'], match['odds_a'],
                                              df_hist, xgb_model, xgb_over_model)
                            probs = [r['主胜'], r['平局'], r['客胜']]
                            sorted_probs = sorted(probs, reverse=True)
                            confidence = sorted_probs[0] - sorted_probs[1]
                            top_score_prob = r['比分概率'][0][1] if r['比分概率'] else 0
                            confidence_score = confidence * 0.6 + top_score_prob * 0.4
                            conf_results.append({
                                'match': match, 'result': r, 'confidence': confidence_score,
                                'direction': max(['主胜', '平局', '客胜'], key=lambda x: r[x]),
                                'prob': max(probs)
                            })
                        except Exception:
                            continue
                    conf_results.sort(key=lambda x: x['confidence'], reverse=True)
                    st.session_state.confidence_results = conf_results[:st.session_state.confidence_slider]
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
                    if upset['analysis']:
                        st.caption(f"📝 {upset['analysis']}")
                    if upset['diff'] > 0:
                        st.info(f"💡 提示：{upset['away']} 有 {upset['weak_win_prob']*100:.1f}% 的概率客场爆冷击败 {upset['home']}")
                    else:
                        st.info(f"💡 提示：{upset['home']} 有 {upset['weak_win_prob']*100:.1f}% 的概率主场爆冷击败 {upset['away']}")
                    if upset['big_score_prob'] > 0.05:
                        st.warning(f"⚡ 大比分预警：本场打出 6 球以上的概率为 {upset['big_score_prob']*100:.1f}%")
                    st.markdown("---")

        if st.session_state.focus_matches:
            st.markdown("---")
            st.subheader("⭐ 焦点战推荐")
            for i, focus in enumerate(st.session_state.focus_matches, 1):
                st.markdown(f"**⭐ #{i} {focus['home']} vs {focus['away']}**")
                st.write(f"实力差: {focus['diff']:.2f} | 积分差距: {focus['points_diff']:.1f}")
                st.write(f"关注度得分: {focus['focus_score']:.2f}")
                st.info("💡 提示：该比赛实力或状态差距明显")
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
                    if r.get('analysis'):
                        st.caption(f"📝 {r['analysis']}")
                    st.markdown("---")

        if st.session_state.yesterday_results is not None and st.session_state.yesterday_accuracy is not None:
            st.markdown("---")
            st.subheader("📈 昨日复盘结果")
            st.success("复盘完成！")
            st.write("**准确率统计**")
            acc_df = pd.DataFrame([st.session_state.yesterday_accuracy]).T.rename(columns={0: '值'})
            st.table(acc_df)
            st.write("**详细对比**")
            st.dataframe(st.session_state.yesterday_results, use_container_width=True)

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
                                    'home': match['home_cn'], 'away': match['away_cn'],
                                    'most_likely': result['靠谱比分'],
                                    'win': result['主胜'], 'draw': result['平局'], 'lose': result['客胜'],
                                    'time': datetime.now().strftime("%Y-%m-%d %H:%M")
                                })
                                if len(st.session_state.prediction_history) > 20:
                                    st.session_state.prediction_history = st.session_state.prediction_history[-20:]
                                st.rerun()
                            except Exception as e:
                                st.error(f"预测出错: {e}")
                                st.code(traceback.format_exc())

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
                    st.write(f"**大小球概率**: {result['大球概率']*100:.1f}%，动态阈值 {result['大球阈值']*100:.1f}%")
                    st.write(f"**大球判定**: {result['大球判定']}")
                    st.write("**推荐比分**")
                    st.write(f"靠谱: {result['靠谱比分']} ({result['靠谱概率']*100:.1f}%)")
                    st.write(f"激进: {result['激进比分']} ({result['激进概率']*100:.1f}%)")
                    st.write(f"稳健: {result['稳健比分']} ({result['稳健概率']*100:.1f}%)")
                    if result.get('analysis'):
                        st.caption(f"📝 {result['analysis']}")
                    if result.get('data_source'):
                        st.caption(f"📊 数据来源：{result['data_source']}")
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
        # 过滤掉 NaN 和空字符串
        hist_teams = set()
        if df_hist is not None and 'hometeam' in df_hist.columns:
            for t in df_hist['hometeam'].dropna().unique():
                if isinstance(t, str) and t.strip():
                    hist_teams.add(t.strip())

        map_teams = set()
        for k in TEAM_NAME_MAP.keys():
            if isinstance(k, str) and k.strip():
                map_teams.add(k.strip())

        team_list = sorted(hist_teams | map_teams)
        logger.info(f"可选球队总数：{len(team_list)}（历史库 {len(hist_teams)} + 映射表 {len(map_teams)}）")
    except Exception as e:
        logger.error(f"构建球队列表失败: {e}")
        team_list = sorted([k for k in TEAM_NAME_MAP.keys() if isinstance(k, str)])

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        home_team = st.selectbox("主队名称", options=team_list,
                              format_func=safe_format_team,
                              index=0 if team_list else None, key="home_team_select")
        if home_team != st.session_state.home_team:
            st.session_state.home_team = home_team
        odds_h = st.number_input("主胜赔率", key="odds_h_manual", min_value=1.01,
                                  value=st.session_state.get("odds_h", 2.00), step=0.01)
    with col2:
        away_team = st.selectbox("客队名称", options=team_list,
                              format_func=safe_format_team,
                              index=1 if len(team_list) > 1 else None, key="away_team_select")
        if away_team != st.session_state.away_team:
            st.session_state.away_team = away_team
        odds_d = st.number_input("平局赔率", key="odds_d_manual", min_value=1.01,
                                  value=st.session_state.get("odds_d", 3.40), step=0.01)
    with col3:
        odds_a = st.number_input("客胜赔率", key="odds_a_manual", min_value=1.01,
                                  value=st.session_state.get("odds_a", 3.80), step=0.01)

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
                result = predict_match(home_team, away_team, odds_h, odds_d, odds_a,
                                        df_hist, xgb_model, xgb_over_model)
                st.session_state.prediction_history.append({
                    'home': translate_team_name(home_team),
                    'away': translate_team_name(away_team),
                    'most_likely': result['靠谱比分'],
                    'win': result['主胜'], 'draw': result['平局'], 'lose': result['客胜'],
                    'time': datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                if len(st.session_state.prediction_history) > 20:
                    st.session_state.prediction_history = st.session_state.prediction_history[-20:]
                st.markdown("---")
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric(f"🏠 {translate_team_name(home_team)} 胜", f"{result['主胜']*100:.1f}%")
                col_r2.metric("🤝 平局", f"{result['平局']*100:.1f}%")
                col_r3.metric(f"✈️ {translate_team_name(away_team)} 胜", f"{result['客胜']*100:.1f}%")
                st.write("**让球胜平负**")
                st.write(f"让球-1: 主胜 {result['让球-1']['主胜']*100:.1f}% | 平 {result['让球-1']['平局']*100:.1f}% | 客胜 {result['让球-1']['客胜']*100:.1f}%")
                st.write(f"让球+1: 主胜 {result['让球+1']['主胜']*100:.1f}% | 平 {result['让球+1']['平局']*100:.1f}% | 客胜 {result['让球+1']['客胜']*100:.1f}%")
                st.write(f"**大小球概率**: {result['大球概率']*100:.1f}%，动态阈值 {result['大球阈值']*100:.1f}%")
                st.write(f"**大球判定**: {result['大球判定']}")
                st.write("**推荐比分**")
                st.write(f"靠谱: {result['靠谱比分']} ({result['靠谱概率']*100:.1f}%)")
                st.write(f"激进: {result['激进比分']} ({result['激进概率']*100:.1f}%)")
                st.write(f"稳健: {result['稳健比分']} ({result['稳健概率']*100:.1f}%)")
                if result.get('analysis'):
                    st.caption(f"📝 {result['analysis']}")
                if result.get('data_source'):
                    st.caption(f"📊 数据来源：{result['data_source']}")
                st.caption("📊 比分概率 Top5")
                score_df = pd.DataFrame(result['比分概率'], columns=["比分", "概率"])
                score_df["概率"] = score_df["概率"].apply(lambda x: f"{x*100:.1f}%")
                st.table(score_df)
                plot_prob_distribution(result['比分概率'])
            except Exception as e:
                st.error(f"预测出错: {e}")
                st.code(traceback.format_exc())


# ===================== 回测模式 =====================
elif mode == "回测":
    st.subheader("📈 回测分析")
    st.markdown("选择要回测的时间范围，系统将模拟预测并统计准确率。")
    latest_date = df_hist['date'].max().date()
    earliest_date = df_hist['date'].min().date()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", value=earliest_date,
                                    min_value=earliest_date, max_value=latest_date)
    with col2:
        end_date = st.date_input("结束日期", value=latest_date,
                                  min_value=earliest_date, max_value=latest_date)
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
                    acc_df = pd.DataFrame([accuracy]).T.rename(columns={0: '值'})
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


# ===================== 周报模式 =====================
elif mode == "周报":
    render_weekly_report(df_hist)


# ===================== 批量预测模式（CSV上传） =====================
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
                            r = predict_match(row['home_team'], row['away_team'],
                                              row['odds_h'], row['odds_d'], row['odds_a'],
                                              df_hist, xgb_model, xgb_over_model)
                            results.append({
                                '主队': translate_team_name(row['home_team']),
                                '客队': translate_team_name(row['away_team']),
                                '主胜概率': f"{r['主胜']*100:.1f}%",
                                '平局概率': f"{r['平局']*100:.1f}%",
                                '客胜概率': f"{r['客胜']*100:.1f}%",
                                '靠谱比分': f"{r['靠谱比分']} ({r['靠谱概率']*100:.1f}%)",
                                '激进比分': f"{r['激进比分']} ({r['激进概率']*100:.1f}%)",
                                '稳健比分': f"{r['稳健比分']} ({r['稳健概率']*100:.1f}%)",
                                '分析摘要': r.get('analysis', ''),
                                '数据来源': r.get('data_source', '模型')
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
