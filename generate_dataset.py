import pandas as pd
import numpy as np
import datetime

ANCHORS = [
    # Malaimanthiripalayam & User Home
    {"name": "Bhagavanth Sakthivel Home, Malaimanthiripalayam", "lat": 10.87354, "lon": 77.18957, "weight": 0.35},
    {"name": "Malaimanthiripalayam Village Center", "lat": 10.87410, "lon": 77.18840, "weight": 0.35},
    {"name": "Malaimanthiripalayam Vinayagar Temple", "lat": 10.87280, "lon": 77.19020, "weight": 0.30},

    # Senjerimalai & Sultanpet Corridors
    {"name": "Senjerimalai Sri Mandhrakiri Murugan Temple (Hilltop Premises)", "lat": 10.82985, "lon": 77.19420, "weight": 0.50},
    {"name": "Senjerimalai Foothills Ratha Mandapam", "lat": 10.83180, "lon": 77.18520, "weight": 0.45},
    {"name": "Senjeri Medu Bus Terminus & Market", "lat": 10.83780, "lon": 77.17790, "weight": 0.60},
    {"name": "Sri Murugan Bakery, Senjeri Medu", "lat": 10.83820, "lon": 77.17840, "weight": 0.45},
    {"name": "Sultanpet Police Station Roundabout", "lat": 10.87480, "lon": 77.15490, "weight": 0.65},
    {"name": "Vadambacheri SH 174 Four Roads", "lat": 10.89180, "lon": 77.17480, "weight": 0.55},
    {"name": "Varapatti Windmill Road", "lat": 10.88490, "lon": 77.11480, "weight": 0.40},
    {"name": "Kallapalayam Junction", "lat": 10.92480, "lon": 77.09780, "weight": 0.50},

    # Kondampatti & Sri Eshwar College Belt
    {"name": "Sri Eshwar College of Engineering (SECE Main Gate)", "lat": 10.82736, "lon": 77.06052, "weight": 0.65},
    {"name": "SECE Academic Block", "lat": 10.82810, "lon": 77.05940, "weight": 0.50},
    {"name": "Kondampatti Village Bus Stop", "lat": 10.83091, "lon": 77.05432, "weight": 0.55},
    {"name": "Kondampatti Mariamman Temple Ground", "lat": 10.83150, "lon": 77.05380, "weight": 0.40},
    {"name": "Kothavadi Lake Breeze View Point", "lat": 10.84310, "lon": 77.04920, "weight": 0.35},
    {"name": "Nallattipalayam Pirivu (SECE Highway Access)", "lat": 10.83415, "lon": 77.02534, "weight": 0.60},
    {"name": "Akshaya College of Engineering Road", "lat": 10.82580, "lon": 77.04180, "weight": 0.50},

    # Kinathukadavu Town & NH 83
    {"name": "Ponmalai Velayudhaswamy Murugan Temple Footpath", "lat": 10.82390, "lon": 77.01420, "weight": 0.45},
    {"name": "Kinathukadavu New Bus Stand", "lat": 10.82110, "lon": 77.01825, "weight": 0.80},
    {"name": "Kinathukadavu Railway Station Road", "lat": 10.82315, "lon": 77.02080, "weight": 0.60},
    {"name": "Kinathukadavu NH 83 Flyover Underpass", "lat": 10.81745, "lon": 77.02160, "weight": 0.75},
    {"name": "Aavin Milk Parlour & Bakery, Kinathukadavu NH", "lat": 10.81980, "lon": 77.02010, "weight": 0.50},
    {"name": "Sulakkal Mariamman Temple Arch", "lat": 10.80190, "lon": 76.99480, "weight": 0.45},
    {"name": "Singarampalayam Pirivu", "lat": 10.80780, "lon": 77.02210, "weight": 0.55},
    {"name": "Kakkadavu Village Junction", "lat": 10.79520, "lon": 77.04180, "weight": 0.45},

    # City Links & Connectors
    {"name": "Othakalmandapam Toll Plaza (NH 83)", "lat": 10.89190, "lon": 76.98520, "weight": 0.85},
    {"name": "Karpagam Academy of Higher Education (KAHE)", "lat": 10.90780, "lon": 76.97790, "weight": 0.75},
    {"name": "Malumichampatti SIDCO Junction", "lat": 10.89490, "lon": 76.98490, "weight": 0.70},
    {"name": "Eachanari Vinayagar Temple Entrance", "lat": 10.93245, "lon": 76.97215, "weight": 0.75},
    {"name": "Sundarapuram Signal Junction", "lat": 10.94190, "lon": 76.97090, "weight": 0.80},
    {"name": "Kurichi Kulam Promenade", "lat": 10.95180, "lon": 76.96790, "weight": 0.50},
    {"name": "Podanur Junction Railway Station", "lat": 10.96280, "lon": 76.99580, "weight": 0.70},
    {"name": "Ukkadam Central Bus Stand & Lakefront", "lat": 10.98590, "lon": 76.96280, "weight": 0.90},
    {"name": "Gandhipuram Central Bus Terminus", "lat": 11.01675, "lon": 76.96775, "weight": 0.90}
]

TOTAL_POINTS = 4000
print(f"Generating {TOTAL_POINTS} road telemetry pinpoints...")
np.random.seed(42)

records = []
start_time = datetime.datetime(2026, 9, 20, 8, 0, 0)
weathers = ["Clear", "Cloudy", "Rain", "Breezy"]
tags = ["Approach", "Junction", "Link Road", "Bypass", "Crossway", "Checkpost"]

points_per_anchor = TOTAL_POINTS // len(ANCHORS)
remainder = TOTAL_POINTS % len(ANCHORS)

for idx, anchor in enumerate(ANCHORS):
    count = points_per_anchor + (1 if idx < remainder else 0)
    for i in range(count):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.exponential(scale=0.0035)
        
        lat = anchor["lat"] + radius * np.cos(angle)
        lon = anchor["lon"] + radius * np.sin(angle)
        
        weight = anchor["weight"]
        rand = np.random.uniform(0, 1)
        
        if rand < weight * 0.60:
            traffic = "High"
            speed = np.random.uniform(12.0, 22.0)
            occupancy = np.random.uniform(75.0, 96.0)
        elif rand < weight * 0.88:
            traffic = "Medium"
            speed = np.random.uniform(24.0, 45.0)
            occupancy = np.random.uniform(40.0, 74.0)
        else:
            traffic = "Low"
            speed = np.random.uniform(46.0, 72.0)
            occupancy = np.random.uniform(10.0, 39.0)
            
        accident = 1 if (traffic == "High" and np.random.uniform(0, 1) < 0.035) else 0
        weather = np.random.choice(weathers, p=[0.60, 0.25, 0.10, 0.05])
        time_offset = np.random.randint(0, 720)
        timestamp = start_time + datetime.timedelta(minutes=time_offset)
        
        records.append({
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Pinpoint_ID": f"CBE-PIN-{1000 + len(records)}",
            "Location_Name": f"{anchor['name']} ({np.random.choice(tags)} {i+1})",
            "Latitude": round(float(lat), 6),
            "Longitude": round(float(lon), 6),
            "Vehicle_Count": int(occupancy * np.random.uniform(1.8, 3.2)),
            "Traffic_Speed_kmh": round(float(speed), 2),
            "Road_Occupancy_%": round(float(occupancy), 2),
            "Traffic_Light_State": np.random.choice(["Red", "Yellow", "Green"], p=[0.4, 0.1, 0.5]),
            "Weather_Condition": weather,
            "Accident_Report": accident,
            "Traffic_Condition": traffic
        })

df = pd.DataFrame(records)
df.to_csv("smart_mobility_dataset.csv", index=False)
print(f"✅ Created smart_mobility_dataset.csv with {len(df)} telemetry rows!")
