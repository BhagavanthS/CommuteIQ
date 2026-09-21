import pandas as pd
import numpy as np
import datetime

ANCHORS = [
    # --- Home, Kondampatti & Sri Eshwar College Corridors ---
    {"name": "Bhagavanth Sakthivel Home, Malaimanthiripalayam", "lat": 10.825917, "lon": 77.221222, "weight": 0.35},
    {"name": "Malaimanthiripalayam Village Center", "lat": 10.827100, "lon": 77.220400, "weight": 0.35},
    {"name": "Malaimanthiripalayam Vinayagar Temple", "lat": 10.826400, "lon": 77.222100, "weight": 0.30},
    {"name": "Sri Eshwar College of Engineering (SECE Main Gate)", "lat": 10.827360, "lon": 77.060520, "weight": 0.65},
    {"name": "SECE Academic Block & Labs", "lat": 10.828100, "lon": 77.059400, "weight": 0.50},
    {"name": "Kondampatti Village Bus Stop", "lat": 10.830910, "lon": 77.054320, "weight": 0.55},
    {"name": "Kondampatti Mariamman Temple Ground", "lat": 10.831500, "lon": 77.053800, "weight": 0.40},
    {"name": "Kothavadi Lake Breeze View Point", "lat": 10.843100, "lon": 77.049200, "weight": 0.35},
    {"name": "Nallattipalayam Pirivu (SECE Highway Access)", "lat": 10.834150, "lon": 77.025340, "weight": 0.60},
    {"name": "Akshaya College of Engineering Road", "lat": 10.825800, "lon": 77.041800, "weight": 0.50},

    # --- Senjerimalai, Sultanpet & Rural Link Corridors ---
    {"name": "Senjerimalai Sri Mandhrakiri Murugan Temple (Hilltop Premises)", "lat": 10.829850, "lon": 77.194200, "weight": 0.50},
    {"name": "Senjerimalai Hill Ghat Road Arch (Foothills)", "lat": 10.832100, "lon": 77.191800, "weight": 0.45},
    {"name": "Senjeri Medu Bus Terminus & Market", "lat": 10.837800, "lon": 77.177900, "weight": 0.65},
    {"name": "Sri Murugan Bakery, Senjeri Medu", "lat": 10.838200, "lon": 77.178400, "weight": 0.45},
    {"name": "Sultanpet Police Station Roundabout", "lat": 10.874800, "lon": 77.154900, "weight": 0.65},
    {"name": "Vadambacheri SH 174 Four Roads", "lat": 10.891800, "lon": 77.174800, "weight": 0.55},
    {"name": "Varapatti Windmill Road", "lat": 10.884900, "lon": 77.114800, "weight": 0.40},
    {"name": "Kallapalayam Junction", "lat": 10.924800, "lon": 77.097800, "weight": 0.50},

    # --- Kinathukadavu & Pollachi Arterial Sector ---
    {"name": "Ponmalai Velayudhaswamy Murugan Temple Footpath", "lat": 10.823900, "lon": 77.014200, "weight": 0.45},
    {"name": "Kinathukadavu New Bus Stand", "lat": 10.821100, "lon": 77.018250, "weight": 0.80},
    {"name": "Kinathukadavu Railway Station Road", "lat": 10.823150, "lon": 77.020800, "weight": 0.60},
    {"name": "Kinathukadavu NH 83 Flyover Underpass", "lat": 10.817450, "lon": 77.021600, "weight": 0.75},
    {"name": "Aavin Milk Parlour & Bakery, Kinathukadavu NH", "lat": 10.819800, "lon": 77.020100, "weight": 0.50},
    {"name": "Sulakkal Mariamman Temple Arch", "lat": 10.801900, "lon": 76.994800, "weight": 0.45},
    {"name": "Singarampalayam Pirivu", "lat": 10.807800, "lon": 77.022100, "weight": 0.55},
    {"name": "Kakkadavu Village Junction", "lat": 10.795200, "lon": 77.041800, "weight": 0.45},
    {"name": "Pollachi Central Bus Stand", "lat": 10.661200, "lon": 77.006500, "weight": 0.85},
    {"name": "Pollachi Gandhi Statue Roundabout", "lat": 10.658700, "lon": 77.009100, "weight": 0.80},
    {"name": "Mahalingam College of Engineering (MCET), Pollachi", "lat": 10.682100, "lon": 77.034500, "weight": 0.60},
    {"name": "Achipatti Industrial Estate, Pollachi", "lat": 10.697400, "lon": 77.019800, "weight": 0.65},

    # --- NH 83 Urban Approach & Madukkarai Corridor ---
    {"name": "Othakalmandapam Toll Plaza (NH 83)", "lat": 10.891900, "lon": 76.985200, "weight": 0.85},
    {"name": "Premier Mills Bus Stop, Othakalmandapam", "lat": 10.887200, "lon": 76.987100, "weight": 0.70},
    {"name": "Karpagam Academy of Higher Education (KAHE Main Gate)", "lat": 10.907800, "lon": 76.977900, "weight": 0.75},
    {"name": "Karpagam Hospital & Medical College", "lat": 10.912500, "lon": 76.974500, "weight": 0.70},
    {"name": "Malumichampatti SIDCO Industrial Junction", "lat": 10.894900, "lon": 76.984900, "weight": 0.75},
    {"name": "Eachanari Vinayagar Temple Entrance", "lat": 10.932450, "lon": 76.972150, "weight": 0.80},
    {"name": "Rathinam Techzone & College, Eachanari", "lat": 10.929800, "lon": 76.965400, "weight": 0.75},
    {"name": "Madukkarai Market & Cement Factory Road", "lat": 10.902300, "lon": 76.958200, "weight": 0.70},
    {"name": "Sundarapuram Signal Junction", "lat": 10.941900, "lon": 76.970900, "weight": 0.85},
    {"name": "Kurichi Kulam Promenade & Lakeview", "lat": 10.951800, "lon": 76.967900, "weight": 0.55},
    {"name": "Podanur Junction Railway Station", "lat": 10.962800, "lon": 76.995800, "weight": 0.75},
    {"name": "Vellalore Bus Terminus Hub", "lat": 10.958900, "lon": 77.021500, "weight": 0.65},

    # --- Coimbatore City Core & Commercial Hubs ---
    {"name": "Ukkadam Central Bus Stand & Lakefront", "lat": 10.985900, "lon": 76.962800, "weight": 0.90},
    {"name": "Town Hall & Clock Tower Junction", "lat": 10.995400, "lon": 76.963200, "weight": 0.90},
    {"name": "Coimbatore Railway Junction (Main Entrance)", "lat": 10.997800, "lon": 76.968500, "weight": 0.85},
    {"name": "Gandhipuram Central Bus Terminus", "lat": 11.016750, "lon": 76.967750, "weight": 0.90},
    {"name": "Cross Cut Road Commercial Shopping Sector", "lat": 11.019500, "lon": 76.968200, "weight": 0.90},
    {"name": "100 Feet Road Signal, Gandhipuram", "lat": 11.021200, "lon": 76.964100, "weight": 0.85},
    {"name": "Brookefields Mall, Sukrawarpettai", "lat": 11.006800, "lon": 76.958200, "weight": 0.85},
    {"name": "RS Puram DB Road Roundabout", "lat": 11.008900, "lon": 76.946500, "weight": 0.80},
    {"name": "Prozone Mall, Sathy Road", "lat": 11.054200, "lon": 76.992800, "weight": 0.85},

    # --- Avinashi Road, Airport, Peelamedu & Singanallur ---
    {"name": "Lakshmi Mills Signal Junction", "lat": 11.009800, "lon": 76.985400, "weight": 0.85},
    {"name": "Nava India Signal, Avinashi Road", "lat": 11.014500, "lon": 76.996500, "weight": 0.85},
    {"name": "PSG College of Technology, Peelamedu", "lat": 11.024500, "lon": 77.003200, "weight": 0.80},
    {"name": "Peelamedu Railway Station Road", "lat": 11.031200, "lon": 77.011200, "weight": 0.70},
    {"name": "Hope College Flyover Junction", "lat": 11.028900, "lon": 77.018900, "weight": 0.85},
    {"name": "Coimbatore International Airport (CJB Terminal)", "lat": 11.029750, "lon": 77.043350, "weight": 0.75},
    {"name": "Tidel Park Coimbatore, Civil Aerodrome", "lat": 11.027500, "lon": 77.026800, "weight": 0.80},
    {"name": "KMCH Hospital, Avinashi Road", "lat": 11.042500, "lon": 77.054200, "weight": 0.75},
    {"name": "Singanallur Bus Stand & Signal", "lat": 10.998900, "lon": 77.025400, "weight": 0.85},
    {"name": "Ramanathapuram Signal, Trichy Road", "lat": 10.993400, "lon": 76.994200, "weight": 0.85},
    {"name": "Ondipudur Flyover & Bus Depot", "lat": 10.997200, "lon": 77.058200, "weight": 0.80},

    # --- IT Corridor & Sathy Road Sector ---
    {"name": "Saravanampatti Four Roads Junction", "lat": 11.079500, "lon": 76.998500, "weight": 0.90},
    {"name": "CHIL SEZ IT Park (Keeranatham)", "lat": 11.092400, "lon": 77.006800, "weight": 0.85},
    {"name": "Kumaraguru College of Technology (KCT Gate)", "lat": 11.080500, "lon": 76.991200, "weight": 0.75},
    {"name": "Ganapathy Bus Stand & Market", "lat": 11.038500, "lon": 76.979800, "weight": 0.80},
    {"name": "Vilankurichi Road Tech Corridor", "lat": 11.054200, "lon": 77.018900, "weight": 0.75},
    {"name": "Kovilpalayam NH 209 Toll Junction", "lat": 11.141200, "lon": 77.038500, "weight": 0.70},
    {"name": "Annur Bus Terminus", "lat": 11.234500, "lon": 77.132500, "weight": 0.65},

    # --- Western & Northern Corridors (Mettupalayam, Thudiyalur, Perur) ---
    {"name": "Thudiyalur Junction & Market", "lat": 11.079200, "lon": 76.942500, "weight": 0.85},
    {"name": "Koundampalayam Flyover Junction", "lat": 11.042500, "lon": 76.942100, "weight": 0.80},
    {"name": "Saibaba Colony Signal, NSR Road", "lat": 11.028900, "lon": 76.948500, "weight": 0.80},
    {"name": "Perur Patteeswarar Temple Arch", "lat": 10.971200, "lon": 76.918900, "weight": 0.70},
    {"name": "Kuniyamuthur Signal, Palakkad Road", "lat": 10.962500, "lon": 76.945800, "weight": 0.85},
    {"name": "Kovaipudur Roundabout & Ashram", "lat": 10.938500, "lon": 76.924500, "weight": 0.60},
    {"name": "Marudhamalai Hill Temple Foothills", "lat": 11.046500, "lon": 76.852400, "weight": 0.65},
    {"name": "Mettupalayam Bus Terminus & Ooty Ghat Access", "lat": 11.298500, "lon": 76.942500, "weight": 0.75},
    {"name": "Karamadai Ranganathar Temple Junction", "lat": 11.242500, "lon": 76.958200, "weight": 0.65},

    # --- Eastern Corridors (Sulur, Neelambur & Karumathampatti) ---
    {"name": "Sulur Ranganatha Temple Lake & Bus Stand", "lat": 11.028500, "lon": 77.125400, "weight": 0.75},
    {"name": "Neelambur L&T Bypass Toll Junction", "lat": 11.054200, "lon": 77.085400, "weight": 0.85},
    {"name": "Karumathampatti Junction (NH 544)", "lat": 11.112400, "lon": 77.182500, "weight": 0.80},
    {"name": "Palladam Four Roads Highway Junction", "lat": 10.998500, "lon": 77.284500, "weight": 0.85}
]

TOTAL_POINTS = 60000
print(f"Generating {TOTAL_POINTS} precision road telemetry pinpoints across Coimbatore district...")
np.random.seed(42)

records = []
start_time = datetime.datetime(2026, 9, 21, 8, 0, 0)
weathers = ["Clear", "Cloudy", "Rain", "Breezy"]
tags = ["Approach", "Junction", "Link Road", "Bypass", "Crossway", "Checkpost", "Ghat Sector", "Speed Corridor", "Service Road"]

points_per_anchor = TOTAL_POINTS // len(ANCHORS)
remainder = TOTAL_POINTS % len(ANCHORS)

for idx, anchor in enumerate(ANCHORS):
    count = points_per_anchor + (1 if idx < remainder else 0)
    for i in range(count):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.exponential(scale=0.0040)
        
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
            
        accident = 1 if (traffic == "High" and np.random.uniform(0, 1) < 0.03) else 0
        weather = np.random.choice(weathers, p=[0.60, 0.25, 0.10, 0.05])
        time_offset = np.random.randint(0, 720)
        timestamp = start_time + datetime.timedelta(minutes=time_offset)
        
        records.append({
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Pinpoint_ID": f"CBE-PIN-{10000 + len(records)}",
            "Location_Name": f"{anchor['name']} ({np.random.choice(tags)} {i+1})",
            "Latitude": round(float(lat), 6),
            "Longitude": round(float(lon), 6),
            "Vehicle_Count": int(occupancy * np.random.uniform(1.8, 3.2)),
            "Traffic_Speed_kmh": round(float(speed), 2),
            "Road_Occupancy_%": round(float(occupancy), 2),
            "Weather_Condition": weather,
            "Accident_Report": accident,
            "Traffic_Condition": traffic
        })

df = pd.DataFrame(records)
df.to_csv("smart_mobility_dataset.csv", index=False)
print(f"✅ Created smart_mobility_dataset.csv with {len(df)} telemetry rows!")
