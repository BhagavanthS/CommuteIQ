import os
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

gemini_client = None
if GEMINI_API_KEY and not GEMINI_API_KEY.startswith("your_actual"):
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Online AI: Google Gemini Client initialized.")
    except Exception as e:
        print(f"⚠️ Gemini Init Warning: {e}")
        gemini_client = None
else:
    print("ℹ️ Offline AI: No GEMINI_API_KEY found in .env.")

app = Flask(__name__)
CORS(app)

# Authorized Geographic Bounding Box for Full Coimbatore District & Rural Corridors
GEO_BOUNDS = {
    "min_lat": 10.6000,
    "max_lat": 11.3500,
    "min_lon": 76.7000,
    "max_lon": 77.4000
}

def is_within_bounds(lat, lon):
    return (GEO_BOUNDS["min_lat"] <= lat <= GEO_BOUNDS["max_lat"]) and \
           (GEO_BOUNDS["min_lon"] <= lon <= GEO_BOUNDS["max_lon"])

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

PRESET_LANDMARKS = [
    # Rural & Home Belt
    {"name": "Bhagavanth Sakthivel Home, Malaimanthiripalayam", "category": "Home & Residence", "lat": 10.825917, "lon": 77.221222},
    {"name": "Malaimanthiripalayam Village Center", "category": "Village", "lat": 10.827100, "lon": 77.220400},
    {"name": "Malaimanthiripalayam Vinayagar Temple", "category": "Temple", "lat": 10.826400, "lon": 77.222100},
    {"name": "Sri Eshwar College of Engineering (SECE Main Gate)", "category": "College", "lat": 10.827360, "lon": 77.060520},
    {"name": "SECE Academic Block", "category": "College", "lat": 10.828100, "lon": 77.059400},
    {"name": "Kondampatti Village Bus Stop", "category": "Village", "lat": 10.830910, "lon": 77.054320},
    {"name": "Kondampatti Mariamman Temple Ground", "category": "Temple", "lat": 10.831500, "lon": 77.053800},
    {"name": "Kothavadi Lake Breeze View Point", "category": "Landmark", "lat": 10.843100, "lon": 77.049200},
    {"name": "Nallattipalayam Pirivu (SECE Highway Access)", "category": "Junction", "lat": 10.834150, "lon": 77.025340},
    {"name": "Akshaya College of Engineering Road", "category": "College", "lat": 10.825800, "lon": 77.041800},

    # Senjerimalai & Sultanpet
    {"name": "Senjerimalai Sri Mandhrakiri Murugan Temple (Hilltop Premises)", "category": "Temple", "lat": 10.829850, "lon": 77.194200},
    {"name": "Senjerimalai Hill Ghat Road Arch (Foothills)", "category": "Junction", "lat": 10.832100, "lon": 77.191800},
    {"name": "Senjeri Medu Bus Terminus & Market", "category": "Transit", "lat": 10.837800, "lon": 77.177900},
    {"name": "Sri Murugan Bakery, Senjeri Medu", "category": "Shop", "lat": 10.838200, "lon": 77.178400},
    {"name": "Sultanpet Police Station Roundabout", "category": "Junction", "lat": 10.874800, "lon": 77.154900},
    {"name": "Vadambacheri SH 174 Four Roads", "category": "Junction", "lat": 10.891800, "lon": 77.174800},
    {"name": "Varapatti Windmill Road", "category": "Village", "lat": 10.884900, "lon": 77.114800},
    {"name": "Kallapalayam Junction", "category": "Industrial", "lat": 10.924800, "lon": 77.097800},

    # Kinathukadavu & Pollachi
    {"name": "Ponmalai Velayudhaswamy Murugan Temple Footpath", "category": "Temple", "lat": 10.823900, "lon": 77.014200},
    {"name": "Kinathukadavu New Bus Stand", "category": "Transit", "lat": 10.821100, "lon": 77.018250},
    {"name": "Kinathukadavu Railway Station Road", "category": "Transit", "lat": 10.823150, "lon": 77.020800},
    {"name": "Kinathukadavu NH 83 Flyover Underpass", "category": "Junction", "lat": 10.817450, "lon": 77.021600},
    {"name": "Aavin Milk Parlour & Bakery, Kinathukadavu NH", "category": "Shop", "lat": 10.819800, "lon": 77.020100},
    {"name": "Sulakkal Mariamman Temple Arch", "category": "Temple", "lat": 10.801900, "lon": 76.994800},
    {"name": "Singarampalayam Pirivu", "category": "Village", "lat": 10.807800, "lon": 77.022100},
    {"name": "Kakkadavu Village Junction", "category": "Village", "lat": 10.795200, "lon": 77.041800},
    {"name": "Pollachi Central Bus Stand", "category": "Transit", "lat": 10.661200, "lon": 77.006500},
    {"name": "Mahalingam College of Engineering (MCET), Pollachi", "category": "College", "lat": 10.682100, "lon": 77.034500},

    # Arterial Corridors & City
    {"name": "Othakalmandapam Toll Plaza (NH 83)", "category": "Toll", "lat": 10.891900, "lon": 76.985200},
    {"name": "Karpagam Academy of Higher Education (KAHE)", "category": "College", "lat": 10.907800, "lon": 76.977900},
    {"name": "Malumichampatti SIDCO Junction", "category": "Industrial", "lat": 10.894900, "lon": 76.984900},
    {"name": "Eachanari Vinayagar Temple Entrance", "category": "Temple", "lat": 10.932450, "lon": 76.972150},
    {"name": "Rathinam Techzone & College, Eachanari", "category": "IT / College", "lat": 10.929800, "lon": 76.965400},
    {"name": "Sundarapuram Signal Junction", "category": "Junction", "lat": 10.941900, "lon": 76.970900},
    {"name": "Kurichi Kulam Promenade", "category": "Landmark", "lat": 10.951800, "lon": 76.967900},
    {"name": "Podanur Junction Railway Station", "category": "Transit", "lat": 10.962800, "lon": 76.995800},
    {"name": "Ukkadam Central Bus Stand & Lakefront", "category": "Transit", "lat": 10.985900, "lon": 76.962800},
    {"name": "Town Hall & Clock Tower Junction", "category": "Commercial", "lat": 10.995400, "lon": 76.963200},
    {"name": "Coimbatore Railway Junction (Main Entrance)", "category": "Transit", "lat": 10.997800, "lon": 76.968500},
    {"name": "Gandhipuram Central Bus Terminus", "category": "Transit", "lat": 11.016750, "lon": 76.967750},
    {"name": "Cross Cut Road Commercial Sector", "category": "Shopping", "lat": 11.019500, "lon": 76.968200},
    {"name": "Brookefields Mall, Sukrawarpettai", "category": "Mall", "lat": 11.006800, "lon": 76.958200},
    {"name": "RS Puram DB Road Roundabout", "category": "Commercial", "lat": 11.008900, "lon": 76.946500},
    {"name": "PSG College of Technology, Peelamedu", "category": "College", "lat": 11.024500, "lon": 77.003200},
    {"name": "Hope College Flyover Junction", "category": "Junction", "lat": 11.028900, "lon": 77.018900},
    {"name": "Coimbatore International Airport (CJB Terminal)", "category": "Airport", "lat": 11.029750, "lon": 77.043350},
    {"name": "Tidel Park Coimbatore, Civil Aerodrome", "category": "IT Hub", "lat": 11.027500, "lon": 77.026800},
    {"name": "Singanallur Bus Stand & Signal", "category": "Transit", "lat": 10.998900, "lon": 77.025400},
    {"name": "Ramanathapuram Signal, Trichy Road", "category": "Junction", "lat": 10.993400, "lon": 76.994200},
    {"name": "Saravanampatti Four Roads Junction", "category": "IT Hub", "lat": 11.079500, "lon": 76.998500},
    {"name": "CHIL SEZ IT Park (Keeranatham)", "category": "IT Hub", "lat": 11.092400, "lon": 77.006800},
    {"name": "Kumaraguru College of Technology (KCT Gate)", "category": "College", "lat": 11.080500, "lon": 76.991200},
    {"name": "Thudiyalur Junction & Market", "category": "Commercial", "lat": 11.079200, "lon": 76.942500},
    {"name": "Perur Patteeswarar Temple Arch", "category": "Temple", "lat": 10.971200, "lon": 76.918900},
    {"name": "Kuniyamuthur Signal, Palakkad Road", "category": "Junction", "lat": 10.962500, "lon": 76.945800},
    {"name": "Sulur Ranganatha Temple Lake & Bus Stand", "category": "Transit", "lat": 11.028500, "lon": 77.125400},
    {"name": "Neelambur L&T Bypass Toll Junction", "category": "Junction", "lat": 11.054200, "lon": 77.085400},
    {"name": "Karumathampatti Junction (NH 544)", "category": "Junction", "lat": 11.112400, "lon": 77.182500},
    {"name": "Palladam Four Roads Highway Junction", "category": "Transit", "lat": 10.998500, "lon": 77.284500},
    {"name": "Mettupalayam Bus Terminus & Ooty Ghat Access", "category": "Transit", "lat": 11.298500, "lon": 76.942500}
]

print("Loading 60,000 smart mobility telemetry records...")
try:
    raw_df = pd.read_csv("smart_mobility_dataset.csv")
    DATASET_METADATA = {
        "avg_speed": round(float(raw_df['Traffic_Speed_kmh'].mean()), 1),
        "weather_conditions": raw_df['Weather_Condition'].dropna().unique().tolist(),
        "total_records": len(raw_df)
    }
    print(f"Loaded {len(raw_df)} telemetry pinpoints.")
except Exception:
    raw_df = pd.DataFrame()
    DATASET_METADATA = {"avg_speed": 42.0, "weather_conditions": ["Clear", "Cloudy", "Rain"], "total_records": 60000}

def get_closest_landmark(lat, lon):
    return min(PRESET_LANDMARKS, key=lambda lm: haversine(lat, lon, lm['lat'], lm['lon']))['name']

def get_traffic_telemetry_for_point(lat, lon):
    if len(raw_df) > 0:
        hash_val = int(abs(hash((round(lat, 4), round(lon, 4)))))
        sample_row = raw_df.iloc[hash_val % len(raw_df)]
        return {
            "traffic": str(sample_row['Traffic_Condition']),
            "speed": round(float(sample_row['Traffic_Speed_kmh']), 1),
            "weather": str(sample_row['Weather_Condition'])
        }
    return {"traffic": "Low", "speed": 45.0, "weather": "Clear"}

def query_osrm_single(coords_str):
    url = f"https://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=geojson&steps=true"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get("routes"):
                r = data["routes"][0]
                coords = [[c[1], c[0]] for c in r["geometry"]["coordinates"]]
                return coords, round(r["distance"] / 1000.0, 2), round(r["duration"] / 60.0, 1)
    except Exception:
        pass
    return None, None, None

def fetch_multi_routes_detailed(start_lat, start_lon, end_lat, end_lon):
    routes_found = []
    
    # 1. Main Direct Road (Shortest / Optimal)
    coords_main, dist_main, dur_main = query_osrm_single(f"{start_lon},{start_lat};{end_lon},{end_lat}")
    if coords_main:
        routes_found.append({
            "id": 0,
            "title": "Fastest Route (Direct Road)",
            "tag": "Optimal Flow & Shortest",
            "coords": coords_main,
            "distance_km": dist_main,
            "time_mins": dur_main
        })

    # 2. Alternative via Sultanpet Corridor
    via1_lat, via1_lon = 10.87480, 77.15490
    coords_alt1, dist_alt1, dur_alt1 = query_osrm_single(f"{start_lon},{start_lat};{via1_lon},{via1_lat};{end_lon},{end_lat}")
    if coords_alt1 and dist_main and abs(dist_alt1 - dist_main) > 0.4:
        routes_found.append({
            "id": 1,
            "title": "Via Sultanpet Corridor",
            "tag": "State Highway 174",
            "coords": coords_alt1,
            "distance_km": dist_alt1,
            "time_mins": dur_alt1
        })

    # 3. Alternative via Senjeri Medu Bypass
    via2_lat, via2_lon = 10.83780, 77.17790
    coords_alt2, dist_alt2, dur_alt2 = query_osrm_single(f"{start_lon},{start_lat};{via2_lon},{via2_lat};{end_lon},{end_lat}")
    if coords_alt2 and dist_main and abs(dist_alt2 - dist_main) > 0.4:
        routes_found.append({
            "id": 2,
            "title": "Via Senjeri Medu Bypass",
            "tag": "Village Link Arterial",
            "coords": coords_alt2,
            "distance_km": dist_alt2,
            "time_mins": dur_alt2
        })

    if not routes_found:
        d = round(haversine(start_lat, start_lon, end_lat, end_lon), 2)
        routes_found.append({
            "id": 0,
            "title": "Direct Road Path",
            "tag": "Standard",
            "coords": [[start_lat, start_lon], [end_lat, end_lon]],
            "distance_km": d,
            "time_mins": round((d / 38.0) * 60, 1)
        })

    routes_found.sort(key=lambda x: x['time_mins'])
    for idx, r in enumerate(routes_found):
        r['id'] = idx
        if idx == 0:
            r['tag'] = "Fastest & Recommended"
            
    return routes_found

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/places")
def get_places():
    return jsonify(PRESET_LANDMARKS)

@app.route("/api/status")
def get_status():
    return jsonify({
        "online": gemini_client is not None,
        "model": "gemini-3.6-flash" if gemini_client else "offline",
        "total_points": DATASET_METADATA["total_records"],
        "bounds": GEO_BOUNDS
    })

@app.route("/api/search")
def search_places():
    q = request.args.get("q", "").strip()
    if not q or len(q) < 2:
        return jsonify([])

    results = []
    q_lower = q.lower()
    for lm in PRESET_LANDMARKS:
        if q_lower in lm['name'].lower() or q_lower in lm['category'].lower():
            results.append(lm)

    # Search OpenStreetMap via Nominatim with smart dual query strategy
    queries = [f"{q}, Coimbatore", q]
    headers = {"User-Agent": "CommuteIQ-Platform/5.0 (https://commuteiq-org.onrender.com)"}
    
    for query_str in queries:
        if len(results) >= 8:
            break
        try:
            params = {
                "q": query_str,
                "format": "json",
                "addressdetails": 1,
                "limit": 10,
                "viewbox": f"{GEO_BOUNDS['min_lon']},{GEO_BOUNDS['max_lat']},{GEO_BOUNDS['max_lon']},{GEO_BOUNDS['min_lat']}",
                "bounded": 0
            }
            resp = requests.get("https://nominatim.openstreetmap.org/search", params=params, headers=headers, timeout=3.5)
            if resp.status_code == 200:
                for item in resp.json():
                    lat = float(item["lat"])
                    lon = float(item["lon"])
                    if is_within_bounds(lat, lon):
                        parts = [p.strip() for p in item.get("display_name", "").split(",")]
                        short_name = ", ".join(parts[:3])
                        cat = item.get("type", "Place").replace("_", " ").title()
                        results.append({
                            "name": short_name,
                            "category": cat,
                            "lat": lat,
                            "lon": lon
                        })
        except Exception:
            pass

    seen = set()
    unique_results = []
    for r in results:
        if r['name'] not in seen:
            seen.add(r['name'])
            unique_results.append(r)

    if not unique_results and len(q) >= 3:
        return jsonify([{
            "name": f"'{q}' is outside permitted coverage",
            "category": "Restricted Region",
            "restricted": True,
            "lat": 10.82736,
            "lon": 77.06052
        }])

    return jsonify(unique_results[:12])

@app.route("/api/route", methods=["POST"])
def calculate_route():
    data = request.json
    start_lat, start_lon = float(data['start_lat']), float(data['start_lon'])
    end_lat, end_lon = float(data['end_lat']), float(data['end_lon'])

    if not is_within_bounds(start_lat, start_lon) or not is_within_bounds(end_lat, end_lon):
        return jsonify({
            "status": "restricted",
            "message": "Telemetry coverage is restricted strictly to the Coimbatore district and rural corridor network."
        })

    start_place = get_closest_landmark(start_lat, start_lon)
    dest_place = get_closest_landmark(end_lat, end_lon)

    raw_routes = fetch_multi_routes_detailed(start_lat, start_lon, end_lat, end_lon)
    
    processed_routes = []
    for r in raw_routes:
        coords = r["coords"]
        chunk_size = max(1, len(coords) // 16)
        segments = []
        waypoint_places = []

        for i in range(0, len(coords) - 1, chunk_size):
            chunk = coords[i : min(i + chunk_size + 1, len(coords))]
            mid_pt = chunk[len(chunk) // 2]
            telemetry = get_traffic_telemetry_for_point(mid_pt[0], mid_pt[1])
            place_at_pt = get_closest_landmark(mid_pt[0], mid_pt[1])

            segments.append({
                "coords": chunk,
                "traffic": telemetry["traffic"],
                "speed": telemetry["speed"],
                "place": place_at_pt
            })

            if not waypoint_places or waypoint_places[-1]["place"] != place_at_pt:
                waypoint_places.append({
                    "place": place_at_pt,
                    "lat": mid_pt[0],
                    "lon": mid_pt[1],
                    "traffic": telemetry["traffic"],
                    "speed": telemetry["speed"]
                })

        high_count = sum(1 for s in segments if s["traffic"] == "High")
        congestion_risk = round((high_count / max(1, len(segments))) * 100)

        processed_routes.append({
            "id": r["id"],
            "title": r["title"],
            "tag": r["tag"],
            "distance_km": r["distance_km"],
            "time_mins": r["time_mins"],
            "congestion_risk": congestion_risk,
            "segments": segments,
            "waypoint_places": waypoint_places
        })

    return jsonify({
        "status": "success",
        "start_place": start_place,
        "dest_place": dest_place,
        "routes": processed_routes
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("message", "").strip()
    route_context = data.get("route_context", {})

    active_route = route_context.get("active_route", {})
    all_suggestions = route_context.get("all_routes_summary", [])
    
    suggestions_text = "\n".join([
        f"- {r.get('title')}: {r.get('time_mins')} min ({r.get('distance_km')} km) | Congestion: {r.get('congestion_risk')}% | Tag: {r.get('tag')}"
        for r in all_suggestions
    ])

    if gemini_client:
        try:
            from google.genai import types
            system_instruction = f"""
You are the CommuteIQ Traffic Intelligence Agent, specialized in analyzing the entire Coimbatore district network (including Kinathukadavu, Kondampatti, Malaimanthiripalayam, Senjerimalai, Peelamedu, Gandhipuram, Saravanampatti, and Pollachi).
The sensor network is calibrated with {DATASET_METADATA['total_records']} telemetry pinpoints.
Average Network Speed: {DATASET_METADATA['avg_speed']} km/h.
Observed Weather: {', '.join(DATASET_METADATA['weather_conditions'])}.

RESTRICTION NOTICE:
If the user asks about other cities or states (Chennai, Bangalore, Kerala, Delhi), clearly state:
"CommuteIQ data is strictly restricted to the Coimbatore district and surrounding rural corridors."

Always maintain 100% numerical consistency with the active route and suggestion box options below.
Answer concisely and highlight key route names, ETAs, and road speeds using bold formatting (**bold**).
"""
            context_prompt = f"""
ORIGIN: {route_context.get('start_place', 'N/A')}
DESTINATION: {route_context.get('dest_place', 'N/A')}

ACTIVE ROUTE:
- Name: {active_route.get('title', 'Fastest Route')}
- Travel Time: {active_route.get('time_mins', 'N/A')} min
- Distance: {active_route.get('distance_km', 'N/A')} km
- Congestion Risk: {active_route.get('congestion_risk', 'N/A')}%
- Active Corridors & Speed: {active_route.get('waypoint_names', [])}

SUGGESTION DRAWER ALTERNATIVES:
{suggestions_text}

USER QUESTION:
{user_query}
"""
            response = gemini_client.models.generate_content(
                model="gemini-3.6-flash",
                contents=context_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2
                )
            )
            return jsonify({"status": "success", "mode": "online", "reply": response.text})
        except Exception as e:
            print(f"❌ Online API Call Error: {e}")

    # Fallback to local offline telemetry
    q = user_query.lower()
    if any(city in q for city in ["chennai", "bangalore", "bengaluru", "delhi", "mumbai", "kerala", "kochi", "madurai"]):
        reply = "⚠️ **Region Restriction:** CommuteIQ sensor telemetry is restricted exclusively to the Coimbatore district corridor network. Data for other cities and states is unavailable."
    elif "traffic" in q or "jam" in q or "speed" in q or "fast" in q:
        reply = f"The selected **{active_route.get('title')}** takes **{active_route.get('time_mins')} min** ({active_route.get('distance_km')} km) with a **{active_route.get('congestion_risk')}%** congestion risk."
    elif "time" in q or "eta" in q:
        reply = f"⏱️ Active Route ETA: **{active_route.get('time_mins')} mins** ({active_route.get('distance_km')} km)."
    else:
        reply = f"📍 Active route: From **{route_context.get('start_place')}** to **{route_context.get('dest_place')}** via **{active_route.get('title')}**."

    return jsonify({"status": "success", "mode": "offline", "reply": reply})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
