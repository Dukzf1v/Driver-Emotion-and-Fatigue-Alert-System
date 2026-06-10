# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import time
import os
import queue
import json
import threading
import sqlite3
import config

app = Flask(__name__)
CORS(app)

DATABASE_FILE = "dms_database.db"
db_lock = threading.Lock()

# Dictionary storing active status of each vehicle
latest_data = {}
listeners = []

# Prevent logging warnings continuously to DB
last_logged_warning = {}  # vehicle_id -> {"state": state, "timestamp": timestamp}

def get_db_connection():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db_lock:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicles (
                vehicle_id TEXT PRIMARY KEY,
                driver_name TEXT,
                last_seen REAL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS warning_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id TEXT,
                driver_name TEXT,
                state TEXT,
                f_score REAL,
                anger_level REAL,
                fear_level REAL,
                timestamp REAL
            )
        """)
        conn.commit()
        conn.close()


def load_vehicles_from_db():
    global latest_data
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT vehicle_id, driver_name, last_seen FROM vehicles")
        rows = cursor.fetchall()
        for row in rows:
            vid = row["vehicle_id"]
            latest_data[vid] = {
                "vehicle_id": vid,
                "driver_name": row["driver_name"],
                "state": "UNKNOWN",
                "f_score": 0.0,
                "anger_level": 0.0,
                "fear_level": 0.0,
                "timestamp": row["last_seen"]
            }
        conn.close()

def db_update_vehicle(vehicle_id, driver_name, timestamp):
    with db_lock:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO vehicles (vehicle_id, driver_name, last_seen)
            VALUES (?, ?, ?)
            ON CONFLICT(vehicle_id) DO UPDATE SET
                driver_name = excluded.driver_name,
                last_seen = excluded.last_seen
        """, (vehicle_id, driver_name, timestamp))
        conn.commit()
        conn.close()

def db_add_warning_log(vehicle_id, driver_name, state, f_score, anger_level, fear_level, timestamp):
    global last_logged_warning
    if state in ("NORMAL", "UNKNOWN"):
        return
        
    # Apply a 10-second debounce mechanism for the same state of the vehicle
    last_log = last_logged_warning.get(vehicle_id)
    if last_log and last_log["state"] == state and (timestamp - last_log["timestamp"] < 10.0):
        return
        
    # Record the latest log timestamp
    last_logged_warning[vehicle_id] = {"state": state, "timestamp": timestamp}
    
    with db_lock:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO warning_logs (vehicle_id, driver_name, state, f_score, anger_level, fear_level, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (vehicle_id, driver_name, state, f_score, anger_level, fear_level, timestamp))
        conn.commit()
        conn.close()

def announce(msg):
    payload = f"data: {json.dumps(msg)}\n\n"
    for q in list(listeners):
        try:
            q.put_nowait(payload)
        except queue.Full:
            pass

def get_active_vehicles():
    now = time.time()
    res = {}
    for vid, vdata in latest_data.items():
        status = "online" if now - vdata["timestamp"] < 4.0 else "offline"
        res[vid] = {**vdata, "status": status}
    return res

def check_offline_loop():
    last_states = {}
    while True:
        time.sleep(1.0)
        now = time.time()
        updates = {}
        for vid, vdata in list(latest_data.items()):
            is_online = (now - vdata["timestamp"] < 4.0)
            was_online = last_states.get(vid, True)
            if is_online != was_online:
                last_states[vid] = is_online
                updates[vid] = "online" if is_online else "offline"
        
        if updates:
            announce({
                "type": "status_change",
                "updates": updates
            })

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/update", methods=["POST"])
def update_data():
    global latest_data
    try:
        data = request.json
        if data:
            vehicle_id = data.get("vehicle_id", config.DEFAULT_VEHICLE_ID)
            driver_name = data.get("driver_name", config.DEFAULT_DRIVER_NAME)
            state = data.get("state", "NORMAL")
            f_score = data.get("f_score", 0.0)
            anger_level = data.get("anger_level", 0.0)
            fear_level = data.get("fear_level", 0.0)
            timestamp = time.time()
            
            # Update information in memory
            latest_data[vehicle_id] = {
                "vehicle_id": vehicle_id,
                "driver_name": driver_name,
                "state": state,
                "f_score": f_score,
                "anger_level": anger_level,
                "fear_level": fear_level,
                "timestamp": timestamp
            }
            
            # Update SQLite database
            db_update_vehicle(vehicle_id, driver_name, timestamp)
            db_add_warning_log(vehicle_id, driver_name, state, f_score, anger_level, fear_level, timestamp)
            
            # Push data via SSE
            msg = {
                "type": "update",
                "vehicle_id": vehicle_id,
                "data": {**latest_data[vehicle_id], "status": "online"}
            }
            announce(msg)
            
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/status", methods=["GET"])
def get_status():
    vehicles = get_active_vehicles()
    if not vehicles:
        return jsonify({
            "state": "UNKNOWN",
            "f_score": 0.0,
            "anger_level": 0.0,
            "fear_level": 0.0,
            "status": "offline"
        })
    first_vehicle = list(vehicles.values())[0]
    return jsonify(first_vehicle)

@app.route("/api/history", methods=["GET"])
def get_history():
    search_q = request.args.get("vehicle_id", "").strip()
    filter_state = request.args.get("state", "").strip()
    
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM warning_logs WHERE 1=1"
        params = []
        
        if search_q:
            query += " AND vehicle_id LIKE ?"
            params.append(f"%{search_q}%")
        if filter_state:
            query += " AND state = ?"
            params.append(filter_state)
            
        query += " ORDER BY timestamp DESC LIMIT 100"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        history = []
        for r in rows:
            history.append({
                "id": r["id"],
                "vehicle_id": r["vehicle_id"],
                "driver_name": r["driver_name"],
                "state": r["state"],
                "f_score": r["f_score"],
                "anger_level": r["anger_level"],
                "fear_level": r["fear_level"],
                "timestamp": r["timestamp"]
            })
        conn.close()
        
    return jsonify(history)

@app.route("/api/drivers", methods=["GET"])
def get_drivers():
    with db_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT vehicle_id, driver_name FROM vehicles")
        vehicles_rows = cursor.fetchall()
        
        drivers_stats = []
        for row in vehicles_rows:
            vid = row["vehicle_id"]
            dname = row["driver_name"]
            
            # Count warnings by each state
            cursor.execute("""
                SELECT state, COUNT(*) as count 
                FROM warning_logs 
                WHERE vehicle_id = ? 
                GROUP BY state
            """, (vid,))
            warn_rows = cursor.fetchall()
            
            warns = {r["state"]: r["count"] for r in warn_rows}
            
            fatigue_cnt = warns.get("FATIGUE", 0)
            distracted_cnt = warns.get("DISTRACTED", 0)
            angry_cnt = warns.get("ANGRY", 0)
            fear_cnt = warns.get("FEAR", 0)
            total_warns = fatigue_cnt + distracted_cnt + angry_cnt + fear_cnt
            
            # Calculate safety score
            score = 100.0 - (fatigue_cnt * 10.0 + distracted_cnt * 8.0 + angry_cnt * 5.0 + fear_cnt * 3.0)
            score = max(0.0, score)
            
            if score >= 80:
                status = "Good"
            elif score >= 50:
                status = "Average"
            else:
                status = "Dangerous"
                
            drivers_stats.append({
                "driver_name": dname,
                "vehicle_id": vid,
                "safety_score": score,
                "total_warnings": total_warns,
                "fatigue_count": fatigue_cnt,
                "distracted_count": distracted_cnt,
                "angry_count": angry_cnt,
                "fear_count": fear_cnt,
                "status": status
            })
            
        conn.close()
        
    return jsonify(drivers_stats)

@app.route("/api/stream")
def stream():
    def event_stream():
        q = queue.Queue(maxsize=20)
        listeners.append(q)
        try:
            initial_msg = {
                "type": "init",
                "vehicles": get_active_vehicles()
            }
            yield f"data: {json.dumps(initial_msg)}\n\n"
            
            while True:
                msg = q.get()
                yield msg
        except GeneratorExit:
            pass
        finally:
            if q in listeners:
                listeners.remove(q)
                
    return Response(event_stream(), mimetype="text/event-stream")

# Initialize SQLite DB and load historical data
init_db()
load_vehicles_from_db()

# Start offline checking thread
t = threading.Thread(target=check_offline_loop, daemon=True)
t.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", config.FLASK_PORT))
    print(f"Starting Central Monitoring Server at http://localhost:{port}")
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    app.run(host=config.FLASK_HOST, port=port, debug=False)
