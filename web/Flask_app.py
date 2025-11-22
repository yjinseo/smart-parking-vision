from flask import Flask, render_template, jsonify
import json, os
from datetime import datetime

app = Flask(__name__)

STATUS_FILE = "status.json"
HISTORY_FILE = "history_predict.json"


def load_status():
    """status.json 로딩"""
    try:
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    except:
        return {}


def load_history():
    """history_predict.json 로딩"""
    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data[0] if len(data) > 0 else {}
        return data
    except:
        return {}


@app.route("/")
def home():
    data = load_status()

    zones = {}
    score_table = {}   # 추천 알고리즘용 점수 테이블

    for zid, z in data.items():
        free = z.get("free", 0)
        total = z.get("total", 0)

        # 기본 정보 저장
        zones[zid] = {
            "free": free,
            "occupied": z.get("occ", 0),
            "total": total,
            "disabled_free": z.get("disabled_free", 0),
            "disabled_total": z.get("disabled_total", 0)
        }

        # 추천 점수 (0~1)
        if total > 0:
            score = free / total
        else:
            score = 0
        score_table[zid] = score

    # -------------------------------
    # 추천 알고리즘 (가장 free 비율 높은 존)
    # -------------------------------
    if score_table:
        recommended_zone = max(score_table, key=score_table.get)
    else:
        recommended_zone = None

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return render_template(
        "index.html",
        zones=zones,
        update_time=now,
        recommended_zone=recommended_zone  # ⭐ 추가됨!
    )


@app.route("/zone/<zone>")
def zone_page(zone):
    data = load_status()

    if zone not in data:
        return f"Zone {zone} not found", 404

    zone_data = data[zone]

    # -----------------------------
    # SLOT 리스트 변환
    # -----------------------------
    slots_raw = zone_data.get("slots", [])
    slots = []

    for s in slots_raw:
        slot_id = s.get("id", "UNKNOWN")
        is_occ = s.get("occupied", False)
        is_dis = s.get("disabled", False)

        slots.append({
            "id": slot_id,
            "state": "occupied" if is_occ else "free",
            "disabled": is_dis
        })

    # -----------------------------
    # 예측 기반 그래프 생성
    # -----------------------------
    history = load_history().get(zone, {"avg": [3, 4, 5, 6, 7, 8, 9]})
    history_avg = history["avg"]

    curr = zone_data.get("occ", 0)
    future_prediction = [curr + i * 2 for i in range(7)]

    return render_template(
        "zone.html",
        zone=zone,
        slots=slots,
        future_prediction=future_prediction,
        history_avg=history_avg
    )


@app.route("/status")
def api_status():
    """occupancy_video.py → status.json 확인용 API"""
    return jsonify(load_status())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
