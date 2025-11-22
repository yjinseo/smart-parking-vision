# -----------------------------------------------------------
# app.py : 예측 기반 스마트 주차장 추천 시스템 (LSTM 포함)
# -----------------------------------------------------------
from flask import Flask, jsonify, request
from flask import send_from_directory
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import random, json, os

app = Flask(__name__)

@app.route("/")
def home():
    return send_from_directory(os.path.join(os.getcwd(), "templates"), "index.html")

# ============================================================================
# 1. 기본 설정
# ============================================================================
ZONES = {"A":16, "B":6, "C":14}              # 총 주차칸 수
DISABLED = {"C":["C13","C14"]}              # 장애인구역
HISTORY_FILE = "history_predict.json"       # 과거 저장 파일

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE,"w") as f:
        json.dump([], f)

# 실시간 상태
status_data = {
    zone: {"total":ZONES[zone], "free":ZONES[zone], "occupied":0}
    for zone in ZONES
}

# ============================================================================
# 2. 가상 데이터 생성 (현실적인 패턴 기반)
# ============================================================================
def generate_fake_history(hours=12):
    """
    hours 시간 동안 1분 간격으로 데이터 생성
    실제 주차 패턴을 반영한 랜덤 시계열
    """
    total_minutes = hours * 60
    hist = []

    for minute in range(total_minutes):
        time_ratio = minute / total_minutes  # 0~1 사이

        record = {"time": minute, "zones": {}}

        # 각 Zone별 현실적 패턴
        for z, total in ZONES.items():
            # 기본 패턴 생성
            base = (
                0.2 +                         # 기본 점유율
                0.6 * np.exp(-((time_ratio-0.4)**2) * 12)  # 12시~15시 피크
            )

            # 소규모 랜덤성 추가
            noise = random.uniform(-0.05, 0.05)
            ratio = np.clip(base + noise, 0, 1)

            free = int(total * (1 - ratio))

            record["zones"][z] = {
                "free": free,
                "total": total
            }

        hist.append(record)

    return hist


# ============================================================================
# 3. LSTM 예측 모델 정의
# ============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out,_ = self.lstm(x)
        out = self.fc(out[:,-1])
        return out


# ============================================================================
# 4. LSTM 학습 함수
# ============================================================================
def train_lstm(zone_history):
    """
    zone_history: [0.0~1.0] list (free ratio의 과거 기록)
    """
    data = torch.tensor(zone_history, dtype=torch.float32).view(-1,1)

    # input→target 시퀀스 구성
    seq_len = 30
    xs, ys = [], []
    for i in range(len(data)-seq_len-1):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])

    xs = torch.stack(xs)      # (N, seq_len, 1)
    ys = torch.stack(ys)

    model = LSTMModel()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # 학습
    for _ in range(200):
        pred = model(xs)
        loss = loss_fn(pred, ys)
        optim.zero_grad()
        loss.backward()
        optim.step()

    return model


# ============================================================================
# 5. LSTM 예측 함수
# ============================================================================
def predict_future(model, history_seq, minutes_ahead=10):
    seq = torch.tensor(history_seq[-30:], dtype=torch.float32).view(1,30,1)
    model.eval()
    with torch.no_grad():
        pred = model(seq).item()

    # 예측값 보정 (0~1 범위)
    return float(np.clip(pred, 0, 1))


# ============================================================================
# 6. 추천 알고리즘 (현재상태 + 예측결과 결합)
# ============================================================================
def recommend_zone(current_status, future_pred):
    scores = {}

    for z in ZONES:
        free_now = current_status[z]["free"] / current_status[z]["total"]
        free_future = future_pred[z]  # 이미 ratio 형태
        
        score = 0.6 * free_now + 0.4 * free_future
        scores[z] = score

    best = max(scores, key=scores.get)

    return {
        "recommended_zone": best,
        "scores": scores,
        "future_prediction": future_pred
    }


# ============================================================================
# 7. /update : YOLO 등에서 실시간 업데이트
# ============================================================================
@app.route("/update", methods=["POST"])
def update():
    global status_data

    new_data = request.json
    for z in ZONES:
        status_data[z]["free"] = new_data[z]["free"]
        status_data[z]["occupied"] = ZONES[z] - new_data[z]["free"]

    # 기록 저장
    with open(HISTORY_FILE,"r") as f:
        hist = json.load(f)

    record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "zones": status_data
    }
    hist.append(record)

    with open(HISTORY_FILE,"w") as f:
        json.dump(hist, f, indent=2)

    return jsonify({"result":"ok"})


# ============================================================================
# 8. /status : 예측 기반 추천 포함
# ============================================================================
@app.route("/status", methods=["GET"])
def status():
    # 과거 데이터 로드
    with open(HISTORY_FILE,"r") as f:
        hist = json.load(f)

    # 데이터 부족하면 가상 데이터 생성
    if len(hist) < 50:
        hist = generate_fake_history(hours=6)
        with open(HISTORY_FILE,"w") as f:
            json.dump(hist, f, indent=2)

    # zone별 free ratio 추출
    zone_history = {z: [] for z in ZONES}
    for h in hist:
        for z in ZONES:
            ratio = h["zones"][z]["free"] / ZONES[z]
            zone_history[z].append(ratio)

    # LSTM 학습
    models = {z: train_lstm(zone_history[z]) for z in ZONES}

    # 미래 예측
    future_pred = {z: predict_future(models[z], zone_history[z]) for z in ZONES}

    # 추천결과 계산
    rec = recommend_zone(status_data, future_pred)

    return jsonify({
        "zones": status_data,
        "prediction": future_pred,
        "recommendation": rec
    })


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
