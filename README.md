# 🚙 Smart Parking Vision

### Real-time Camera-based Parking Space Detection & Zone Recommendation

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green)
![YOLO](https://img.shields.io/badge/YOLOv8-Detection-red)

## 📌 Summary

Smart Parking Vision은 CCTV 1대로 주차장 전체 점유 상태를 실시간 분석하고,
Zone A/B/C 중 가장 비어 있는 구역을 자동 추천하는 AI 시스템입니다.
YOLO 기반 차량 탐지 + ROI 기반 주차칸 판단을 통해 센서 없이도 정확한 주차 모니터링이 가능합니다.

## ✨ Features
### 🔎 YOLO 기반 차량 탐지 (Custom Trained)

직접 구축한 데이터셋(수동 + 자동 라벨링)으로 모델 파인튜닝

다양한 조도/혼잡도 조건에서도 안정적 인식

### 🧩 ROI 기반 주차칸 점유 판단

각 주차칸을 Polygon ROI로 설정

차량 bounding box와 교차율(IoU)로 Occupied/Free 계산

장애인 구역 별도 처리

### 🖥 Web Dashboard (Flask)

Zone A/B/C 실시간 Free/Total 표시

장애인구역 별도 표시

Zone 클릭 시 상세 주차칸 상태 + 점유 예측 그래프(Chart.js)

5초 자동 새로고침으로 실시간 데이터 반영

### 🧠 Zone 추천 알고리즘

단순 빈자리 수 비교를 넘어

향후 혼잡도 예측 기반 Score 계산

가장 추천되는 Zone 메인 화면에 표시

### 🟦 Jetson Nano 구동 가능

OpenCV + YOLO + Flask로 Edge 환경에서도 동작

추가 센서 없이 저비용·고확장성

## 📸 Demo
차량 인식 결과
![YOLO ROI Demo](result/output_occupancy.mp4)

실시간 대시보드
![Dashboard](result/스크린샷%202025-11-23%2004-32-21.png)
![Dashboard](result/스크린샷%202025-11-23%2004-32-54.png)
![Dashboard](result/스크린샷%202025-11-23%2004-33-28.png)


## 📂 Folder Structure
'''bash
EE_Project/
│── occupancy_video.py        # YOLO + ROI 실시간 점유 분석
│── web/
│   ├── Flask_app.py          # Flask Web Server
│   ├── static/               # CSS / Icons / JS
│   ├── templates/            # index.html / zone.html
│── rois/                     # ROI polygon configs
│── refs/                     # ORB reference images
│── result/                   # demo video & screenshots
│── dataset_car/              # YOLO training dataset
'''

## 🚀 Quick Start
1) Run Occupancy Detection
python3 occupancy_video.py \
  --cam /dev/video2 \
  --weights ./runs/detect/car_mix_aug_colab_ft/weights/best.pt \
  --rois ./rois/roi_A.yaml ./rois/roi_B.yaml ./rois/roi_C.yaml \
  --width 1280 --height 720

2) Launch Web Dashboard
cd web
python3 Flask_app.py


접속:

http://<Your-IP>:5000

## 🧠 Tech Stack

YOLOv8

OpenCV

NumPy

Flask

Jetson Nano

Chart.js

ORB Feature Matching (ROI Alignment)
