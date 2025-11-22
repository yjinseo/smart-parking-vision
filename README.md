🅿️ Smart Parking Vision

YOLO 기반 실시간 주차 공간 분석 & 추천 시스템

🚀 Overview

복수 주차장 Zone A/B/C의 주차칸 점유 상태를 실시간 분석하고,
빈자리 수를 기반으로 최적의 Zone을 추천하는 Edge AI 시스템입니다.

Jetson Nano에서도 구동 가능하며, 센서 기반 시스템 대비 압도적으로 저비용입니다.

✨ Key Features
🔍 1. 센서 없이 카메라만으로 주차칸 인식

기존 초음파·RFID 기반 주차 시스템의 문제인
설치 비용, 배선, 유지보수를 해결한 영상 기반 솔루션입니다.

🚗 2. YOLO + ROI 기반 점유 판단

YOLO 차량 탐지

주차칸 ROI 교차 분석

A/B/C 전체 칸 별 Occupied / Free 실시간 업데이트

♿ 3. 장애인구역 별도 인식

일반 칸과 별도 관리되어
장애인 주차 공간 점유도 정확하게 표시합니다.

🧠 4. 실시간 Zone 추천 알고리즘

Zone별 Free/Total 계산

Scoring 기반 추천

메인 화면에 가장 추천되는 Zone 표시

📊 5. 시각화 중심의 웹 UI

Zone 상태 한눈에 확인

각 Zone 클릭 시

16/6/14칸 실시간 상태

장애인칸 강조 표시

혼잡도 예측 그래프(Chart.js)

5초 자동 새로고침 → 실시간 반영

🟦 6. Jetson Nano 실시간 시연 지원

USB 카메라 + YOLO + Flask →
임베디드 환경에서도 매끄럽게 동작

🖼️ Demo
📌 YOLO + ROI 차량 점유 분석

![demo1](result/out_vis.jpg)

📌 실시간 웹 대시보드
![dashboard](result/dashboard_example.png)

🎬 시연 영상

📹 GitHub 용량 제한으로 영상은 링크 업로드 예정 

📂 Project Structure
EE_Project/
│── occupancy_video.py        # YOLO + ROI 기반 점유 분석
│── web/
│   ├── Flask_app.py          # Flask 웹 서버
│   ├── static/               # CSS, Icons
│   ├── templates/            # index.html, zone.html
│── rois/                     # ROI polygon 좌표 (A/B/C)
│── refs/                     # ORB alignment reference images
│── result/                   # 시연 영상 & 스크린샷
│── dataset_car/              # 차량 YOLO 학습 데이터

🧰 Tech Stack

YOLOv8

OpenCV

NumPy

Flask

Chart.js

Jetson Nano

🎯 Presentation Bullet Points

“초저비용 AI 기반 주차 모니터링”

“기존 CCTV 활용, 센서 불필요”

“실시간 주차장 추천 기능”

“장애인 주차 공간 관리 강화”

“Edge-AI Jetson Nano 실시간 처리 가능”

“자동 라벨링 포함 YOLO 학습 파이프라인 구축”

📎 How to Run
1) Occupancy Detection (YOLO + ROI)
python3 occupancy_video.py \
  --cam /dev/video2 \
  --weights ./runs/detect/car_mix_aug_colab_ft/weights/best.pt \
  --rois ./rois/roi_A.yaml ./rois/roi_B.yaml ./rois/roi_C.yaml \
  --width 1280 --height 720

2) Run Web Dashboard
cd web
python3 Flask_app.py


웹 접속:

http://<YOUR-IP>:5000
