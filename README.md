🅿️ Smart Parking Vision

YOLO 기반 실시간 주차 공간 분석 & 추천 시스템

📌 프로젝트 개요

주차장의 차량 점유 상태를 실시간 분석하고,
Zone(A/B/C)별 빈자리 비교를 통해 가장 추천하는 주차장을 자동 선정하는 시스템입니다.

✨ 핵심 특징
1. 센서 없이, 카메라 하나로 구현하는 초저비용 시스템

초음파·RFID 센서 기반 시스템의 설치 비용 / 유지보수 / 배선 문제 해결

기존 CCTV 영상만으로 주차 공간 인식 가능 → 비용 90% 절감

2. YOLO 기반 차량 탐지 + ROI 기반 주차칸 점유 판단

사용자 정의 ROI(주차칸)와 YOLO 탐지 박스 IoU 교차 분석

각 칸별 Occupied / Free 상태 실시간 반영

장애인 구역 별도 관리 및 표시

3. 3개 Zone(A/B/C) 비교 후 ‘최적의 Zone 추천’

Zone별 Free/Total을 기반으로 점수 계산

가장 주차 가능성이 높은 Zone을 자동 추천

프론트엔드 메인 화면에 추천 결과 표시

4. 프론트엔드 – 시각화 중심의 UI

메인 UI: Zone 상태, Free / Total, 장애인칸 표시

Zone 상세 UI:

16/6/14칸 실시간 상태

장애인 구역 강조 표시

혼잡도 예측 그래프(예상 점유 / 과거 평균 비교)

5초 자동 새로고침으로 실시간 데이터 반영

5. Jetson Nano 실시간 시연 가능

occupancy_video.py로 USB 카메라 받아 YOLO 분석

Flask 웹 서버와 status.json 실시간 동기화

Jetson Nano에서도 경량 YOLO 모델로 충분히 동작

📷 시연 결과 (이미지 삽입)
1. YOLO 차량 탐지 & 주차 상태 분석
<img src="/mnt/data/out_vis.jpg" width="650">
2. 실시간 웹 대시보드
<img src="/mnt/data/스크린샷 2025-11-23 04-33-14.png" width="650">
3. Zone 상세 화면 예시
<img src="/mnt/data/스크린샷 2025-11-23 04-33-28.png" width="650">
🎬 시연 영상

시스템 전체 파이프라인이 동작하는 영상입니다:

주차칸 실시간 분석 영상

/mnt/data/output_occupancy.mp4

최종 결과 영상

/mnt/data/result.mp4

📂 프로젝트 구조
EE_Project/
│── occupancy_video.py      # YOLO + ROI 기반 차량 인식 및 상태 저장
│── web/                    # Flask 웹 서비스
│   ├── Flask_app.py
│   ├── templates/
│   └── static/
│── rois/                   # ROI 설정 파일 (A/B/C)
│── refs/                   # ORB 매칭용 기준 이미지
│── result/                 # 시연 영상 & 스크린샷
│── dataset_car/            # 차량 YOLO 학습 데이터

🧠 기술 스택

YOLOv8

OpenCV / Numpy

Flask 웹 서비스

Jetson Nano

Chart.js (UI 그래프 시각화)
