#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
영상 스트림에서 주차장 점유 여부 확인:
- YOLO로 차량 탐지
- ROI YAML 불러와 슬롯 점유 판정 (교차율 + 중심점 기반)
- 점유 슬롯만 ROI 안에 ID 표시
- FREE/OCC 상태는 ROI 바깥에 작게 표시
- 구역(A/B/C)별 색상/대시보드로 시각적 구분 강화
- MJPG, 1280x720 강제 설정 옵션 제공

사용 예:
python occupancy_video.py \
  --cam /dev/video2 \
  --weights ./runs/detect/car_mix_aug_colab_ft/weights/best.pt \
  --rois ./rois/roi_A.yaml ./rois/roi_B.yaml ./rois/roi_C.yaml \
  --thresh 0.30 \
  --width 1280 --height 720 \
  --out result.mp4
"""

import argparse, sys
from pathlib import Path
import yaml, cv2
import numpy as np
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
import json
import os

# -----------------------------
def load_rois(yaml_path: Path):
    data = yaml.safe_load(open(yaml_path, "r"))
    meta = data.get("meta", {})
    img_size = meta.get("image_size", None)  # [W,H] or None
    slots_raw = data.get("slots", [])
    slots = []
    for it in slots_raw:
        sid = it["id"]
        pts = [(int(x), int(y)) for x,y in it["points"]]
        stype = it.get("type","normal")
        tag = sid.lower()
        if stype == "normal" and "disabled" in tag:
            stype = "disabled"
        slots.append({"id":sid, "points":pts, "type":stype, "base_size":img_size})
    return slots

def scale_points(points, from_size, to_size):
    if not from_size: return points
    fw,fh = from_size; tw,th = to_size
    sx,sy = tw/max(fw,1), th/max(fh,1)
    return [(int(round(x*sx)), int(round(y*sy))) for x,y in points]

def poly_from_box(x1,y1,x2,y2):
    return Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])

# -----------------------------
# 점유 판정 (교차율 + 중심점) + 비정상적으로 큰 박스 무시
def is_occupied(roi_poly, boxes, thresh=0.3, max_ratio=1.5):
    roi_area = max(roi_poly.area, 1e-6)
    for (x1,y1,x2,y2) in boxes:
        # ROI 대비 터무니없이 큰 박스 제거(손 등 오탐 방지)
        box_area = max((x2-x1)*(y2-y1), 1)
        if box_area > roi_area * max_ratio:
            continue
        # 교차율
        box_poly = poly_from_box(x1,y1,x2,y2)
        inter = roi_poly.intersection(box_poly).area
        if inter / roi_area >= thresh:
            return True
        # 중심점
        cx, cy = (x1+x2)//2, (y1+y2)//2
        if roi_poly.contains(Point(cx,cy)):
            return True
    return False

# -----------------------------
def overlay_poly(img, pts, color_fill_bgr, alpha=0.35, edge_thick=2, edge_color=None):
    """다각형 반투명 오버레이 + 외곽선(구역색)"""
    overlay = img.copy()
    poly = np.array(pts, np.int32)
    cv2.fillPoly(overlay, [poly], color_fill_bgr)
    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    # 외곽선은 구역 색(없으면 채움색)
    # edge_c = edge_color if edge_color is not None else color_fill_bgr
    cv2.polylines(out, [poly], True, edge_thick, cv2.LINE_AA)
    return out

def put_small_text(img, text, org, color=(255,255,0)):
    """작은 텍스트(슬롯 바깥 상태표시용)"""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

def get_zone_from_id(slot_id: str):
    """ID에서 구역명 추출: 'A_001' -> 'A' (토큰 첫 글자 기준)"""
    token = slot_id.split('_')[0]
    return (token[0].upper() if token else 'Z')

# 구역별 색상(대시보드/외곽선)
ZONE_COLOR = {
    "A": (0,255,255),   # 노랑
    "B": (255,200,100), # 하늘
    "C": (255,0,255),   # 마젠타
}
DEFAULT_ZONE_COLOR = (255,255,255)

def summarize_by_zone(slots, boxes, thresh=0.3):
    """구역별 집계: {zone: {total, occ, free, special_total, special_occ}}"""
    summary = {}
    for s in slots:
        zone = get_zone_from_id(s["id"])
        roi_poly = Polygon(s["points"])
        occupied = is_occupied(roi_poly, boxes, thresh=thresh)

        if zone not in summary:
            summary[zone] = {"total":0, "occ":0,
                             "special_total":0, "special_occ":0}

        summary[zone]["total"] += 1
        if occupied:
            summary[zone]["occ"] += 1

        if s["type"] != "normal":
            summary[zone]["special_total"] += 1
            if occupied:
                summary[zone]["special_occ"] += 1

    for z in summary:
        t = summary[z]["total"]; o = summary[z]["occ"]
        summary[z]["free"] = t - o
    return summary


def draw_dashboard(vis, summary, start=(20,40)):
    """구역별 현황 텍스트 오버레이"""
    x, y = start
    for zone, data in summary.items():
        total, free = data["total"], data["free"]
        line = f"[{zone}] Free {free}/{total}"

        # C구역이면 특수칸 상태 추가
        if zone == "C" and "special_total" in data and data["special_total"] > 0:
            st, so = data["special_total"], data["special_occ"]
            line += f" (Disabled {st-so}/{st})"

        cv2.putText(vis, line, (x,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255,255,255), 2, cv2.LINE_AA)
        y += 30
    return vis

# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=str, default="0", help="카메라 인덱스 또는 /dev/video 경로")
    ap.add_argument("--weights", required=True, type=str)
    ap.add_argument("--rois", nargs="+", required=True, help="ROI yaml 파일들")
    ap.add_argument("--thresh", type=float, default=0.30)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--out", type=str, default="", help="저장 mp4 경로")
    ap.add_argument("--width", type=int, default=1280, help="캡처 폭")
    ap.add_argument("--height", type=int, default=720, help="캡처 높이")
    args = ap.parse_args()

    # ROI 로딩
    slots_all = []
    for rp in args.rois:
        base = Path(rp)
        if not base.exists():
            print(f"[WARN] ROI 없음: {base}"); continue
        slots_all.extend(load_rois(base))
    if not slots_all:
        print("[ERR] ROI 없음"); sys.exit(1)

    # YOLO 모델
    model = YOLO(args.weights)

    # 카메라 열기
    if args.cam.isdigit():
        cap = cv2.VideoCapture(int(args.cam), cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERR] 카메라 열기 실패: {args.cam}"); sys.exit(1)

    # MJPG + 해상도 강제
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # 실제 적용된 해상도
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 실제 해상도: {w}x{h}")

    # 출력 저장기
    out_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        out_writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W = frame.shape[:2]
        # ROI 스케일
        slots = [{**s, "points": scale_points(s["points"], s["base_size"], (W, H))} for s in slots_all]

        # YOLO 추론
        res = model.predict(frame, conf=args.conf, verbose=False)
        boxes = []
        if len(res) > 0 and res[0].boxes is not None:
            for b in res[0].boxes.xyxy.cpu().numpy().tolist():
                x1, y1, x2, y2 = map(int, b[:4])
                boxes.append((x1, y1, x2, y2))

        # 점유 판정 및 시각화
        summary = summarize_by_zone(slots, boxes, thresh=args.thresh)

        vis = frame.copy()
        for s in slots:
            roi_poly = Polygon(s["points"])
            occupied = is_occupied(roi_poly, boxes, thresh=args.thresh, max_ratio=1.5)

            # 채움색: 점유 빨강 / 빈 초록 / 장애인 주황
            if s["type"] != "normal":
                fill = (0,165,255) if not occupied else (0,140,255)
            else:
                fill = (0,200,0) if not occupied else (0,0,220)

            # 폴리곤 칠하기
            vis = overlay_poly(vis, s["points"], fill, alpha=0.30, edge_thick=2)

            # 점유된 슬롯은 ID 표시
            if occupied:
                cx = int(sum(x for x,_ in s["points"]) / len(s["points"]))
                cy = int(sum(y for _,y in s["points"]) / len(s["points"]))
                cv2.putText(vis, s["id"], (cx-20, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # FREE/OCC 텍스트(슬롯 바깥)
            x1 = min(p[0] for p in s["points"])
            y1 = min(p[1] for p in s["points"])
            status = "OCC" if occupied else "FREE"
            put_small_text(vis, f"{s['id']}:{status}", (x1, max(12, y1-6)), color=(255,255,0))


        # -----------------------------------------
        # ② status.json 업데이트 (웹 연동용)
        # -----------------------------------------
        status_out = {}
        for z, v in summary.items():
            status_out[z] = {
                "total": v["total"],
                "free": v["free"],
                "occ": v["occ"],
                "disabled_total": v["special_total"],
                "disabled_occ": v["special_occ"],
                "disabled_free": v["special_total"] - v["special_occ"],
                "slots": []
            }

        for s in slots:
            occupied = is_occupied(Polygon(s["points"]), boxes, thresh=args.thresh)
            zone = get_zone_from_id(s["id"])
            status_out[zone]["slots"].append({
                "id": s["id"],
                "occupied": occupied,
                "disabled": (s["type"] != "normal")
            })

        with open("web/status.json", "w") as f:
            json.dump(status_out, f, indent=2)

        # -----------------------------------------
        # ③ 상단 대시보드 렌더링
        # -----------------------------------------
        vis = draw_dashboard(vis, summary, start=(16, 18))

        # 출력/표시
        if out_writer: out_writer.write(vis)
        cv2.imshow("Occupancy", vis)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    if out_writer: out_writer.release()
    cv2.destroyAllWindows()
    print("[INFO] 종료")
    
if __name__ == "__main__":
    main()
