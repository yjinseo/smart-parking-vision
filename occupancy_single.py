#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단일 이미지에서:
- 여러 ROI YAML(A/B/C…) 불러오기
- 학습된 YOLO 모델로 차량 탐지
- ROI × 박스 교차율로 점유 판정
- 결과 콘솔 요약 + 시각화 이미지 저장

사용 예:
python occupancy_single.py \
  --image ./test_frame.jpg \
  --weights ./runs/detect/car_mix_aug_colab_ft/weights/best.pt \
  --rois ./rois/roi_A.yaml ./rois/roi_B.yaml ./rois/roi_C.yaml \
  --thresh 0.30 \
  --out ./out_vis.jpg
"""

import argparse, sys
from pathlib import Path
import yaml
import cv2
import numpy as np
from shapely.geometry import Polygon
from ultralytics import YOLO

# -----------------------------
# 유틸
# -----------------------------
def load_rois(yaml_path: Path):
    """
    YAML 구조 예시:
    meta: {image_size: [W,H], ...}
    slots: - {id: "...", points: [[x,y],...], type: "disabled"(선택)}
    """
    data = yaml.safe_load(open(yaml_path, "r"))
    meta = data.get("meta", {})
    img_size = meta.get("image_size", None)  # [W,H] or None
    slots_raw = data.get("slots", [])
    slots = []
    for it in slots_raw:
        sid = it["id"]
        pts = [(int(x), int(y)) for x, y in it["points"]]
        stype = it.get("type", None)
        # id에 disabled 등의 태그가 붙었으면 자동 추론
        tag = sid.lower()
        if stype is None:
            if "disabled" in tag or "_d" == tag[-2:] or tag.endswith("disabled"):
                stype = "disabled"
            else:
                stype = "normal"
        slots.append({"id": sid, "points": pts, "type": stype})
    return img_size, slots

def scale_points(points, from_size, to_size):
    """ROI 정의 해상도(from_size=[W,H]) → 테스트 이미지(to_size=[W,H])에 맞게 스케일"""
    if not from_size:  # 이미지 크기 정보가 없으면 그대로 반환
        return points
    fw, fh = float(from_size[0]), float(from_size[1])
    tw, th = float(to_size[0]), float(to_size[1])
    sx, sy = tw / max(fw, 1.0), th / max(fh, 1.0)
    return [(int(round(x * sx)), int(round(y * sy))) for (x, y) in points]

def poly_from_box(x1, y1, x2, y2):
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

def overlay_poly(img, poly_pts, color_bgr, alpha=0.35, edge_thick=2):
    """다각형 반투명 오버레이"""
    overlay = img.copy()
    pts = np.array(poly_pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(overlay, [pts], color_bgr)
    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.polylines(out, [pts], True, color_bgr, edge_thick, cv2.LINE_AA)
    return out

def put_label(img, text, org, color=(255,255,255), bg=(0,0,0)):
    """읽기 쉬운 텍스트 라벨"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6; thick = 2
    (w, h), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = org
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y + 4), bg, -1)
    cv2.putText(img, text, (x + 3, y - 4), font, scale, color, thick, cv2.LINE_AA)

# -----------------------------
# 메인
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=str, help="테스트 이미지 경로")
    ap.add_argument("--weights", required=True, type=str, help="학습된 가중치 .pt 경로")
    ap.add_argument("--rois", nargs="+", required=True, help="ROI yaml 여러 개 (공백 구분)")
    ap.add_argument("--thresh", type=float, default=0.30, help="ROI 교차율 임계값(기본 0.30)")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence 임계값")
    ap.add_argument("--out", type=str, default="./out_vis.jpg", help="시각화 결과 저장 경로")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERR] 이미지 없음: {img_path}"); sys.exit(1)
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERR] 이미지 로드 실패: {img_path}"); sys.exit(1)
    H, W = img.shape[:2]

    # 1) ROI 로딩(여러 파일 병합)
    all_slots = []
    for rp in args.rois:
        rp = Path(rp)
        if not rp.exists():
            print(f"[WARN] ROI 파일 없음: {rp}"); continue
        base_size, slots = load_rois(rp)
        for s in slots:
            s_scaled = s.copy()
            s_scaled["points"] = scale_points(s["points"], base_size, (W, H))
            all_slots.append(s_scaled)
    if not all_slots:
        print("[ERR] 사용할 ROI가 없습니다."); sys.exit(1)

    # 2) YOLO 추론
    model = YOLO(str(Path(args.weights)))
    res = model.predict(source=img, conf=args.conf, verbose=False)
    boxes_xyxy = []
    if len(res) > 0 and res[0].boxes is not None:
        for b in res[0].boxes.xyxy.cpu().numpy().tolist():
            x1, y1, x2, y2 = map(int, b[:4])
            boxes_xyxy.append((x1, y1, x2, y2))

    # 3) 점유 판정
    occupancy = []  # [{id, type, occupied, iou, area, ...}]
    for s in all_slots:
        rid = s["id"]; pts = s["points"]; stype = s.get("type", "normal")
        roi_poly = Polygon(pts)
        roi_area = max(roi_poly.area, 1e-6)
        best_ratio = 0.0
        for (x1, y1, x2, y2) in boxes_xyxy:
            inter = roi_poly.intersection(poly_from_box(x1, y1, x2, y2)).area
            best_ratio = max(best_ratio, inter / roi_area)
        occupancy.append({
            "id": rid, "type": stype,
            "occupied": best_ratio >= args.thresh,
            "ratio": best_ratio
        })

    # 4) 집계
    total = len(occupancy)
    occ = sum(1 for o in occupancy if o["occupied"])
    free = total - occ
    special_total = sum(1 for o in occupancy if o["type"] != "normal")
    special_occ = sum(1 for o in occupancy if o["type"] != "normal" and o["occupied"])
    special_free = special_total - special_occ

    print("=== Occupancy Summary ===")
    print(f"Total: {total} | Occupied: {occ} | Free: {free}")
    if special_total > 0:
        print(f"Special: total {special_total} | occupied {special_occ} | free {special_free}")
    print(f"(thresh={args.thresh}, conf={args.conf})")

    # 5) 시각화
    vis = img.copy()
    # ROI 색: 점유 빨강, 비어있음 초록, 특수칸은 색조 살짝 변경
    for o in occupancy:
        pts = np.array(o["id"])  # dummy to silence linters
    for s, o in zip(all_slots, occupancy):
        pts = s["points"]
        if o["type"] != "normal":
            color = (0, 165, 255) if not o["occupied"] else (0, 140, 255)  # special: orange 계열
        else:
            color = (0, 200, 0) if not o["occupied"] else (0, 0, 220)      # normal: green/red
        vis = overlay_poly(vis, pts, color, alpha=0.35, edge_thick=2)
        # 라벨 위치(중심)
        cx = int(sum([p[0] for p in pts]) / len(pts))
        cy = int(sum([p[1] for p in pts]) / len(pts))
        tag = "OCC" if o["occupied"] else "FREE"
        put_label(vis, f"{s['id']} {tag} {o['ratio']:.2f}", (max(5, cx-50), max(20, cy)))

    # 탐지 박스도 함께 표시(디버깅용)
    for (x1, y1, x2, y2) in boxes_xyxy:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)

    out_path = Path(args.out)
    cv2.imwrite(str(out_path), vis)
    print(f"[SAVE] visualization → {out_path.resolve()}")
    print("완료.")

if __name__ == "__main__":
    main()

