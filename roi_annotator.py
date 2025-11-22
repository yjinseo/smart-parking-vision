#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI Annotator for Parking Slots (OpenCV)
- 이미지 프레임을 띄우고 다각형 ROI(주차칸)들을 마우스로 지정하여 YAML로 저장

사용
python roi_annotator.py --image ./refs/A_ref.jpg --out ./rois/roi_A.yaml
"""

import cv2
import yaml
import argparse
from pathlib import Path
from datetime import datetime

HELP_TEXT = """
- 좌클릭: 현재 슬롯에 점 추가
- n : 현재 슬롯 종료 → 다음 슬롯 시작
- u : 현재 슬롯의 마지막 점 제거(Undo)
- d : 마지막으로 '완료된' 슬롯 삭제
- r : 현재 슬롯 ID 변경 (기본 slot_### 자동증가)
- s : YAML 저장 (기본: roi_config.yaml)
- l : YAML 불러오기 (파일 선택 다이얼로그 없음 → --out 경로 기준)
- h : 도움말 토글
- q/ESC : 종료
"""

def draw_overlay(canvas, slots, curr_points, curr_id, show_help, color_done=(0,255,0), color_curr=(0,200,255)):
    disp = canvas.copy()
    # 완료된 슬롯
    for sid, pts in slots:
        if len(pts) >= 2:
            for i in range(len(pts)):
                cv2.line(disp, pts[i], pts[(i+1)%len(pts)], color_done, 2)
        for p in pts:
            cv2.circle(disp, p, 3, color_done, -1)
        # 라벨
        if pts:
            x = int(sum([p[0] for p in pts]) / len(pts))
            y = int(sum([p[1] for p in pts]) / len(pts))
            cv2.putText(disp, sid, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_done, 2, cv2.LINE_AA)

    # 현재 슬롯
    if curr_points:
        for i in range(len(curr_points)-1):
            cv2.line(disp, curr_points[i], curr_points[i+1], color_curr, 2)
        for p in curr_points:
            cv2.circle(disp, p, 3, color_curr, -1)
        # 시작점과 현재점 안내
        cv2.putText(disp, f"editing: {curr_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_curr, 2, cv2.LINE_AA)

    if show_help:
        y0 = 60
        for i, line in enumerate(HELP_TEXT.splitlines()):
            if not line: continue
            cv2.putText(disp, line, (10, y0 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(disp, line, (10, y0 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)  # outline
            cv2.putText(disp, line, (10, y0 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1, cv2.LINE_AA)
    return disp

def save_yaml(out_path: Path, slots, img_w, img_h):
    data = {
        "meta": {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_size": [int(img_w), int(img_h)],
            "note": "slots are polygons in image pixel coordinates (x,y)"
        },
        "slots": [
            {
                "id": sid,
                "points": [[int(x), int(y)] for (x,y) in pts]
            } for sid, pts in slots
        ]
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    print(f"[SAVE] {out_path} (slots={len(slots)})")

def load_yaml(in_path: Path):
    with open(in_path, "r") as f:
        data = yaml.safe_load(f)
    slots = []
    for item in data.get("slots", []):
        sid = item["id"]
        pts = [(int(x), int(y)) for x, y in item["points"]]
        slots.append((sid, pts))
    print(f"[LOAD] {in_path} (slots={len(slots)})")
    return slots

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True, help="ROI를 지정할 기준 이미지 경로")
    ap.add_argument("--out", type=str, default="roi_config.yaml", help="YAML 저장 경로")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {img_path}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError("이미지 로드 실패")

    H, W = img.shape[:2]
    out_path = Path(args.out)

    # 상태 변수
    slots = []                 # [(id, [(x,y), ...]), ...]
    curr_points = []           # 현재 편집중 다각형
    slot_index = 1             # 기본 ID 시드
    curr_id = f"slot_{slot_index:03d}"
    show_help = True

    win = "ROI Annotator"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, param):
        nonlocal curr_points
        if event == cv2.EVENT_LBUTTONDOWN:
            curr_points.append((x, y))

    cv2.setMouseCallback(win, on_mouse)

    while True:
        disp = draw_overlay(img, slots, curr_points, curr_id, show_help)
        cv2.imshow(win, disp)
        key = cv2.waitKey(16) & 0xFF

        if key == ord('h'):
            show_help = not show_help

        elif key == ord('u'):  # undo point
            if curr_points:
                curr_points.pop()

        elif key == ord('d'):  # delete last finished slot
            if slots:
                removed = slots.pop()
                print(f"[DEL] remove slot: {removed[0]}")

        elif key == ord('r'):  # rename current slot id
            # OpenCV에서 텍스트 입력 다이얼로그가 없으므로 콘솔 입력 사용
            cv2.destroyWindow(win)
            new_id = input(f"현재 ID: {curr_id} → 새 ID 입력(엔터=변경 안 함): ").strip()
            if new_id:
                curr_id = new_id
            cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(win, on_mouse)

        elif key == ord('n'):  # finish current polygon
            if len(curr_points) >= 3:
                slots.append((curr_id, curr_points.copy()))
                print(f"[ADD] slot: {curr_id} with {len(curr_points)} points")
                slot_index += 1
                curr_id = f"slot_{slot_index:03d}"
                curr_points = []
            else:
                print("[WARN] 슬롯을 종료하려면 최소 3점이 필요합니다.")

        elif key == ord('s'):  # save yaml
            save_yaml(out_path, slots, W, H)

        elif key == ord('l'):  # load yaml
            if out_path.exists():
                slots = load_yaml(out_path)
                # 다음 ID 자동 증가 갱신
                nums = [int(s[0].split('_')[-1]) for s in slots if '_' in s[0] and s[0].split('_')[-1].isdigit()]
                slot_index = max(nums)+1 if nums else (len(slots)+1)
                curr_id = f"slot_{slot_index:03d}"
            else:
                print(f"[WARN] 불러올 파일이 없습니다: {out_path}")

        elif key in (ord('q'), 27):  # q or ESC
            break

    cv2.destroyAllWindows()
    # 종료 시 자동 저장(선택)
    if slots:
        save_yaml(out_path, slots, W, H)

if __name__ == "__main__":
    main()

