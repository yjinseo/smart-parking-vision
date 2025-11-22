#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import os
import argparse
from datetime import datetime

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam', type=int, default=2, help='VideoCapture index (default: 0)')
    ap.add_argument('--out', type=str, default='./captures', help='Output directory')
    ap.add_argument('--width', type=int, default=1280, help='Capture width')
    ap.add_argument('--height', type=int, default=720, help='Capture height')
    ap.add_argument('--prefix', type=str, default='cap', help='저장 파일명 접두사')
    ap.add_argument('--interval', type=float, default=0, help='자동 캡처 간격(초). 0이면 수동만')
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera index {args.cam}')

    # MJPG 포맷 강제 (지원 시)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # 실제 적용된 해상도 확인
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 실제 해상도: {w}x{h}")
    print("[INFO] 프리뷰 창: 'q'=저장, ESC=종료")

    count = 0
    last_save = datetime.now()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 프레임 읽기 실패")
            break

        cv2.imshow(f'/dev/video{args.cam}', frame)
        key = cv2.waitKey(1) & 0xFF

        # 수동 저장
        if key == ord('q'):
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            out_path = os.path.join(args.out, f'{args.prefix}_{ts}.jpg')
            if cv2.imwrite(out_path, frame):
                count += 1
                print(f'[SAVE] {out_path} (총 {count}장)')

        # 자동 저장
        if args.interval > 0:
            now = datetime.now()
            if (now - last_save).total_seconds() >= args.interval:
                ts = now.strftime('%Y%m%d_%H%M%S_%f')
                out_path = os.path.join(args.out, f'{args.prefix}_{ts}.jpg')
                if cv2.imwrite(out_path, frame):
                    count += 1
                    print(f'[AUTO SAVE] {out_path} (총 {count}장)')
                last_save = now

        # ESC 종료
        if key == 27:
            print('[INFO] 종료합니다.')
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()



# 수동 촬영만 (q로 저장)
# python capture_usb_cam.py --cam 2 --out ./raw_images --prefix A

# 2초 간격 자동 저장 + 수동 저장 가능
# python capture_usb_cam.py --cam 2 --out ./raw_images --prefix scene1 --interval 2
