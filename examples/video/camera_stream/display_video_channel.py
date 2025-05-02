import os
import sys
import cv2
import numpy as np
import asyncio
import logging
import threading
import time
from queue import Queue, Empty

import torch
from ultralytics import YOLO

# adjust these imports to point at your local repo structure
file_dir = os.path.dirname(os.path.abspath(__file__))
cmd_folder = os.path.normpath(
    os.path.join(file_dir, '..', '..', 'data_channel', 'main')
)
sys.path.insert(0, cmd_folder)
import commands

from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
from aiortc import MediaStreamTrack

logging.basicConfig(level=logging.FATAL)

# — Pick device dynamically —
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

# — Load model on appropriate device, half-precision only on GPU —
model = YOLO('yolov8n.pt').to(device)
if device.startswith('cuda'):
    model = model.half()

# configuration constants
TURN_DELAY = 1                 # seconds between any turn commands
SPIN_THRESHOLD_FRAMES = 3      # require this many consecutive misses before spinning
CENTER_TOL = 0.1               # tolerance as fraction of small_w
HYST_FRACTION = 0.02           # hysteresis band as fraction of small_w
MIN_CONFIDENCE = 0.5           # minimum YOLO confidence
MIN_AREA_FRAC = 0.02           # minimum box area fraction (small frame) for detection
APPROACH_AREA_FRAC = 0.40      # box area fraction (small frame) to consider “close enough”

async def move_forward(conn):
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["Move"],
            "parameter": {"x": 1, "y": 0, "z": 0}
        }
    )

def main():
    frame_queue  = Queue()
    target       = None
    centering    = False
    approaching  = False
    running      = True

    # state for throttles & debouncing
    last_turn_time           = 0.0
    last_forward_time        = 0.0
    last_no_detect_spin_time = 0.0
    no_detect_frames         = 0

    # 1) WebRTC + asyncio setup
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
    loop = asyncio.new_event_loop()

    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            frame_queue.put(img)

    def rtc_thread(loop):
        asyncio.set_event_loop(loop)
        async def setup():
            await conn.connect()
            conn.video.switchVideoChannel(True)
            conn.video.add_track_callback(recv_camera_stream)
        loop.run_until_complete(setup())
        loop.run_forever()

    threading.Thread(target=rtc_thread, args=(loop,), daemon=True).start()

    # 2) Input thread for commands
    def input_thread():
        nonlocal target, centering, approaching, running
        while running:
            cmd = input('> ').strip().lower()
            if cmd.startswith('target '):
                raw = cmd.split(' ', 1)[1]
                target = raw.replace(' ', '').lower()
                print(f"[INFO] Target set to “{raw}”")
            elif cmd == 'start':
                if target:
                    centering = True
                    print("[INFO] Auto-centering ON")
                else:
                    print("[WARN] You must set a target first: target <object_name>")
            elif cmd == 'approach':
                approaching = not approaching
                print(f"[INFO] Auto-approach {'ON' if approaching else 'OFF'}")
            elif cmd == 'stop':
                centering = False
                approaching = False
                print("[INFO] Auto-centering & Auto-approach OFF")
            elif cmd == 'quit':
                running = False
                print("[INFO] Quitting...")
            else:
                print(f"[WARN] Unknown command: {cmd}")

    threading.Thread(target=input_thread, daemon=True).start()

    win_name = 'Object Detection'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # detection loop
    while running:
        # — Always work on the freshest frame —
        try:
            frame = frame_queue.get(timeout=0.1)
        except Empty:
            continue
        while True:
            try:
                frame = frame_queue.get_nowait()
            except Empty:
                break

        img_h, img_w = frame.shape[:2]

        # — Speed-up: resize for faster inference —
        small_w, small_h = 320, 180
        small = cv2.resize(frame, (small_w, small_h))

        # run detection on the small frame
        results = model(small, verbose=False)[0]

        # annotate on full-size frame
        annotated = frame.copy()
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = box
            # scale coords back up
            x1 = int(x1 * img_w / small_w)
            y1 = int(y1 * img_h / small_h)
            x2 = int(x2 * img_w / small_w)
            y2 = int(y2 * img_h / small_h)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
            name = results.names[int(cls)]
            cv2.putText(
                annotated, name, (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1
            )

        # centering, approach & spin logic
        if centering and target:
            names      = results.names
            boxes      = results.boxes
            candidates = []

            # build filtered candidate list in small-frame coords
            area_threshold      = MIN_AREA_FRAC * (small_w * small_h)
            approach_area_thresh= APPROACH_AREA_FRAC * (small_w * small_h)
            for box, cls_id, conf in zip(boxes.xyxy, boxes.cls, results.boxes.conf):
                if float(conf) < MIN_CONFIDENCE:
                    continue
                x1, y1, x2, y2 = box
                area = float((x2 - x1) * (y2 - y1))
                if area < area_threshold:
                    continue
                name = names[int(cls_id)].lower().replace(' ', '')
                if name == target:
                    cx = float((x1 + x2) / 2)
                    candidates.append((area, cx))

            if candidates:
                # reset miss counter
                no_detect_frames = 0

                # pick largest
                _, cx = max(candidates, key=lambda x: x[0])
                dx    = cx - (small_w / 2)

                tol       = CENTER_TOL * small_w
                hysteresis= HYST_FRACTION * small_w

                # turning with hysteresis + delay
                now = time.time()
                if now - last_turn_time >= TURN_DELAY:
                    if dx > tol + hysteresis:
                        asyncio.run_coroutine_threadsafe(
                            commands.turn_right_min(conn), loop
                        )
                        last_turn_time = now
                    elif dx < -tol - hysteresis:
                        asyncio.run_coroutine_threadsafe(
                            commands.turn_left_min(conn), loop
                        )
                        last_turn_time = now

                # approaching when centered
                if approaching and abs(dx) <= tol:
                    # reuse area from before
                    area, _ = max(candidates, key=lambda x: x[0])
                    if area < approach_area_thresh and (now - last_forward_time) >= 1.0:
                        asyncio.run_coroutine_threadsafe(move_forward(conn), loop)
                        last_forward_time = now
                    elif area >= approach_area_thresh:
                        print("[INFO] Close enough to the target.")

            else:
                # count consecutive misses
                no_detect_frames += 1
                now = time.time()
                if (no_detect_frames >= SPIN_THRESHOLD_FRAMES and
                    now - last_no_detect_spin_time >= TURN_DELAY):
                    asyncio.run_coroutine_threadsafe(
                        commands.turn_left_min(conn), loop
                    )
                    last_no_detect_spin_time = now
                print(f"[INFO] No “{target}” detected — spinning to search...")

        cv2.imshow(win_name, annotated)
        if cv2.waitKey(1) == ord('q'):
            running = False

    # cleanup
    cv2.destroyAllWindows()
    loop.call_soon_threadsafe(loop.stop)

if __name__ == '__main__':
    main()
