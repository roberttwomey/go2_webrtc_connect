import os
import sys
import cv2
import numpy as np
import asyncio
import logging
import threading
import time
from queue import Queue

file_dir = os.path.dirname(os.path.abspath(__file__))
cmd_folder = os.path.normpath(
    os.path.join(file_dir, '..', '..', 'data_channel', 'main')
)
sys.path.insert(0, cmd_folder)
import commands

from ultralytics import YOLO
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
from aiortc import MediaStreamTrack

logging.basicConfig(level=logging.FATAL)

# Load YOLOv8 (will download weights if missing)
model = YOLO('yolov8n.pt')

# --- movement functions ---
async def move_forward(conn):
    print("Moving forward...")
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["Move"],
            "parameter": {"x": 1, "y": 0, "z": 0}
        }
    )

# … (other moves left, right, turn_left_min, turn_right_min, etc.) …

def main():
    frame_queue = Queue()
    target     = None
    centering  = False
    approaching = False 
    running    = True

    # throttle timers
    last_turn_time    = 0.0
    last_forward_time = 0.0 

    # 1) Set up WebRTC connection & asyncio loop in a separate thread
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

    # 2) Console‑input thread (extended for "approach")
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
                    print("[INFO] Auto‑centering ON")
                else:
                    print("[WARN] You must set a target first: target <object_name>")
            elif cmd == 'approach':          # ← new command
                approaching = not approaching
                state = "ON" if approaching else "OFF"
                print(f"[INFO] Auto‑approach {state}")
            elif cmd == 'stop':
                centering = False
                approaching = False        # ← also turn off approach
                print("[INFO] Auto‑centering & Auto‑approach OFF")
            elif cmd == 'quit':
                running = False
                print("[INFO] Quitting...")
            else:
                print(f"[WARN] Unknown command: {cmd}")

    threading.Thread(target=input_thread, daemon=True).start()

    # 3) Main display + detection loop
    win_name = 'Object Detection'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while running:
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame   = frame_queue.get()
        results = model(frame, verbose=False)[0]
        annotated = results.plot()
        img_h, img_w = frame.shape[:2]

        if centering and target:
            names      = results.names
            boxes      = results.boxes
            candidates = []

            for box, cls_id in zip(boxes.xyxy, boxes.cls):
                name = names[int(cls_id)].lower().replace(' ', '')
                if name == target:
                    x1, y1, x2, y2 = box
                    area = float((x2 - x1) * (y2 - y1))
                    cx   = float((x1 + x2) / 2)
                    candidates.append((area, cx))

            if candidates:
                # compute offset
                _, cx = max(candidates, key=lambda x: x[0])
                dx    = cx - img_w / 2
                tol   = 0.05 * img_w

                # --- Turning logic ---
                if abs(dx) > tol and (time.time() - last_turn_time) >= 1.5:
                    turn = commands.turn_right_min if dx > 0 else commands.turn_left_min
                    asyncio.run_coroutine_threadsafe(turn(conn), loop)
                    last_turn_time = time.time()

                # --- Approaching logic ---
                if approaching and abs(dx) <= tol:
                    # pick largest box area
                    area, _ = max(candidates, key=lambda x: x[0])
                    min_area = 0.40 * img_w * img_h  # object must cover 40% of frame before we stop
                    if area < min_area and (time.time() - last_forward_time) >= 1.0:
                        asyncio.run_coroutine_threadsafe(move_forward(conn), loop)
                        last_forward_time = time.time()
                    elif area >= min_area:
                        print("[INFO] Close enough to the target.")
            else:
                print(f"[INFO] No “{target}” detected this frame.")

        cv2.imshow(win_name, annotated)
        if cv2.waitKey(1) == ord('q'):
            running = False

    # 4) Cleanup
    cv2.destroyAllWindows()
    loop.call_soon_threadsafe(loop.stop)

if __name__ == '__main__':
    main()
