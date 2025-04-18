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
from aiortc import MediaStreamTrack

logging.basicConfig(level=logging.FATAL)

# Load YOLOv8 (will download weights if missing)
model = YOLO('yolov8n.pt')

def main():
    frame_queue = Queue()
    target = None
    centering = False
    running = True

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

    # 2) Spawn a console‑input thread to handle user commands
    def input_thread():
        nonlocal target, centering, running
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
            elif cmd == 'stop':
                centering = False
                print("[INFO] Auto‑centering OFF")
            elif cmd == 'quit':
                running = False
                print("[INFO] Quitting...")
            else:
                print(f"[WARN] Unknown command: {cmd}")

    threading.Thread(target=input_thread, daemon=True).start()

    # 3) Main display + centering loop (with 2 s delay between turns)
    win_name = 'Object Detection'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # keep track of when we last sent a turn
    last_turn_time = 0.0

    while running:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # run detection and annotate
            results = model(frame, verbose=False)[0]
            annotated = results.plot()

            if centering and target:
                names = results.names
                boxes = results.boxes
                candidates = []

                for box, cls_id in zip(boxes.xyxy, boxes.cls):
                    name = names[int(cls_id)].lower().replace(' ', '')
                    if name == target:
                        x1, y1, x2, y2 = box
                        area = float((x2 - x1) * (y2 - y1))
                        cx = float((x1 + x2) / 2)
                        candidates.append((area, cx))

                if candidates:
                    # select the largest detection
                    _, cx = max(candidates, key=lambda x: x[0])
                    img_h, img_w = frame.shape[:2]
                    dx = cx - img_w / 2

                    tol = 0.05 * img_w  # dead‑zone tolerance
                    if abs(dx) > tol:
                        # choose turn command
                        if dx > 0:
                            turn = commands.turn_right_min
                        else:
                            turn = commands.turn_left_min

                        # throttle to one turn every 1 second
                        now = time.time()
                        if now - last_turn_time >= 1.0:
                            asyncio.run_coroutine_threadsafe(turn(conn), loop)
                            last_turn_time = now
                else:
                    print(f"[INFO] No “{target}” detected this frame.")

            cv2.imshow(win_name, annotated)
            if cv2.waitKey(1) == ord('q'):
                running = False
        else:
            time.sleep(0.01)

    # 4) Cleanup
    cv2.destroyAllWindows()
    loop.call_soon_threadsafe(loop.stop)

if __name__ == '__main__':
    main()
