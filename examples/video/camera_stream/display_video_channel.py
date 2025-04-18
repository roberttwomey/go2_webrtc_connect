import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
import asyncio
import logging
import threading
import time
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from aiortc import MediaStreamTrack
from ultralytics import YOLO
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
from data_channel.main.commands import (
    turn_left, turn_left_small, turn_left_min,
    turn_right, turn_right_small, turn_right_min
)

logging.basicConfig(level=logging.FATAL)

model = YOLO("yolov8n.pt")

FRAME_WIDTH = 640
FRAME_HEIGHT = 360
CENTER_X = FRAME_WIDTH // 2
TOLERANCE = 40

latest_frame = None
frame_lock = threading.Lock()
latest_frame_timestamp = 0


target_object = "person"
centering_enabled = False
object_detected_once = False
search_start_time = None

missing_frame_count = 0
MAX_MISSING_FRAMES = 30  # If no frame for ~1s, reconnect

def detection_loop():
    global latest_frame, object_detected_once, search_start_time, latest_frame_timestamp
    last_detection_time = 0
    DETECTION_INTERVAL = 0.5  # seconds

    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            frame = latest_frame.copy()
            frame_time = latest_frame_timestamp
            latest_frame = None

        now = time.time()
        print(f"[DETECT] Frame delay: {now - frame_time:.2f}s")

        if not centering_enabled:
            continue

        if now - last_detection_time < DETECTION_INTERVAL:
            continue

        last_detection_time = now
        detected = False

        try:
            infer_start = time.time()
            results = model(frame, verbose=False)
            infer_end = time.time()
            print(f"[YOLO] Inference time: {infer_end - infer_start:.2f}s")
        except Exception as e:
            print(f"[YOLO ERROR] {e}")
            continue

        if results:
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    if label == target_object:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = (x1 + x2) // 2
                        offset = center_x - CENTER_X

                        if abs(offset) > TOLERANCE:
                            if offset > 0:
                                turn_right_min()
                            else:
                                turn_left_min()
                        detected = True
                        object_detected_once = True
                        break

        if not detected and not object_detected_once:
            if search_start_time is None:
                search_start_time = time.time()
                print("[INFO] Waiting 10 seconds before rotating...")
            elif time.time() - search_start_time >= 10:
                print("[INFO] Target not detected after 10 seconds, rotating...")
                turn_left_min()


def command_loop():
    global target_object, centering_enabled, object_detected_once, search_start_time
    print("Command Interface Ready. Type 'help' for options.")
    while True:
        cmd = input("[Command]> ").strip().lower()
        if cmd.startswith("set "):
            target_object = cmd[4:].strip()
            object_detected_once = False
            search_start_time = None
            print(f"[INFO] Target object set to '{target_object}'")
        elif cmd == "start":
            centering_enabled = True
            object_detected_once = False
            search_start_time = None
            print("[INFO] Object search started.")
        elif cmd == "stop":
            centering_enabled = False
            print("[INFO] Object search stopped.")
        elif cmd == "help":
            print("\nCommands:")
            print("  set <object>  - Change target object (e.g., set bottle)")
            print("  start         - Start object detection and centering")
            print("  stop          - Stop object detection")
            print("  help          - Show this help message\n")
        else:
            print("[ERROR] Unknown command. Type 'help' for available commands.")


class VideoStreamTrack(MediaStreamTrack):
    kind = "video"

    async def recv(self):
        global latest_frame, latest_frame_timestamp, missing_frame_count
        recv_time = time.time()
        print(f"[RECV] Frame received at {recv_time:.2f}")

        try:
            frame = await super().recv()
        except Exception as e:
            print(f"[RECV ERROR] {e} — trying to request keyframe...")
            try:
                await self._track._sender._transport._connection.request_keyframe()
                print("[INFO] Keyframe requested")
            except Exception as ex:
                print(f"[ERROR] Failed to request keyframe: {ex}")
            missing_frame_count += 1
            return None

        img = frame.to_ndarray(format="bgr24")
        resized = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))

        with frame_lock:
            latest_frame = resized
            latest_frame_timestamp = recv_time

        missing_frame_count = 0
        await asyncio.sleep(0.03)
        return frame


def start_webrtc():
    global missing_frame_count
    connection = Go2WebRTCConnection(
        rtc_topic=RTC_TOPIC,
        method=WebRTCConnectionMethod.AUTO,
        video_transform=VideoStreamTrack
    )

    async def connect_loop():
        while True:
            await connection.connect()
            print("Go2 connection mode:", connection.connectionMethod.name)
            while connection.isConnected:
                await asyncio.sleep(1)
                if missing_frame_count > MAX_MISSING_FRAMES:
                    print("[WARN] No frames received. Reconnecting...")
                    await connection.reconnect()
                    missing_frame_count = 0

    asyncio.run(connect_loop())


def main():
    threading.Thread(target=detection_loop, daemon=True).start()
    threading.Thread(target=command_loop, daemon=True).start()
    start_webrtc()


if __name__ == "__main__":
    main()