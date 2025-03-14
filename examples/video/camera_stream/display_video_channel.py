import cv2
import numpy as np
import sys
import os

# Create an OpenCV window and display a blank image
height, width = 720, 1280  # Adjust the size as needed
img = np.zeros((height, width, 3), dtype=np.uint8)
cv2.imshow('Video', img)
cv2.waitKey(1)  # Ensure the window is created

# Get the absolute path of the 'examples' directory
script_dir = os.path.dirname(os.path.abspath(__file__))
examples_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))

# Add 'examples' to Python's search path
sys.path.append(examples_dir)

import asyncio
import logging
import threading
import time
from queue import Queue
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from aiortc import MediaStreamTrack
from ultralytics import YOLO
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
from data_channel.main.commands import turn_left, turn_left_small, turn_left_min, turn_right, turn_right_small, turn_right_min


# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)

# Initialize YOLO
model = YOLO("yolov8n.pt")

# Frame queue for WebRTC video
frame_queue = Queue()

# Define image dimensions
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2
TOLERANCE = 50  # Acceptable pixel range for centering

target_object = "none" # Default = no tracking
centering_enabled = False # enable/disable centering

# Processing video frames
async def recv_camera_stream(track):
    while True:
        frame = await track.recv()
        img = frame.to_ndarray(format="bgr24")
        frame_queue.put(img)

# Rotate Laika based on target object's position
async def adjust_rotation(conn, obj_x):
    """Rotates the robot based on target object's position."""
    global target_object, centering_enabled

    if not centering_enabled:
        return  # Do nothing if centering is disabled
    
    x_offset = obj_x - CENTER_X

    if abs(x_offset) > TOLERANCE:
        if x_offset < 0:
            print(f"{target_object} is to the left → Rotating left")
            await turn_left_min(conn)
        else:
            print(f"{target_object} is to the right → Rotating right")
            await turn_right_min(conn)
    else:
        print(f"{target_object} is centered! No rotation needed.")

def handle_user_commands():
    """Listens for user commands to change the target object or enable/disable centering."""
    global target_object, centering_enabled

    while True:
        command = input("\nEnter command (target object / 'center' / 'stop' / 'q' to quit): ").strip().lower()

        if command == "q":
            print("🚀 Exiting program...")
            os._exit(0)

        elif command == "center":
            centering_enabled = True
            print("✅ Centering enabled!")

        elif command == "stop":
            centering_enabled = False
            print("⏹️ Centering disabled!")

        else:
            target_object = command  # Update target object
            print(f"🎯 Target object updated to: {target_object}")


async def main():

    # Choose a connection method (uncomment the correct one)
    # conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.8.181")
    # conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000XXXXXXXX")
    # conn = Go2WebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber="B42D2000XXXXXXXX", username="email@gmail.com", password="pass")
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)

    await conn.connect()

    # Start video stream
    conn.video.switchVideoChannel(True)
    conn.video.add_track_callback(recv_camera_stream)

    # Start user input thread
    threading.Thread(target=handle_user_commands, daemon=True).start()

    while True:
        if not frame_queue.empty():
            img = frame_queue.get()

            if img is None:
                continue

            # Run YOLO object detection without verbose logging
            results = model(img, verbose=False)

            detected_target = False

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    obj_x = (x1 + x2) // 2
                    confidence = float(box.conf[0])
                    obj_class = int(box.cls[0])
                    label = result.names[obj_class]

                    # Draw bounding box and label
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Check if detected object matches the target
                    if label.lower() == target_object.lower():
                        detected_target = True
                        await adjust_rotation(conn, obj_x)

            # Display the frame
            cv2.imshow("YOLO Object Tracking", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        await asyncio.sleep(0.1)  # Prevent high CPU usage


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
        sys.exit(0)
