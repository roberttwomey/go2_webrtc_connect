import os
import sys
import cv2
import asyncio
import logging
import time

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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'YOLO-3D')))
from detection_model import ObjectDetector
from depth_model import DepthEstimator

detector = ObjectDetector(model_path='path/to/yolo3d_model.pth', conf_thresh=0.5)
depth_estimator = DepthEstimator(model_name='depth-anything')

# silence h264 decode errors from aiortc
logging.getLogger('aiortc.codecs.h264').setLevel(logging.ERROR)
# logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# device & model
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")
model = YOLO('yolov8n.pt').to(device)
if device.startswith('cuda'):
    model = model.half()

# control params
TURN_DELAY = 1.0           # seconds between turn/move commands
EMA_ALPHA = 0.3            # smoothing factor for dx
AREA_ALPHA = 0.2           # smoothing for area
CONF_THRESH = 0.3          # detection confidence threshold
DETECT_AREA_RATIO = 0.02   # ignore boxes smaller than 2% of frame area

MIN_AREA_RATIO = 0.2       # stop approaching when object covers 20% of frame
SPIN_THRESHOLD = 5         # consecutive misses before spin
RESIZE_W, RESIZE_H = 640, 360
APPROACH_SPEED = 0.3       # forward speed (0-1)

async def recv_camera_stream(track: MediaStreamTrack, queue: asyncio.Queue):
    while True:
        frame = await track.recv()
        img = frame.to_ndarray(format='bgr24')
        if not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await queue.put(img)

async def input_loop(state):
    loop = asyncio.get_event_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            continue
        cmd = line.strip().lower()
        if cmd.startswith('target '):
            raw = cmd.split(' ',1)[1]
            state['target'] = raw.replace(' ','').lower()
            logger.info(f"Target set to '{raw}'")
        elif cmd == 'start':
            if state['target']:
                state.update({'centering': True, 'state': 'CENTER', 'missed_count': 0})
                logger.info("Auto-centering ON")
            else:
                logger.warning("Set a target first: target <object_name>")
        elif cmd == 'approach':
            state['approaching'] = not state['approaching']
            state['state'] = 'APPROACH' if state['approaching'] else 'CENTER'
            logger.info(f"Auto-approach {'ON' if state['approaching'] else 'OFF'}")
        elif cmd == 'stop':
            state.update({'centering': False, 'approaching': False, 'state': 'IDLE'})
            logger.info("Auto-centering & approach OFF")
        elif cmd == 'quit':
            state['running'] = False
            logger.info("Quitting...")
            return
        else:
            logger.warning(f"Unknown command: {cmd}")

async def detection_loop(conn, queue, state):
    last_turn = last_forward = last_spin = 0.0
    ema_dx = ema_area = 0.0
    win_name = 'Object Detection'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while state['running']:
        try:
            frame = await asyncio.wait_for(queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (RESIZE_W, RESIZE_H))
        results = model(small, verbose=False)[0]

        yolo3d_detections = detector.detect(frame)
        depth_map = depth_estimator.estimate(frame)

        for detection in yolo3d_detections:
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            distance = depth_map[center_y, center_x]
            logger.info(f"YOLO-3D detected object at ({center_x},{center_y}), distance: {distance:.2f} m")

        annotated = frame.copy()
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1,y1,x2,y2 = box
            x1,y1,x2,y2 = map(int, [x1*w/RESIZE_W, y1*h/RESIZE_H, x2*w/RESIZE_W, y2*h/RESIZE_H])
            cv2.rectangle(annotated, (x1,y1),(x2,y2),(0,255,0),2)
            name = results.names[int(cls)]
            cv2.putText(annotated, name, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        if state['state'] in ('CENTER','APPROACH') and state['target']:
            candidates = []
            for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                if conf < CONF_THRESH:
                    continue
                x1,y1,x2,y2 = box
                area = (x2-x1)*(y2-y1)
                # ignore tiny detections
                if area < DETECT_AREA_RATIO * w * h:
                    continue
                name = results.names[int(cls_id)].lower().replace(' ','')
                if name == state['target']:
                    cx = (x1+x2)/2
                    candidates.append((area, cx))

            if candidates:
                # reset miss counter
                state['missed_count'] = 0
                area, cx = max(candidates, key=lambda x: x[0])
                dx = cx - RESIZE_W/2
                ema_dx = EMA_ALPHA * dx + (1-EMA_ALPHA) * ema_dx
                norm_area = area / (w*h)
                ema_area = AREA_ALPHA * norm_area + (1-AREA_ALPHA) * ema_area

                # turning to center
                if abs(ema_dx) > 0.1*RESIZE_W and time.time() - last_turn >= TURN_DELAY:
                    if ema_dx > 0:
                        await commands.turn_right_min(conn)
                    else:
                        await commands.turn_left_min(conn)
                    last_turn = time.time()

                # controlled approach
                if state['state'] == 'APPROACH' and abs(ema_dx) <= 0.1*RESIZE_W:
                    if ema_area < MIN_AREA_RATIO:
                        if time.time() - last_forward >= TURN_DELAY:
                            await conn.datachannel.pub_sub.publish_request_new(
                                RTC_TOPIC['SPORT_MOD'],
                                {'api_id': SPORT_CMD['Move'], 'parameter':{'x':APPROACH_SPEED,'y':0,'z':0}}
                            )
                            last_forward = time.time()
                    else:
                        logger.info("Reached stopping distance.")
            else:
                # increment miss counter and spin when needed
                state['missed_count'] += 1
                if state['missed_count'] >= SPIN_THRESHOLD and time.time() - last_spin >= TURN_DELAY:
                    await commands.turn_right_min(conn)
                    last_spin = time.time()
                    state['missed_count'] = 0
                logger.info(f"No '{state['target']}' detected ({state['missed_count']}/{SPIN_THRESHOLD})")

        cv2.imshow(win_name, annotated)
        if cv2.waitKey(1) == ord('q'):
            state['running'] = False
            break

    cv2.destroyAllWindows()

async def main():
    state = {'target':None, 'centering':False, 'approaching':False, 'state':'IDLE', 'running':True, 'missed_count':0}

    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
    await conn.connect()
    conn.video.switchVideoChannel(True)

    frame_queue = asyncio.Queue(maxsize=1)
    conn.video.add_track_callback(lambda track: asyncio.create_task(recv_camera_stream(track, frame_queue)))

    await asyncio.gather(input_loop(state), detection_loop(conn, frame_queue, state))

if __name__ == '__main__':
    asyncio.run(main())
