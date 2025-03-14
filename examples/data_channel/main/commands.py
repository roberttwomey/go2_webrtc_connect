import asyncio
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD

# Individual movements
# Execute them by using commannds like: python main.py "move right"

async def move_forward(conn):
    print("Moving forward...")
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["Move"],
            "parameter": {"x": 1, "y": 0, "z": 0}
        }
    )

async def move_back(conn):
    print("Moving forward...")
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["Move"],
            "parameter": {"x": -1, "y": 0, "z": 0}
        }
    )

async def move_right(conn):
    print("Moving to the right...")
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["Move"],
            "parameter": {"x": 0, "y": 1, "z": 0}
        }
    )

async def move_left(conn):
    print("Moving to the left...")
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["Move"],
            "parameter": {"x": 0, "y": -1, "z": 0}
        }
    )

async def turn_left(conn):
    print("Turning 45 degrees to the left..")
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["Move"],
            "parameter": {"x": 0, "y": 0, "z": 3} 
            # 3 = about 45 degrees
            # 2 = about 22 degrees
            # 1 = about 10 degrees
            # the biggest turn it can make is 45 degrees (3)
            # positive z value means turning left
            # negative z value means turning right
        }
    )

async def turn_left_small(conn):
    print("Turning 22 degrees to the left..")
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["Move"],
            "parameter": {"x": 0, "y": 0, "z": 2} 
        }
    )

async def turn_left_min(conn):
    print("Turning 10 degrees to the left..")
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["Move"],
            "parameter": {"x": 0, "y": 0, "z": 1} 
        }
    )


async def turn_right(conn):
    print("Turning 45 degrees to the right..")
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["Move"],
            "parameter": {"x": 0, "y": 0, "z": -3} 
        }
    )

async def turn_right_small(conn):
    print("Turning 22 degrees to the right..")
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["Move"],
            "parameter": {"x": 0, "y": 0, "z": -2} 
        }
    )

async def turn_right_min(conn):
    print("Turning 10 degrees to the right..")
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["Move"],
            "parameter": {"x": 0, "y": 0, "z": -1} 
        }
    )