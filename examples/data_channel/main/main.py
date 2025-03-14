import asyncio
import argparse
import logging
import json
import sys
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
from commands import move_forward, move_back, move_right, move_left, turn_left, turn_left_small, turn_left_min, turn_right, turn_right_small, turn_right_min
from speech.recognition import listen_for_command

# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)

async def execute_command(conn, command_text):
    """
    Maps recognized speech command to the corresponding robot action.
    """
    if command_text is None:
        print("No valid command detected.")
        return

    if "move front" in command_text:
        await move_forward(conn)
    elif "move back" in command_text:
        await move_back(conn)
    elif "move right" in command_text:
        await move_right(conn)
    elif "move left" in command_text:
        await move_left(conn)
    elif "turn left" in command_text:
        await turn_left(conn)
    elif "turn left small" in command_text:
        await turn_left_small(conn)
    elif "turn left minimum" in command_text:
        await turn_left_min(conn)
    elif "turn right" in command_text:
        await turn_right(conn)
    elif "turn right small" in command_text:
        await turn_right_small(conn)
    elif "turn right minimum" in command_text:
        await turn_right_min(conn)
    else:
        print("Command not recognized:", command_text)
    
async def main(command):
    try:
        # Choose a connection method (uncomment the correct one)
        # conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.8.181")
        # conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000XXXXXXXX")
        # conn = Go2WebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber="B42D2000XXXXXXXX", username="email@gmail.com", password="pass")
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)

        # Connect to the WebRTC service.
        await conn.connect()

        ####### NORMAL MODE ########
        print("Checking current motion mode...")

        # Get the current motion_switcher status
        response = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"], 
            {"api_id": 1001}
        )

        if response['data']['header']['status']['code'] == 0:
            data = json.loads(response['data']['data'])
            current_motion_switcher_mode = data['name']
            print(f"Current motion mode: {current_motion_switcher_mode}")

        # Switch to "normal" mode if not already
        if current_motion_switcher_mode != "normal":
            print(f"Switching motion mode from {current_motion_switcher_mode} to 'normal'...")
            await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"], 
                {
                    "api_id": 1002,
                    "parameter": {"name": "normal"}
                }
            )
            await asyncio.sleep(5)  # Wait while it stands up

        if command is not None:
            print(f"Using command: {command}")
            command_text = command.lower()
        else:
            print("No command provided; switching to speech mode.")
            command_text = listen_for_command()
        
        await execute_command(conn, command_text)
        await asyncio.sleep(3) 
    

        # Perform a "Hello" movement
        # print("Performing 'Hello' movement...")
        # await conn.datachannel.pub_sub.publish_request_new(
            # RTC_TOPIC["SPORT_MOD"], 
            # {"api_id": SPORT_CMD["Hello"]}
        # )
        # await asyncio.sleep(3)

        # Perform a "Move Backward" movement
        # print("Moving backward...")
        # await conn.datachannel.pub_sub.publish_request_new(
            # RTC_TOPIC["SPORT_MOD"], 
            # {
                # "api_id": SPORT_CMD["Move"],
                # "parameter": {"x": -0.5, "y": 0, "z": 0}
            # }
        # )

        # await asyncio.sleep(3)

        ####### AI MODE ########

        # # Switch to AI mode
        # print("Switching motion mode to 'AI'...")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["MOTION_SWITCHER"], 
        #     {
        #         "api_id": 1002,
        #         "parameter": {"name": "ai"}
        #     }
        # )
        # await asyncio.sleep(10)

        # # Switch to Handstand Mode
        # print("Switching to Handstand Mode...")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"], 
        #     {
        #         "api_id": SPORT_CMD["StandOut"],
        #         "parameter": {"data": True}
        #     }
        # )

        # await asyncio.sleep(5)

        # # Switch back to StandUp Mode
        # print("Switching back to StandUp Mode...")
        # await conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"], 
        #     {
        #         "api_id": SPORT_CMD["StandOut"],
        #         "parameter": {"data": False}
        #     }
        # )

        # # await asyncio.sleep(5)
        # # Perform a backflip
        # # print(f"Performing BackFlip")
        # # await conn.datachannel.pub_sub.publish_request_new(
        # #     RTC_TOPIC["SPORT_MOD"], 
        # #     {
        # #         "api_id": SPORT_CMD["BackFlip"],
        # #         "parameter": {"data": True}
        # #     }
        # # )

        # # Keep the program running for a while
        # await asyncio.sleep(3600)
    
    except ValueError as e:
        # Log any value errors that occur during the process.
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a movement command to the WebRTC device.")
    parser.add_argument("command", nargs="?", default=None, help="Movement command (e.g., move_forward, move_right, turn45)")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.command))
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)