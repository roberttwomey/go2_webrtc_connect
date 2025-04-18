# Commands
## Movement Commands
- I programmed Laika to make specific movements based on the command it receives.
1. Go to the directory where `main` folder is located.
2. Enter `python main.py "COMMAND"` .
    - Replace COMMAND with the user’s desired command. (`move front`, `move back`, `move left`, `move left small`, `move left minimum`, `move right`, `move right small`, `move left minimum`)
    - Make sure to include the double quotation marks.

## Speech Recognition Commands (incomplete)
- I used Speech Recognition library to implement the speech recognition feature.
- This feature doesn’t work without internet connection, as it uses Google Cloud Speech API.
1.  Go to the directory where `main` folder is located.
2. Enter `python main.py` . Laika will automatically switch to speech recognition mode.
3. Speak the user’s desired command. (`move front`, `move back`, `move left`, `move left small`, `move left minimum`, `move right`, `move right small`, `move left minimum`)

## Object Detection & Rotation Commands
- Go2 will fail to locate or rotate to an object if it moves too fast.
1. Go to `camera_stream` directory.
2. Enter `python display_video_channel.py`  in terminal to initiate video streaming and object detection.
3. Enter one of these desired commands in terminal.
- `target <OBJECT NAME>` – replace "OBJECT NAME" with a class to center on
- `start` / `stop` – begin or pause auto‑centering
- `quit` – cleanly exit


--------------------


# Full Unitree Go2 WebRTC Driver

This repository contains a Python implementation of the WebRTC driver to connect to the Unitree Go2 Robot. WebRTC is used by the Unitree Go APP and provides high-level control through it. Therefore, no jailbreak or firmware manipulation is required. It works out of the box for Go2 AIR/PRO/EDU models.

![Description of the image](./images/screenshot_1.png)

## Supported Versions

The currently supported firmware packages are:
- 1.1.1 - 1.1.3 (latest available)
- 1.0.19 - 1.0.25

## Audio and Video Support

There are video (recvonly) and audio (sendrecv) channels in WebRTC that you can connect to. Check out the examples in the `/example` folder.

## Lidar support

There is a lidar decoder built in, so you can handle decoded PoinClouds directly. Check out the examples in the `/example` folder.

## Connection Methods

The driver supports three types of connection methods:

1. **AP Mode**: Go2 is in AP mode, and the WebRTC client is connected directly to it:

    ```python
    Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
    ```

2. **STA-L Mode**: Go2 and the WebRTC client are on the same local network. An IP or Serial number is required:

    ```python
    Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.8.181")
    ```


    If the IP is unknown, you can specify only the serial number, and the driver will try to find the IP using the special Multicast discovery feature available on Go2:

    ```python
    Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000XXXXXXXX")
    ```

3. **STA-T mode**: Remote connection through remote Unitrees TURN server. Could control your Go2 even being on the diffrent network. Requires username and pass from Unitree account

    ```python
    Go2WebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber="B42D2000XXXXXXXX", username="email@gmail.com", password="pass")
    ```

## Multicast scanner
The driver has a built-in Multicast scanner to find the Unitree Go2 on the local network and connect using only the serial number.


## Installation

```sh
cd ~
sudo apt update
sudo apt install python3-pip
sudo apt install portaudio19-dev
git clone --recurse-submodules https://github.com/legion1581/go2_webrtc_connect.git
cd go2_webrtc_connect
pip install -e .
```

## Usage 
Example programs are located in the /example directory.

### Thanks

A big thank you to TheRoboVerse community! Visit us at [TheRoboVerse](https://theroboverse.com) for more information and support.

Special thanks to the [tfoldi WebRTC project](https://github.com/tfoldi/go2-webrtc) and [abizovnuralem](https://github.com/abizovnuralem) for adding LiDAR support and [MrRobotow](https://github.com/MrRobotoW) for providing a plot LiDAR example.

 
### Support

If you like this project, please consider buying me a coffee:

<a href="https://www.buymeacoffee.com/legion1581" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>
