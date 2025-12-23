<div align="center">

# SDV - Software Defined Vehicle

**NVIDIA Jetson Orin Nano 기반 자율주행 로봇 플랫폼**

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue?logo=ros)](https://docs.ros.org/en/humble/)
[![Jetson](https://img.shields.io/badge/Jetson-Orin_Nano-76B900?logo=nvidia)](https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.3-76B900?logo=nvidia)](https://developer.nvidia.com/tensorrt)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Architecture](#architecture)

</div>

---

## Overview

SDV(Software Defined Vehicle)는 NVIDIA Jetson Orin Nano Super를 기반으로 한 자율주행 로봇 플랫폼입니다. 웹 브라우저를 통해 로봇을 제어하고, SLAM/Navigation, AI 객체 인식 등의 기능을 제공합니다.

### Key Features

| Feature | Description |
|---------|-------------|
| **Web Control** | 모바일/PC 브라우저에서 실시간 제어 |
| **SLAM** | Cartographer 기반 실시간 맵 생성 |
| **Navigation** | Nav2 기반 자율 주행 |
| **Auto Explore** | LiDAR 기반 자동 맵 탐색 (로봇 청소기 모드) |
| **AI Detection** | YOLOv8 TensorRT 객체 인식 (147 FPS) |
| **Gesture Control** | MediaPipe 손 제스처로 로봇 제어 |
| **Camera** | 16:9 실시간 스트리밍 (MJPEG) + 자동 재연결 |

---

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| **Board** | NVIDIA Jetson Orin Nano Super (8GB) |
| **Camera** | IMX219 CSI Camera (160° FOV) |
| **LiDAR** | LD19 360° LiDAR (0.1-12m range) |
| **Motors** | ROBOTIS Dynamixel XL430-W250-T × 2 |
| **Controller** | OpenCR 1.0 (ARM Cortex-M7) |
| **Power** | DC 9V (2S LiPo / DC Adapter) |
| **Wheels** | 66mm diameter, 160mm separation |

---

## Installation

### Prerequisites

```bash
# ROS2 Humble 설치 확인
ros2 --version

# JetPack 6.x with TensorRT 10.x
dpkg -l | grep tensorrt
```

### Clone & Build

```bash
# 1. Clone repository
cd ~
git clone https://github.com/hwkim3330/jetson_robot.git ros2_ws

# 2. Install dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# 3. Build
source /opt/ros/humble/setup.bash
colcon build --symlink-install

# 4. Setup environment
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
echo "export KETI_MODEL=robot" >> ~/.bashrc
echo "export LDS_MODEL=LDS-04" >> ~/.bashrc
source ~/.bashrc
```

### Web UI Setup

```bash
# Install nginx
sudo apt install nginx -y

# Copy web files
sudo cp -r ~/ros2_ws/src/robot_web/www/* /var/www/robot/

# Configure nginx (port 8888)
sudo tee /etc/nginx/sites-available/robot << 'EOF'
server {
    listen 8888;
    root /var/www/robot;
    index index.html;
    location / {
        try_files $uri $uri/ =404;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/robot /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx
```

### Auto-Start Service

```bash
# Create systemd service
sudo tee /etc/systemd/system/robot.service << 'EOF'
[Unit]
Description=SDV Robot Service
After=network.target

[Service]
Type=simple
User=nvidia
WorkingDirectory=/home/nvidia/ros2_ws
ExecStart=/home/nvidia/ros2_ws/scripts/start_robot.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable robot.service nginx
sudo systemctl start robot.service
```

---

## Quick Start

### 1. Access Web UI

```
http://<ROBOT_IP>:8888
```

### 2. Control Tabs

| Tab | Function |
|-----|----------|
| **Control** | Camera view + Joystick control |
| **Map** | SLAM / Navigation / Goal setting |
| **Clean** | Auto SLAM exploration |
| **AI** | YOLO object detection |

### 3. Keyboard Control

| Key | Action |
|-----|--------|
| `W` | Forward |
| `S` | Backward |
| `A` | Turn Left |
| `D` | Turn Right |
| `Space` | Stop |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Web Browser (port 8888)                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │   [Control]  [Map]  [Clean]  [AI]                         │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │                                                           │  │
│  │          Camera / Map Canvas                              │  │
│  │                                                           │  │
│  ├─────────────────┬─────────────────────────────────────────┤  │
│  │     LiDAR       │              Joystick                   │  │
│  └─────────────────┴─────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │ WebSocket (9090)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ROS2 Humble                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  rosbridge   │  │ camera_node  │  │mode_controller│          │
│  │    (9090)    │  │   (8080)     │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ robot_driver │  │  LD19 LiDAR  │  │ yolo_detector│          │
│  │ (Dynamixel)  │  │   (/scan)    │  │  (TensorRT)  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ cartographer │  │    Nav2      │  │ auto_explore │          │
│  │   (SLAM)     │  │ (Navigation) │  │   (Clean)    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Hardware                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Jetson Orin  │  │  IMX219 CSI  │  │  LD19 LiDAR  │          │
│  │  Nano Super  │  │   Camera     │  │    360°      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │   OpenCR     │  │  Dynamixel   │                             │
│  │ Controller   │  │  XL430 ×2    │                             │
│  └──────────────┘  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## ROS Topics

### Core Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/cmd_vel` | Twist | Velocity command |
| `/odom` | Odometry | Robot odometry |
| `/scan` | LaserScan | LiDAR scan data |
| `/map` | OccupancyGrid | SLAM map |
| `/camera/image_raw/compressed` | CompressedImage | Camera stream |

### Control Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/robot/mode` | String | Mode selection (manual/slam/nav/explore) |
| `/robot/ai_mode` | String | AI mode (yolo/off) |
| `/robot/save_map` | String | Save map command |
| `/goal_pose` | PoseStamped | Navigation goal |
| `/initialpose` | PoseWithCovarianceStamped | Initial pose |

### AI Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/detections` | String | YOLO detections (JSON) |
| `/person_detected` | Bool | Person detection flag |
| `/gesture` | String | Hand gesture (stop/go/left/right/thumbs_up) |
| `/hand_landmarks` | Float32MultiArray | MediaPipe 21 hand landmarks |
| `/gesture_enable` | Bool | Enable gesture control |

---

## Ports

| Port | Service | Description |
|------|---------|-------------|
| **8888** | nginx | Web UI |
| **8080** | camera_node | MJPEG stream |
| **9090** | rosbridge | WebSocket |

---

## Performance

### YOLOv8n TensorRT (FP16)

| Metric | Value |
|--------|-------|
| Engine Size | 9 MB |
| Throughput | **147 FPS** |
| Latency (mean) | 7.6 ms |
| GPU Compute | 6.8 ms |

### Camera

| Metric | Value |
|--------|-------|
| Resolution | 640 x 360 (16:9) |
| FPS | 15 |
| Encoding | Hardware JPEG (nvjpegenc) |

---

## Gesture Control

MediaPipe를 사용한 손 제스처 인식으로 로봇을 제어할 수 있습니다.

### Supported Gestures

| Gesture | Action | Description |
|---------|--------|-------------|
| ✋ **STOP** | 정지 | 손바닥 펴기 (5손가락) |
| ✊ **GO** | 전진 | 주먹 쥐기 |
| 👈 **LEFT** | 좌회전 | 검지로 왼쪽 가리키기 |
| 👉 **RIGHT** | 우회전 | 검지로 오른쪽 가리키기 |
| ☝️ **FORWARD** | 전진 | 검지로 위 가리키기 |
| 👇 **BACKWARD** | 후진 | 검지로 아래 가리키기 |
| 👍 **THUMBS UP** | Follow | 엄지 위로 |
| 👎 **THUMBS DOWN** | - | 엄지 아래로 |
| ✌️ **PEACE** | 빠른전진 | V 사인 |
| 🤟 **ROCK** | - | 검지+새끼손가락 |

### Features

- **방향 무관**: 손이 거꾸로, 옆으로 있어도 인식
- **거리 기반 감지**: 카메라 거리에 무관하게 동작
- **웹 오버레이**: 손 랜드마크가 카메라 영상에 실시간 표시

### Usage

1. 웹 UI의 Control 탭에서 🖐️ 버튼 클릭
2. 카메라에 손을 보여주면 랜드마크 표시
3. 제스처에 따라 로봇 제어

---

## Package Structure

```
ros2_ws/
├── src/
│   ├── robot_bringup/        # Launch files
│   ├── robot_driver/         # OpenCR/Dynamixel driver
│   ├── robot_description/    # URDF model
│   ├── robot_web/            # Web UI
│   │   ├── www/              # HTML/CSS/JS
│   │   └── scripts/          # camera_node.py
│   ├── robot_ai/             # AI features
│   │   ├── scripts/          # yolo_detector, mode_controller
│   │   └── models/           # TensorRT engines
│   ├── robot_slam/           # Cartographer SLAM
│   ├── robot_navigation/     # Nav2 navigation
│   └── ldlidar_stl_ros2/     # LD19 driver
├── maps/                     # Saved maps
├── scripts/                  # Start/stop scripts
└── docs/                     # Documentation
```

---

## Troubleshooting

### Check Services

```bash
# Service status
systemctl status robot.service nginx

# View logs
journalctl -u robot.service -f

# ROS2 nodes
ros2 node list

# Topic frequency
ros2 topic hz /scan
ros2 topic hz /camera/image_raw/compressed
```

### Common Issues

| Issue | Solution |
|-------|----------|
| Web UI not loading | Check nginx: `sudo systemctl restart nginx` |
| Camera not working | Check CSI connection, restart service |
| LiDAR no data | Check USB connection, `/dev/ttyUSB0` |
| SLAM map not showing | Press SLAM button, drive around |
| Navigation fails | Save map first, then start Nav |

### Restart Services

```bash
# Restart robot service
sudo systemctl restart robot.service

# Restart all
sudo systemctl restart robot.service nginx
```

---

## Documentation

Full documentation available at: [GitHub Pages](https://hwkim3330.github.io/jetson_robot/)

- [Installation Guide](docs/installation.md)
- [Hardware Setup](docs/hardware.md)
- [Web UI Guide](docs/web-ui.md)
- [SLAM & Navigation](docs/slam-navigation.md)
- [AI Features](docs/ai-features.md)
- [API Reference](docs/api-reference.md)

---

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [ROS2](https://docs.ros.org/en/humble/) - Robot Operating System
- [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing) - Edge AI Platform
- [Cartographer](https://google-cartographer-ros.readthedocs.io/) - SLAM
- [Nav2](https://navigation.ros.org/) - Navigation Stack
- [YOLOv8](https://docs.ultralytics.com/) - Object Detection
- [KETI](https://www.keti.re.kr/) - Korea Electronics Technology Institute

---

<div align="center">

**Built with NVIDIA Jetson Orin Nano Super**

</div>
