# SDV - Self-Driving Vehicle Platform

NVIDIA Jetson Orin Nano Super 기반 자율주행 로봇

## 시스템 구성

```
┌─────────────────────────────────────────────────────────────┐
│              Web Interface (port 8888)                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  [Logo] SDV    [Control][Map][AI]        ● Online   │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │                                                     │    │
│  │        Camera View (16:9) or Map View               │    │
│  │                                                     │    │
│  ├─────────────┬───────────────────┬───────────────────┤    │
│  │   LiDAR     │    X / Y / θ      │    Joystick      │    │
│  │   (scan)    │   Speed Slider    │    Control       │    │
│  └─────────────┴───────────────────┴───────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     ROS2 Humble                              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐ │
│  │ camera_node│ │ rosbridge  │ │ robot_     │ │ mode_    │ │
│  │   (8080)   │ │   (9090)   │ │  driver    │ │controller│ │
│  └────────────┘ └────────────┘ └────────────┘ └──────────┘ │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐              │
│  │ yolo_      │ │ rf2o_laser │ │ LD19       │              │
│  │ detector   │ │ _odometry  │ │ LiDAR      │              │
│  └────────────┘ └────────────┘ └────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

## 포트 구성

| Port | Service | Description |
|------|---------|-------------|
| 8888 | nginx | Web UI |
| 8080 | camera_node | MJPEG 스트림 |
| 9090 | rosbridge | WebSocket |

## Web UI 탭

### Control 탭
- 카메라 실시간 영상 (16:9)
- LiDAR 시각화 (거리별 색상)
- 조이스틱 제어
- Odom 값 (X, Y, θ)
- 속도 조절 슬라이더

### Map 탭
- ros2djs 지도 뷰어
- SLAM / Save / Load / Nav / Stop 버튼
- LiDAR + 조이스틱 제어

### AI 탭
- 카메라 영상 + YOLO 버튼
- 감지 로그 표시
- LiDAR + 조이스틱 제어

## AI 모드

| Mode | 설명 | TensorRT |
|------|------|----------|
| **YOLO** | YOLOv8n 객체 감지 | FP16, 147 FPS |
| **Follower** | 사람 추종 | - |
| **Patrol** | 경로 순찰 | - |
| **Lane** | 차선 주행 | - |

### TensorRT YOLOv8n 성능
- Engine: `yolov8n_fp16.engine` (9MB)
- Throughput: **147 FPS**
- Latency: 7.6ms (mean)
- GPU Compute: 6.8ms

## 카메라

```
IMX219 CSI → nvarguscamerasrc → nvjpegenc → CompressedImage
                                    │
                                    ├→ ROS2: /camera/image_raw/compressed
                                    └→ HTTP: http://IP:8080/stream
```

- 해상도: **640x360** (16:9)
- FPS: **15**
- 하드웨어 JPEG 인코딩

## 빠른 시작

```bash
# 서비스 상태 확인
systemctl status robot.service nginx

# 웹 접속
http://<IP>:8888

# 로그 확인
journalctl -u robot.service -f
```

## 수동 실행

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export KETI_MODEL=robot
export LDS_MODEL=LDS-04

ros2 launch robot_bringup robot.launch.py
```

## 패키지 구조

```
src/
├── robot_bringup/       # Launch 파일
├── robot_driver/        # 모터 드라이버 (Modbus)
├── robot_description/   # URDF
├── robot_web/           # Web UI + Camera
│   ├── www/             # HTML/CSS/JS
│   └── scripts/         # camera_node.py
├── robot_ai/            # AI 모드
│   ├── scripts/         # yolo_detector.py, mode_controller.py
│   └── models/          # TensorRT engines (gitignore)
├── robot_slam/          # Cartographer
├── robot_navigation/    # Nav2
└── ldlidar_stl_ros2/    # LD19 드라이버
```

## 키보드 제어

| Key | Action |
|-----|--------|
| W | 전진 |
| S | 후진 |
| A | 좌회전 |
| D | 우회전 |
| Space | 정지 |

## 하드웨어

- **보드**: Jetson Orin Nano Super (8GB)
- **카메라**: IMX219 CSI
- **LiDAR**: LD19 (360°)
- **모터**: Modbus (ttyTHS1)

## 트러블슈팅

```bash
# 노드 확인
ros2 node list

# 토픽 확인
ros2 topic hz /scan
ros2 topic hz /camera/image_raw/compressed

# 서비스 재시작
sudo systemctl restart robot.service
```

## License

Apache 2.0
