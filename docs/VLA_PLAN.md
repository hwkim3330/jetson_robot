# VLA (Vision-Language-Action) 모델 도입 계획

## 발견한 경량 VLA 모델들

### 1. SmolVLA (450M params) - 가장 유망!
- **출처**: [HuggingFace Blog](https://huggingface.co/blog/smolvla) | [arXiv](https://arxiv.org/abs/2506.01844)
- **크기**: 450M 파라미터 (7B 대비 15배 작음)
- **성능**: 10-20배 큰 모델과 비슷한 성능
- **특징**:
  - 싱글 GPU 학습 가능
  - Consumer GPU/CPU에서 실행 가능
  - 비동기 추론으로 30% 빠른 응답
  - 오픈소스 (LeRobot 데이터셋)

```
SmolVLA 아키텍처:
┌─────────────────────────────────────┐
│  SmolVLM2 (Vision-Language Model)   │
│  ├── SigLIP (Vision Encoder)        │
│  └── SmolLM2 (Language Decoder)     │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│       Action Expert (MLP)           │
│       → Robot Control Output        │
└─────────────────────────────────────┘
```

### 2. NaVILA (Navigation VLA with LiDAR)
- **출처**: [Project Page](https://navila-bot.github.io/) | [GitHub](https://github.com/AnjieCheng/NaVILA)
- **특징**: Vision + LiDAR 사용!
- **구조**: 2-레벨 시스템
  - High-level: VLA → 중간 레벨 명령 ("75cm 전진")
  - Low-level: LiDAR 기반 locomotion policy
- **성능**: 실제 환경에서 88% 성공률

```
NaVILA 아키텍처:
┌──────────────┐    ┌──────────────────┐
│   Camera     │───▶│   VLA Model      │
│   Image      │    │ (High-level)     │
└──────────────┘    └────────┬─────────┘
                             │ "move forward 75cm"
                             ▼
┌──────────────┐    ┌──────────────────┐
│   LiDAR      │───▶│   Locomotion     │
│   2.5D Map   │    │   RL Policy      │
└──────────────┘    └────────┬─────────┘
                             │
                             ▼
                      Robot Actions
```

---

## Vision + LiDAR VLA: 현재 상태

### 기존 접근 방식
| 모델 | Vision | LiDAR | 통합 방식 |
|------|--------|-------|-----------|
| OpenVLA | ✅ | ❌ | Vision only |
| SmolVLA | ✅ | ❌ | Vision only |
| NaVILA | ✅ | ✅ | 분리 (2-level) |
| MLA (2025) | ✅ | ✅ | Multisensory fusion |

### 결론: 직접 Vision+LiDAR VLA는 거의 없음!
- 대부분 Vision만 사용
- LiDAR는 별도 안전 시스템으로 사용
- 통합된 End-to-end 모델은 연구 초기 단계

---

## 🚀 Vision+LiDAR VLA 개발 계획

### 목표
```
입력: Camera Image + LiDAR Scan + Language Command
출력: Robot Action (linear_vel, angular_vel)
```

### 제안 아키텍처: "LidarVLA"

```
┌─────────────────────────────────────────────────────────┐
│                    LidarVLA Model                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Camera Image │  │ LiDAR Scan   │  │ Language Cmd │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │          │
│         ▼                 ▼                 ▼          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ SigLIP       │  │ PointNet     │  │ SmolLM2      │  │
│  │ Encoder      │  │ or BEV CNN   │  │ Tokenizer    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │          │
│         └────────┬────────┴────────┬────────┘          │
│                  │                 │                   │
│                  ▼                 ▼                   │
│         ┌──────────────────────────────┐               │
│         │    Cross-Modal Attention     │               │
│         │    (Vision-LiDAR-Language)   │               │
│         └──────────────┬───────────────┘               │
│                        │                               │
│                        ▼                               │
│         ┌──────────────────────────────┐               │
│         │     Action Expert (MLP)      │               │
│         │  → (linear_vel, angular_vel) │               │
│         └──────────────────────────────┘               │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Phase 1: 데이터 수집 (2주)

```python
# 수집할 데이터 구조
{
    "camera_image": "640x480 RGB",
    "lidar_scan": "360 points",
    "language_cmd": "go to the red ball",
    "action": {
        "linear_vel": 0.2,
        "angular_vel": 0.1
    },
    "success": True
}
```

**수집 방법:**
1. 수동 조작하며 데이터 기록
2. 각 에피소드에 자연어 명령 라벨링
3. 목표: 1000+ 에피소드

**ROS2 데이터 수집 노드:**
```python
# data_collector.py
class DataCollector(Node):
    def __init__(self):
        # 동기화된 구독
        self.camera_sub = Subscriber(Image, '/camera/image_raw/compressed')
        self.lidar_sub = Subscriber(LaserScan, '/scan')
        self.cmd_sub = Subscriber(Twist, '/cmd_vel')

        # 동기화
        self.sync = ApproximateTimeSynchronizer(
            [self.camera_sub, self.lidar_sub, self.cmd_sub],
            queue_size=10, slop=0.1
        )
```

### Phase 2: LiDAR 인코더 개발 (1주)

**Option A: BEV (Bird's Eye View) CNN**
```python
class LidarBEVEncoder(nn.Module):
    def __init__(self):
        # LiDAR → 2D 이미지 (위에서 본 시점)
        self.bev_size = (64, 64)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )

    def lidar_to_bev(self, scan):
        # 360도 스캔 → 64x64 occupancy grid
        ...
```

**Option B: PointNet (직접 포인트 처리)**
```python
class LidarPointEncoder(nn.Module):
    def __init__(self):
        self.mlp1 = nn.Linear(2, 64)   # (x, y) per point
        self.mlp2 = nn.Linear(64, 128)
        self.mlp3 = nn.Linear(128, 256)

    def forward(self, points):
        # [B, N, 2] → [B, 256]
        x = F.relu(self.mlp1(points))
        x = F.relu(self.mlp2(x))
        x = self.mlp3(x)
        return x.max(dim=1)[0]  # Max pooling
```

### Phase 3: SmolVLA 수정 (2주)

```python
# 기존 SmolVLA 확장
class LidarVLA(SmolVLA):
    def __init__(self):
        super().__init__()

        # LiDAR 인코더 추가
        self.lidar_encoder = LidarBEVEncoder()

        # Cross-modal fusion
        self.fusion = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8
        )

    def forward(self, image, lidar, text):
        # Vision features
        vis_feat = self.vision_encoder(image)

        # LiDAR features
        lid_feat = self.lidar_encoder(lidar)

        # Language features
        lang_feat = self.language_encoder(text)

        # Fusion
        fused = self.fusion(
            query=lang_feat,
            key=torch.cat([vis_feat, lid_feat], dim=1),
            value=torch.cat([vis_feat, lid_feat], dim=1)
        )

        # Action prediction
        action = self.action_head(fused)
        return action
```

### Phase 4: 학습 (1주)

```python
# 학습 설정
config = {
    "model": "LidarVLA-small",
    "batch_size": 16,
    "lr": 1e-4,
    "epochs": 50,
    "device": "cuda",  # Jetson GPU

    # 데이터 증강
    "augment": {
        "image": ["flip", "color_jitter"],
        "lidar": ["rotate", "noise"],
        "text": ["paraphrase"]
    }
}
```

### Phase 5: 최적화 & 배포 (1주)

```bash
# TensorRT 변환
python export_tensorrt.py \
    --model lidar_vla.pt \
    --precision fp16 \
    --output lidar_vla.engine

# ROS2 노드로 배포
ros2 run robot_ai lidar_vla_node.py
```

---

## 구현 로드맵

```
Week 1-2: 데이터 수집
├── ROS2 데이터 수집 노드 개발
├── 수동 조작으로 에피소드 기록
└── 자연어 명령 라벨링

Week 3: LiDAR 인코더
├── BEV 변환 구현
├── CNN 인코더 학습
└── 단독 성능 검증

Week 4-5: 모델 통합
├── SmolVLA 코드 포크
├── LiDAR 브랜치 추가
├── Cross-modal attention 구현
└── 통합 학습

Week 6: 최적화 & 배포
├── TensorRT 변환
├── ROS2 노드 구현
└── 실제 로봇 테스트
```

---

## 예상 성능

| 메트릭 | 목표 |
|--------|------|
| 모델 크기 | ~500M params |
| 추론 속도 | 10+ Hz (Jetson) |
| 메모리 | < 4GB VRAM |
| 성공률 | > 70% (간단한 명령) |

---

## 필요 리소스

- **하드웨어**: Jetson Orin Nano (현재 보유)
- **데이터**: 자체 수집 (1000+ 에피소드)
- **베이스 모델**: SmolVLA (오픈소스)
- **프레임워크**: PyTorch + TensorRT

---

## 참고 자료

- [SmolVLA - HuggingFace](https://huggingface.co/blog/smolvla)
- [NaVILA - GitHub](https://github.com/AnjieCheng/NaVILA)
- [LeRobot - HuggingFace](https://github.com/huggingface/lerobot)
- [VLA Survey](https://vla-survey.github.io/)

---

## 결론

**Vision+LiDAR VLA는 아직 없다!** → 만들 가치 있음

**SmolVLA 기반으로:**
1. LiDAR 인코더 추가
2. Cross-modal fusion 구현
3. 자체 데이터로 fine-tune
4. Jetson 최적화

**예상 개발 기간: 6주**
