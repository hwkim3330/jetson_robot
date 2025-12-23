# TODO - 앞으로 할 것들

## 현재 완료된 기능
- [x] 웹 UI (카메라, 조이스틱, 맵)
- [x] MediaPipe 제스처 인식
- [x] YOLO 객체 탐지 (TensorRT)
- [x] SLAM 매핑
- [x] Nav2 네비게이션
- [x] 카메라 스트림 (MJPEG)

---

## 단기 TODO

### 웹 UI 개선
- [ ] 제스처 인식 정확도 향상
- [ ] YOLO 박스 성능 최적화
- [ ] 모바일 UI 개선

### 시스템 안정화
- [ ] 자동 시작 스크립트 개선
- [ ] 에러 복구 로직 강화
- [ ] 로그 시스템 정리

---

## 중기 TODO

### VLM (Vision-Language Model) 도입
- [ ] Florence-2 또는 CLIP 테스트
- [ ] "빨간 공 어디?" → 좌표 출력
- [ ] Nav2와 연동

### 데이터 수집 시스템
- [ ] ROS2 데이터 수집 노드 개발
- [ ] Camera + LiDAR + cmd_vel 동기화
- [ ] 웹 UI에서 라벨링 기능

---

## 장기 TODO

### VLA (Vision-Language-Action) 개발
- [ ] SmolVLA 기반 LidarVLA 개발
- [ ] Vision + LiDAR 통합 모델
- [ ] TensorRT 최적화
- [ ] 자연어 명령 지원

### 시뮬레이션
- [ ] Gazebo 환경 구축
- [ ] URDF 모델 정리
- [ ] (선택) Isaac Sim 연동

---

## 아이디어

### 온라인 학습
```
웹 UI에서 조작하면서:
1. 실시간 데이터 수집
2. 좋아요/싫어요 피드백
3. 점진적 모델 개선
```

### 하이브리드 아키텍처
```
VLM (고수준) → 목표 인식 → 좌표
     ↓
Nav2 (저수준) → 경로 계획 → 이동
     ↓
LiDAR (안전) → 장애물 회피
```

---

## 참고 링크
- [SmolVLA](https://huggingface.co/blog/smolvla)
- [NaVILA](https://navila-bot.github.io/)
- [LeRobot](https://github.com/huggingface/lerobot)
- [Florence-2](https://huggingface.co/microsoft/Florence-2-base)
