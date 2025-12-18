#!/usr/bin/env python3
"""
YOLOv8 TensorRT Detector Node for KETI Robot
Uses TensorRT engine for optimal Jetson performance

Topics:
    Subscribes: /camera/image_raw/compressed (CompressedImage)
    Publishes: /detections, /person_target, /yolo/debug
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, Float32, String
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import json


class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        # Parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('detection_rate', 10.0)
        self.declare_parameter('target_class', 0)  # 0 = person in COCO
        self.declare_parameter('use_tensorrt', True)

        self.model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.detection_rate = self.get_parameter('detection_rate').value
        self.target_class = self.get_parameter('target_class').value
        self.use_tensorrt = self.get_parameter('use_tensorrt').value

        self.bridge = CvBridge()
        self.model = None
        self.last_frame = None
        self.ultralytics_available = False

        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

        # Load model
        self._load_model()

        # QoS
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Subscribers (CompressedImage for nvjpegenc camera)
        self.image_sub = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.image_callback, qos)

        # Publishers
        self.person_pub = self.create_publisher(Point, '/person_target', 10)
        self.detected_pub = self.create_publisher(Bool, '/person_detected', 10)
        self.distance_pub = self.create_publisher(Float32, '/person_distance', 10)
        self.detections_pub = self.create_publisher(String, '/detections', 10)
        self.debug_pub = self.create_publisher(Image, '/yolo/debug', 10)

        # Timer
        period = 1.0 / self.detection_rate
        self.timer = self.create_timer(period, self.detect_callback)

        self.get_logger().info(f'YOLO Detector initialized (rate: {self.detection_rate} Hz)')

    def _load_model(self):
        """Load YOLO model with TensorRT if available"""
        try:
            from ultralytics import YOLO
            self.ultralytics_available = True
        except ImportError:
            self.get_logger().warn('Ultralytics not installed. Install with: pip3 install ultralytics')
            return

        # Find model file
        model_paths = [
            self.model_path,
            '/home/nvidia/ros2_ws/src/robot_ai/models/yolov8n_fp16.engine',
            '/home/nvidia/ros2_ws/src/robot_ai/models/yolov8n.pt',
            'yolov8n.pt'  # Will download if not found
        ]

        for path in model_paths:
            if path and os.path.exists(path):
                try:
                    self.model = YOLO(path)
                    self.get_logger().info(f'Loaded model: {path}')

                    # Check if TensorRT
                    if path.endswith('.engine'):
                        self.get_logger().info('Using TensorRT engine (GPU/DLA accelerated)')
                    elif path.endswith('.pt'):
                        self.get_logger().info('Using PyTorch model (GPU accelerated)')

                    return
                except Exception as e:
                    self.get_logger().warn(f'Failed to load {path}: {e}')
                    continue

        # Fallback: download yolov8n
        try:
            self.get_logger().info('Downloading YOLOv8n...')
            self.model = YOLO('yolov8n.pt')
            self.get_logger().info('YOLOv8n loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load any model: {e}')

    def image_callback(self, msg):
        """Handle incoming compressed images"""
        try:
            # Decompress JPEG
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.last_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')

    def detect_callback(self):
        """Run detection at fixed rate"""
        if self.last_frame is None or self.model is None:
            return

        frame = self.last_frame.copy()
        h, w = frame.shape[:2]

        # Run inference
        try:
            results = self.model(frame, verbose=False, conf=self.conf_threshold)
        except Exception as e:
            self.get_logger().error(f'Inference error: {e}')
            return

        person_detected = False
        target_point = Point()
        distance = 0.0
        all_detections = []

        # Process results
        if results and len(results) > 0:
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                # Find best person detection
                best_person = None
                best_conf = 0

                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()

                    # Store all detections
                    all_detections.append({
                        'class': self.class_names[cls] if cls < len(self.class_names) else str(cls),
                        'confidence': conf,
                        'bbox': xyxy.tolist()
                    })

                    # Track person (class 0)
                    if cls == self.target_class and conf > best_conf:
                        best_conf = conf
                        best_person = xyxy

                    # Draw on debug frame
                    x1, y1, x2, y2 = map(int, xyxy)
                    color = (0, 255, 0) if cls == self.target_class else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label = f'{self.class_names[cls] if cls < len(self.class_names) else cls}: {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Process best person detection
                if best_person is not None:
                    x1, y1, x2, y2 = best_person
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    box_height = y2 - y1

                    # Normalize to [-1, 1]
                    target_point.x = (center_x - w/2) / (w/2)
                    target_point.y = (center_y - h/2) / (h/2)
                    target_point.z = best_conf

                    # Estimate distance
                    distance = (h * 0.7) / max(box_height, 1)
                    person_detected = True

                    # Highlight person
                    cv2.circle(frame, (int(center_x), int(center_y)), 8, (0, 0, 255), -1)

        # Publish results
        detected_msg = Bool()
        detected_msg.data = person_detected
        self.detected_pub.publish(detected_msg)

        if person_detected:
            self.person_pub.publish(target_point)
            dist_msg = Float32()
            dist_msg.data = distance
            self.distance_pub.publish(dist_msg)

        # Publish all detections as JSON
        if all_detections:
            det_msg = String()
            det_msg.data = json.dumps(all_detections)
            self.detections_pub.publish(det_msg)

        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            self.debug_pub.publish(debug_msg)
        except:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
