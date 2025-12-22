#!/usr/bin/env python3
"""
Gesture Detector Node for KETI Robot
Uses MediaPipe or OpenCV for hand gesture recognition
Commands robot based on hand gestures:
- Open palm: Move forward
- Fist: Stop
- Point left: Turn left
- Point right: Turn right
- Thumbs up: Follow mode

Falls back to OpenCV-based detection if MediaPipe unavailable
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Bool, String, Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np

# Try to import MediaPipe
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    pass


class GestureDetector(Node):
    def __init__(self):
        super().__init__('gesture_detector')

        # Parameters
        self.declare_parameter('detection_rate', 10.0)
        self.declare_parameter('min_detection_confidence', 0.7)
        self.declare_parameter('min_tracking_confidence', 0.5)
        self.declare_parameter('enable_control', False)
        self.declare_parameter('linear_speed', 0.15)
        self.declare_parameter('angular_speed', 0.5)

        self.detection_rate = self.get_parameter('detection_rate').value
        self.min_detection_conf = self.get_parameter('min_detection_confidence').value
        self.min_tracking_conf = self.get_parameter('min_tracking_confidence').value
        self.enable_control = self.get_parameter('enable_control').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value

        self.bridge = CvBridge()
        self.last_frame = None
        self.current_gesture = "none"
        self.hands = None

        # Initialize detector
        self._init_detector()

        # QoS
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, qos)
        self.compressed_sub = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.compressed_callback, qos)
        self.enable_sub = self.create_subscription(
            Bool, '/gesture_enable', self.enable_callback, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.gesture_pub = self.create_publisher(String, '/gesture', 10)
        self.hand_pub = self.create_publisher(Point, '/hand_position', 10)
        self.landmarks_pub = self.create_publisher(Float32MultiArray, '/hand_landmarks', 10)
        self.debug_pub = self.create_publisher(Image, '/gesture/debug', 10)

        # Timer
        period = 1.0 / self.detection_rate
        self.timer = self.create_timer(period, self.detect_callback)

        backend = "MediaPipe" if MEDIAPIPE_AVAILABLE else "OpenCV (skin detection)"
        self.get_logger().info(f'Gesture Detector initialized using {backend}')
        self.get_logger().info('Publish True to /gesture_enable to enable robot control')

    def _init_detector(self):
        """Initialize hand detector"""
        if MEDIAPIPE_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=self.min_detection_conf,
                min_tracking_confidence=self.min_tracking_conf
            )
            self.get_logger().info('MediaPipe Hands initialized')
        else:
            self.get_logger().warn('MediaPipe not available, using skin color detection')

    def enable_callback(self, msg):
        """Enable/disable robot control"""
        self.enable_control = msg.data
        if self.enable_control:
            self.get_logger().info('Gesture control ENABLED')
        else:
            self.get_logger().info('Gesture control DISABLED')
            self.stop_robot()

    def image_callback(self, msg):
        """Handle incoming images"""
        try:
            self.last_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            pass

    def compressed_callback(self, msg):
        """Handle compressed images"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.last_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            pass

    def detect_callback(self):
        """Run gesture detection"""
        if self.last_frame is None:
            return

        frame = self.last_frame.copy()
        h, w = frame.shape[:2]
        gesture = "none"
        hand_center = None

        if MEDIAPIPE_AVAILABLE and self.hands:
            gesture, hand_center, frame = self._detect_mediapipe(frame)
        else:
            gesture, hand_center, frame = self._detect_skin(frame)

        self.current_gesture = gesture

        # Publish gesture
        gesture_msg = String()
        gesture_msg.data = gesture
        self.gesture_pub.publish(gesture_msg)

        # Publish hand position
        if hand_center:
            point = Point()
            point.x = (hand_center[0] - w/2) / (w/2)
            point.y = (hand_center[1] - h/2) / (h/2)
            point.z = 0.0
            self.hand_pub.publish(point)

        # Control robot
        if self.enable_control:
            self._control_robot(gesture, hand_center, w)

        # Draw gesture on frame
        cv2.putText(frame, f'Gesture: {gesture}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            self.debug_pub.publish(debug_msg)
        except:
            pass

    def _detect_mediapipe(self, frame):
        """Detect hand gestures using MediaPipe"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        gesture = "none"
        hand_center = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Get landmark positions (normalized 0-1)
            h, w = frame.shape[:2]
            landmarks = []
            landmarks_normalized = []
            for lm in hand_landmarks.landmark:
                landmarks.append((int(lm.x * w), int(lm.y * h)))
                landmarks_normalized.extend([lm.x, lm.y])  # 21 points * 2 = 42 floats

            # Publish landmarks for web overlay
            lm_msg = Float32MultiArray()
            lm_msg.data = landmarks_normalized
            self.landmarks_pub.publish(lm_msg)

            # Calculate hand center
            xs = [l[0] for l in landmarks]
            ys = [l[1] for l in landmarks]
            hand_center = (sum(xs) // len(xs), sum(ys) // len(ys))

            # Analyze gesture
            gesture = self._analyze_gesture(landmarks)
        else:
            # No hand - publish empty landmarks
            lm_msg = Float32MultiArray()
            lm_msg.data = []
            self.landmarks_pub.publish(lm_msg)

        return gesture, hand_center, frame

    def _analyze_gesture(self, landmarks):
        """Analyze hand landmarks to determine gesture"""
        if len(landmarks) < 21:
            return "none"

        # Finger tips and bases
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        thumb_base = landmarks[2]
        index_base = landmarks[5]
        middle_base = landmarks[9]
        ring_base = landmarks[13]
        pinky_base = landmarks[17]

        wrist = landmarks[0]

        # Check if fingers are extended
        def is_extended(tip, base):
            return tip[1] < base[1]  # Y increases downward

        index_up = is_extended(index_tip, index_base)
        middle_up = is_extended(middle_tip, middle_base)
        ring_up = is_extended(ring_tip, ring_base)
        pinky_up = is_extended(pinky_tip, pinky_base)

        # Thumb check (horizontal)
        thumb_out = abs(thumb_tip[0] - thumb_base[0]) > 30

        # Count extended fingers
        fingers_up = sum([index_up, middle_up, ring_up, pinky_up])

        # Gesture recognition
        if fingers_up >= 4:
            return "open_palm"  # Stop
        elif fingers_up == 0:
            return "fist"  # Forward
        elif index_up and not middle_up and not ring_up and not pinky_up:
            # Pointing - check direction
            if index_tip[0] < wrist[0] - 50:
                return "point_left"
            elif index_tip[0] > wrist[0] + 50:
                return "point_right"
            else:
                return "point_up"
        elif thumb_out and not index_up and not middle_up:
            return "thumbs_up"  # Follow mode
        elif index_up and middle_up and not ring_up and not pinky_up:
            return "peace"  # Special command

        return "unknown"

    def _detect_skin(self, frame):
        """Fallback skin color detection"""
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        # Skin color range
        lower = np.array([0, 135, 85], dtype=np.uint8)
        upper = np.array([255, 180, 135], dtype=np.uint8)

        mask = cv2.inRange(ycrcb, lower, upper)

        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        gesture = "none"
        hand_center = None

        if contours:
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > 5000:
                # Get bounding box
                x, y, bw, bh = cv2.boundingRect(largest)
                hand_center = (x + bw // 2, y + bh // 2)

                # Draw
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.circle(frame, hand_center, 5, (0, 0, 255), -1)

                # Simple gesture based on position
                h, w = frame.shape[:2]
                if hand_center[0] < w // 3:
                    gesture = "left"
                elif hand_center[0] > 2 * w // 3:
                    gesture = "right"
                else:
                    gesture = "center"

        return gesture, hand_center, frame

    def _control_robot(self, gesture, hand_center, frame_width):
        """Control robot based on gesture"""
        cmd = Twist()

        if gesture == "fist":
            # Stop
            pass
        elif gesture == "open_palm":
            cmd.linear.x = self.linear_speed
        elif gesture == "point_left" or gesture == "left":
            cmd.angular.z = self.angular_speed
        elif gesture == "point_right" or gesture == "right":
            cmd.angular.z = -self.angular_speed
        elif gesture == "thumbs_up":
            # Enable follow mode (handled elsewhere)
            pass
        elif gesture == "center" and hand_center:
            # Move toward hand position
            error = (hand_center[0] - frame_width / 2) / (frame_width / 2)
            cmd.angular.z = -error * self.angular_speed
            if abs(error) < 0.3:
                cmd.linear.x = self.linear_speed * 0.5

        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        """Stop the robot"""
        self.cmd_pub.publish(Twist())

    def destroy_node(self):
        self.stop_robot()
        if self.hands:
            self.hands.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GestureDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
