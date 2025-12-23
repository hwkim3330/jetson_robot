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
        self.declare_parameter('min_detection_confidence', 0.4)
        self.declare_parameter('min_tracking_confidence', 0.3)
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

        # QoS - use RELIABLE to match camera publisher
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)

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
                static_image_mode=True,  # Better for camera stream
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
            if self.last_frame is not None:
                self.get_logger().debug(f'Got frame: {self.last_frame.shape}')
        except Exception as e:
            self.get_logger().error(f'Decode error: {e}')

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

        # Debug: log detection status
        has_hands = results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) > 0
        self.get_logger().info(f'Frame {frame.shape}, hands detected: {has_hands}', throttle_duration_sec=2.0)

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
        """
        Analyze hand landmarks to determine gesture.
        Orientation-independent: works with hand upside down, sideways, etc.

        MediaPipe Hand Landmarks:
        0-WRIST, 1-THUMB_CMC, 2-THUMB_MCP, 3-THUMB_IP, 4-THUMB_TIP
        5-INDEX_MCP, 6-INDEX_PIP, 7-INDEX_DIP, 8-INDEX_TIP
        9-MIDDLE_MCP, 10-MIDDLE_PIP, 11-MIDDLE_DIP, 12-MIDDLE_TIP
        13-RING_MCP, 14-RING_PIP, 15-RING_DIP, 16-RING_TIP
        17-PINKY_MCP, 18-PINKY_PIP, 19-PINKY_DIP, 20-PINKY_TIP
        """
        if len(landmarks) < 21:
            return "none"

        def dist(p1, p2):
            """Calculate distance between two points"""
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

        def normalize(v):
            """Normalize a vector"""
            mag = (v[0]**2 + v[1]**2) ** 0.5
            if mag == 0:
                return (0, 0)
            return (v[0] / mag, v[1] / mag)

        def dot(v1, v2):
            """Dot product of two vectors"""
            return v1[0] * v2[0] + v1[1] * v2[1]

        def cross_2d(v1, v2):
            """2D cross product (z component)"""
            return v1[0] * v2[1] - v1[1] * v2[0]

        # Key landmarks
        wrist = landmarks[0]
        thumb_cmc = landmarks[1]
        thumb_mcp = landmarks[2]
        thumb_ip = landmarks[3]
        thumb_tip = landmarks[4]

        index_mcp = landmarks[5]
        index_pip = landmarks[6]
        index_dip = landmarks[7]
        index_tip = landmarks[8]

        middle_mcp = landmarks[9]
        middle_pip = landmarks[10]
        middle_dip = landmarks[11]
        middle_tip = landmarks[12]

        ring_mcp = landmarks[13]
        ring_pip = landmarks[14]
        ring_dip = landmarks[15]
        ring_tip = landmarks[16]

        pinky_mcp = landmarks[17]
        pinky_pip = landmarks[18]
        pinky_dip = landmarks[19]
        pinky_tip = landmarks[20]

        # Calculate palm reference vectors (orientation-independent)
        # Palm direction: wrist -> middle_mcp (points toward fingers)
        palm_vec = (middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1])
        palm_dir = normalize(palm_vec)
        palm_size = dist(wrist, middle_mcp)

        if palm_size < 10:
            return "none"

        # Palm width direction (perpendicular to palm_dir)
        # For right hand: index side is "left" of palm_dir
        palm_width_vec = (index_mcp[0] - pinky_mcp[0], index_mcp[1] - pinky_mcp[1])
        palm_width = dist(index_mcp, pinky_mcp)

        # Finger extension detection using angle-based method
        # A finger is extended if it's relatively straight (tip far from MCP in finger direction)
        def is_finger_extended(mcp, pip, dip, tip):
            # Method 1: Distance ratio (works regardless of orientation)
            dist_mcp_tip = dist(mcp, tip)
            dist_mcp_pip = dist(mcp, pip)

            # Method 2: Check if finger is straight (pip-dip-tip angle)
            # Bent finger has small angle, extended has large angle (~180)
            v1 = (pip[0] - dip[0], pip[1] - dip[1])
            v2 = (tip[0] - dip[0], tip[1] - dip[1])
            dot_val = dot(v1, v2)
            mag1 = (v1[0]**2 + v1[1]**2) ** 0.5
            mag2 = (v2[0]**2 + v2[1]**2) ** 0.5

            # Also check tip is farther from wrist than pip (finger pointing outward)
            tip_from_wrist = dist(wrist, tip)
            pip_from_wrist = dist(wrist, pip)

            # Combine criteria:
            # 1. Tip significantly farther from MCP than PIP
            # 2. Tip farther from wrist than PIP (finger pointing away)
            is_extended = (dist_mcp_tip > dist_mcp_pip * 1.4) and (tip_from_wrist > pip_from_wrist * 0.9)

            return is_extended

        # Simpler extension check for 3-point fingers (no DIP needed)
        def is_finger_extended_simple(mcp, pip, tip):
            dist_mcp_tip = dist(mcp, tip)
            dist_mcp_pip = dist(mcp, pip)
            tip_from_wrist = dist(wrist, tip)
            pip_from_wrist = dist(wrist, pip)
            return (dist_mcp_tip > dist_mcp_pip * 1.4) and (tip_from_wrist > pip_from_wrist * 0.85)

        # Thumb extension: special case
        def is_thumb_extended():
            dist_tip = dist(thumb_mcp, thumb_tip)
            dist_ip = dist(thumb_mcp, thumb_ip)
            # Thumb spread: distance from thumb tip to index mcp
            thumb_spread = dist(thumb_tip, index_mcp)
            # Extended if: tip far from mcp AND spread away from palm
            return dist_tip > dist_ip * 1.2 and thumb_spread > palm_width * 0.4

        # Check each finger
        thumb_ext = is_thumb_extended()
        index_ext = is_finger_extended(index_mcp, index_pip, index_dip, index_tip)
        middle_ext = is_finger_extended(middle_mcp, middle_pip, middle_dip, middle_tip)
        ring_ext = is_finger_extended(ring_mcp, ring_pip, ring_dip, ring_tip)
        pinky_ext = is_finger_extended(pinky_mcp, pinky_pip, pinky_dip, pinky_tip)

        fingers_extended = sum([index_ext, middle_ext, ring_ext, pinky_ext])
        all_fingers = sum([thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext])

        # Get pointing direction relative to screen (for robot control)
        # Uses the extended finger's direction
        def get_pointing_direction():
            # Vector from wrist to index tip
            point_vec = (index_tip[0] - wrist[0], index_tip[1] - wrist[1])
            mag = (point_vec[0]**2 + point_vec[1]**2) ** 0.5
            if mag < palm_size * 0.5:
                return "none"

            # Use absolute screen coordinates for robot control
            dx, dy = point_vec

            # Determine dominant direction
            if abs(dx) > abs(dy) * 1.3:  # Horizontal dominant
                return "left" if dx < 0 else "right"
            elif abs(dy) > abs(dx) * 1.3:  # Vertical dominant
                return "up" if dy < 0 else "down"
            else:
                # Diagonal - use stronger component
                if abs(dx) > abs(dy):
                    return "left" if dx < 0 else "right"
                else:
                    return "up" if dy < 0 else "down"

        # Thumbs up/down detection (orientation-independent)
        def get_thumb_gesture():
            if not thumb_ext:
                return None
            if fingers_extended > 0:
                return None

            # Thumb direction vector
            thumb_vec = (thumb_tip[0] - thumb_mcp[0], thumb_tip[1] - thumb_mcp[1])
            thumb_mag = (thumb_vec[0]**2 + thumb_vec[1]**2) ** 0.5
            if thumb_mag < palm_size * 0.2:
                return None

            # Check if thumb is perpendicular to palm (pointing away from fingers)
            # Dot product with palm_dir should be small (perpendicular)
            thumb_dir = normalize(thumb_vec)
            alignment = abs(dot(thumb_dir, palm_dir))

            if alignment < 0.5:  # Thumb is roughly perpendicular to palm
                # Check if pointing up or down (screen coordinates)
                if thumb_vec[1] < -palm_size * 0.2:  # Pointing up
                    return "thumbs_up"
                elif thumb_vec[1] > palm_size * 0.2:  # Pointing down
                    return "thumbs_down"
            return None

        # ========== Gesture Recognition ==========

        # 1. STOP - Open palm (all/most fingers extended)
        if all_fingers >= 4 and fingers_extended >= 3:
            return "stop"

        # 2. THUMBS UP/DOWN - Only thumb extended
        thumb_gesture = get_thumb_gesture()
        if thumb_gesture:
            return thumb_gesture

        # 3. GO/FIST - All fingers closed
        if fingers_extended == 0 and not thumb_ext:
            return "go"

        # 4. POINTING - Only index finger extended
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            direction = get_pointing_direction()
            if direction == "left":
                return "left"
            elif direction == "right":
                return "right"
            elif direction == "up":
                return "forward"
            elif direction == "down":
                return "backward"
            return "point"

        # 5. PEACE/TWO - Index and middle extended
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return "peace"

        # 6. THREE - Index, middle, ring extended
        if index_ext and middle_ext and ring_ext and not pinky_ext:
            return "three"

        # 7. FOUR - All except thumb
        if index_ext and middle_ext and ring_ext and pinky_ext and not thumb_ext:
            return "four"

        # 8. OK sign - Thumb and index tips close together
        thumb_index_dist = dist(thumb_tip, index_tip)
        if thumb_index_dist < palm_size * 0.35:
            if middle_ext or ring_ext:  # Other fingers extended
                return "ok"

        # 9. ROCK (pinky and index only) - "I love you" or rock sign
        if index_ext and pinky_ext and not middle_ext and not ring_ext:
            return "rock"

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
        """
        Control robot based on gesture.

        Gesture commands:
        - stop: Stop the robot (open palm)
        - go: Move forward (fist)
        - left: Turn left (point left)
        - right: Turn right (point right)
        - forward: Move forward (point up)
        - backward: Move backward (point down)
        - thumbs_up: Follow mode toggle
        - peace: Speed boost
        """
        cmd = Twist()

        if gesture == "stop":
            # Open palm - STOP
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        elif gesture == "go":
            # Fist - GO forward
            cmd.linear.x = self.linear_speed
        elif gesture == "left":
            # Point left - Turn left
            cmd.angular.z = self.angular_speed
        elif gesture == "right":
            # Point right - Turn right
            cmd.angular.z = -self.angular_speed
        elif gesture == "forward":
            # Point up - Move forward
            cmd.linear.x = self.linear_speed
        elif gesture == "backward":
            # Point down - Move backward
            cmd.linear.x = -self.linear_speed * 0.5
        elif gesture == "thumbs_up":
            # Thumbs up - Follow mode (handled elsewhere)
            pass
        elif gesture == "peace":
            # Peace sign - Fast forward
            cmd.linear.x = self.linear_speed * 1.5
        elif gesture == "three":
            # Three fingers - Medium speed forward + slight turn
            cmd.linear.x = self.linear_speed * 0.7

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
