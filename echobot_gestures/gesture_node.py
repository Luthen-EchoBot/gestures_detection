import cv2
import mediapipe as mp
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class GestureNode(Node):
    def __init__(self):
        super().__init__('gesture_node')

        self.publisher_ = self.create_publisher(String, 'gesture_detected', 10)

        self.cap = cv2.VideoCapture(0)
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.timer = self.create_timer(0.03, self.loop)

    def loop(self):
        msg = String()
        msg.data = "test_gesture"
        self.publisher_.publish(msg)

def main():
    rclpy.init()
    node = GestureNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

