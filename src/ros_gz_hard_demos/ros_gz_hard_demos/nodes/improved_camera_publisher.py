#!/usr/bin/env python3
"""
Optimized Camera Publisher for Logitech C270 - No JPEG Corruption!
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class LogitechC270Publisher(Node):
    def __init__(self):
        super().__init__('improved_camera_publisher')
        
        # Parameters optimized for Logitech C270
        self.declare_parameter('camera_index', 2)
        self.declare_parameter('frame_rate', 30)  # Max for YUYV 640x480
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        
        self.camera_index = self.get_parameter('camera_index').value
        self.frame_rate = self.get_parameter('frame_rate').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Publisher
        self.publisher = self.create_publisher(
            Image, 
            '/hardware/camera/processed', 
            10
        )
        
        # Initialize camera with YUYV format
        self.cap = self.initialize_camera_yuyv()
        
        if self.cap and self.cap.isOpened():
            self.timer = self.create_timer(1.0 / self.frame_rate, self.publish_frame)
            self.get_logger().info(' Logitech C270 Camera Publisher Started')
            self.get_logger().info('Using YUYV format - No JPEG corruption!')
        else:
            self.get_logger().error(' Failed to initialize camera')
    
    def initialize_camera_yuyv(self):
        """Initialize camera with YUYV format to avoid JPEG corruption"""
        self.get_logger().info('🔧 Initializing Logitech C270 with YUYV format...')
        
        try:
            # Use V4L2 backend for Linux with YUYV format
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            
            if cap.isOpened():
                # Set YUYV format (this avoids JPEG corruption)
                yuyv_fourcc = cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V')
                cap.set(cv2.CAP_PROP_FOURCC, yuyv_fourcc)
                
                # Set resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                
                # Set framerate
                cap.set(cv2.CAP_PROP_FPS, self.frame_rate)
                
                # Small buffer to reduce latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                
                # Test capture
                for i in range(3):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        actual_fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        self.get_logger().info(f' Camera initialized successfully!')
                        self.get_logger().info(f'   Format: YUYV (no JPEG compression)')
                        self.get_logger().info(f'   Resolution: {actual_width}x{actual_height}')
                        self.get_logger().info(f'   FPS: {actual_fps}')
                        self.get_logger().info(f'   Frame shape: {frame.shape}')
                        return cap
                    time.sleep(0.1)
            
            # Fallback to default if V4L2 fails
            self.get_logger().warning('V4L2 failed, trying default backend...')
            cap = cv2.VideoCapture(self.camera_index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                return cap
                
        except Exception as e:
            self.get_logger().error(f'Camera initialization error: {e}')
        
        return None
    
    def publish_frame(self):
        """Publish frame - YUYV format doesn't need special handling"""
        if not self.cap or not self.cap.isOpened():
            return
            
        try:
            ret, frame = self.cap.read()
            
            if ret and frame is not None:
                # Convert to grayscale
                if len(frame.shape) == 3:
                    # YUYV format: we can use standard conversion or extract Y channel
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray_frame = frame
                
                # Create ROS message
                img_msg = self.bridge.cv2_to_imgmsg(gray_frame, encoding='mono8')
                img_msg.header.stamp = self.get_clock().now().to_msg()
                img_msg.header.frame_id = 'camera_frame'
                
                self.publisher.publish(img_msg)
                
                # Log first success
                if not hasattr(self, 'first_frame_published'):
                    self.first_frame_published = True
                    self.get_logger().info('First clean frame published! No JPEG corruption!')
                
            else:
                if not hasattr(self, 'frame_warning_count'):
                    self.frame_warning_count = 0
                self.frame_warning_count += 1
                if self.frame_warning_count % 10 == 0:
                    self.get_logger().warning('Camera frame read issues')
                
        except Exception as e:
            self.get_logger().error(f'Publish error: {e}')
    
    def destroy_node(self):
        """Cleanup"""
        if self.cap:
            self.cap.release()
        super().destroy_node()

def main():
    rclpy.init()
    node = LogitechC270Publisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()