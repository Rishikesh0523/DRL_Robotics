# ~/Downloads/gz_ws/src/ros_gz_hard_demos/ros_gz_hard_demos/nodes/jazzy_camera_publisher.py

#!/usr/bin/env python3
"""
OPTIMIZED MR LOGI Camera Publisher for High Performance
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time


class OptimizedCameraPublisher(Node):
    def __init__(self):
        super().__init__('optimized_camera_publisher')
        
        # Parameters for performance
        self.declare_parameter('camera_device', '/dev/video2')
        self.declare_parameter('target_fps', 30)
        self.declare_parameter('resolution', '640x480')  # Options: 320x240, 640x480
        self.declare_parameter('enable_optimizations', True)
        
        # ROS 2 optimized QoS
        qos_profile = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )
        
        self.raw_pub = self.create_publisher(Image, '/mr_logi/camera/raw', qos_profile)
        self.processed_pub = self.create_publisher(Image, '/mr_logi/camera/processed', qos_profile)
        
        self.bridge = CvBridge()
        self.cap = None
        
        # Performance monitoring
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.last_publish_time = self.get_clock().now()
        
        # Initialize camera with optimizations
        self.initialize_optimized_camera()
        
    def initialize_optimized_camera(self):
        """Initialize camera with performance optimizations"""
        camera_device = self.get_parameter('camera_device').value
        target_fps = self.get_parameter('target_fps').value
        resolution = self.get_parameter('resolution').value
        
        self.get_logger().info(f' Initializing MR LOGI with optimizations...')
        self.get_logger().info(f'   Device: {camera_device}')
        self.get_logger().info(f'   Target FPS: {target_fps}')
        self.get_logger().info(f'   Resolution: {resolution}')
        
        # Parse resolution
        width, height = map(int, resolution.split('x'))
        
        # Open camera with V4L2 backend
        self.cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            self.get_logger().error(' Failed to open camera with V4L2')
            return
        
        # CRITICAL OPTIMIZATIONS:
        # 1. Set resolution first
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # 2. Set frame rate
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)
        
        # 3. Single buffer for lowest latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 4. Use MJPG format for better performance if supported
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        
        # 5. Disable auto features for consistent performance
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual exposure
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)       # Manual white balance
        
        # Verify settings
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.get_logger().info(f' Camera configured: {actual_width}x{actual_height} @ {actual_fps}fps')
        
        # Pre-allocate arrays for performance
        self.drl_width, self.drl_height = 160, 120
        self.processed_frame = np.zeros((self.drl_height, self.drl_width), dtype=np.uint8)
        
        # Start high-performance publishing loop
        publish_period = 1.0 / target_fps
        self.timer = self.create_timer(publish_period, self.high_performance_publish)
        
        self.get_logger().info(' MR LOGI Camera optimized and ready!')
        
    def high_performance_publish(self):
        """High-performance publishing with minimal processing"""
        # Read frame
        ret, frame = self.cap.read()
        
        if not ret:
            self.get_logger().warning('Frame capture failed', throttle_duration_sec=5.0)
            return
        
        current_time = self.get_clock().now()
        
        try:
            # OPTIMIZATION: Publish raw frame immediately (no processing delay)
            raw_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            raw_msg.header.stamp = current_time.to_msg()
            raw_msg.header.frame_id = 'mr_logi_camera'
            self.raw_pub.publish(raw_msg)
            
            # OPTIMIZATION: Fast processing for DRL
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Resize for DRL (use pre-allocated array)
            cv2.resize(gray, (self.drl_width, self.drl_height), dst=self.processed_frame)
            
            # Simple contrast enhancement (faster than histogram equalization)
            cv2.normalize(self.processed_frame, self.processed_frame, 0, 255, cv2.NORM_MINMAX)
            
            # Publish processed frame
            processed_msg = self.bridge.cv2_to_imgmsg(self.processed_frame, 'mono8')
            processed_msg.header.stamp = current_time.to_msg()
            processed_msg.header.frame_id = 'mr_logi_camera'
            self.processed_pub.publish(processed_msg)
            
            # Performance monitoring
            self.frame_count += 1
            if self.frame_count % 60 == 0:  # Log every 60 frames
                current_time_sec = time.time()
                elapsed = current_time_sec - self.last_fps_time
                fps = 60 / elapsed if elapsed > 0 else 0
                self.get_logger().info(f' Performance: {fps:.1f} FPS, Total: {self.frame_count}')
                self.last_fps_time = current_time_sec
                
        except Exception as e:
            self.get_logger().error(f'Processing error: {e}', throttle_duration_sec=2.0)
    
    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            self.get_logger().info(' Camera released')


def main():
    rclpy.init()
    
    try:
        node = OptimizedCameraPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n Optimized camera stopped")
    except Exception as e:
        print(f' Fatal error: {e}')
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()