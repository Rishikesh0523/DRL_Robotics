#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import time
import math

class DiagnosticNode(Node):
    def __init__(self):
        super().__init__('diagnostic_node')
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, 
            '/model/vehicle_blue/odometry',
            self.odom_callback, 
            10
        )
        
        self.lidar_sub = self.create_subscription(
            LaserScan, 
            '/lidar',
            self.lidar_callback, 
            10
        )
        
        # Publisher for testing
        self.cmd_pub = self.create_publisher(Twist, '/model/vehicle_blue/cmd_vel', 10)
        
        self.get_logger().info(" Diagnostic Node Started")
        self.get_logger().info("Testing movement and sensors...")
        
        # Test movement
        self.test_movement()
    
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.get_logger().info(f" Odometry - X: {x:.2f}, Y: {y:.2f}")
    
    def lidar_callback(self, msg):
        if msg.ranges:
            min_dist = min([r for r in msg.ranges if r > 0.1])
            self.get_logger().info(f" LiDAR - Min distance: {min_dist:.2f}")
    
    def test_movement(self):
        self.get_logger().info(" Testing robot movement...")
        
        # Move forward
        cmd = Twist()
        cmd.linear.x = 0.3
        self.cmd_pub.publish(cmd)
        self.get_logger().info("▶  Sending forward command")
        
        time.sleep(2.0)
        
        # Stop
        cmd.linear.x = 0.0
        self.cmd_pub.publish(cmd)
        self.get_logger().info("⏹  Stopping robot")
        
        time.sleep(1.0)
        
        # Turn
        cmd.angular.z = 0.5
        self.cmd_pub.publish(cmd)
        self.get_logger().info("↩  Sending turn command")
        
        time.sleep(2.0)
        
        # Stop
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
        self.get_logger().info(" Diagnostic test complete")

def main():
    rclpy.init()
    node = DiagnosticNode()
    
    try:
        # Run for 10 seconds
        start_time = time.time()
        while time.time() - start_time < 10:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()