#!/usr/bin/env python3
"""
Simple LiDAR Processor - Fixed version
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Bool
import numpy as np

class SimpleLidarProcessor(Node):
    def __init__(self):
        super().__init__('simple_lidar_processor')
        
        # Parameters
        self.declare_parameter('collision_threshold', 0.5)  # 0.5 meters
        self.collision_threshold = self.get_parameter('collision_threshold').value
        
        # Subscribe to raw LiDAR
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)
        
        # Publishers for processed data
        self.processed_pub = self.create_publisher(
            Float32MultiArray, '/drl/processed_lidar', 10)
        self.collision_pub = self.create_publisher(
            Bool, '/drl/collision_warning', 10)
        
        self.get_logger().info(' Simple LiDAR Processor Started!')
        self.get_logger().info(f' Collision threshold: {self.collision_threshold}m')
        
        self.count = 0
        
    def lidar_callback(self, msg):
        """Process LiDAR data for navigation"""
        try:
            # Convert to numpy array
            ranges = np.array(msg.ranges, dtype=np.float32)
            
            # Handle invalid values (0.0, inf, nan)
            ranges = np.nan_to_num(ranges, nan=msg.range_max, posinf=msg.range_max)
            
            # Filter out zero values (usually invalid)
            valid_ranges = ranges[ranges > msg.range_min]
            
            if len(valid_ranges) > 0:
                # Get minimum distance (for collision detection)
                min_distance = np.min(valid_ranges)
                
                # Check for collision
                collision = min_distance < self.collision_threshold
                
                # FIX: Create Bool message correctly
                collision_msg = Bool()
                collision_msg.data = bool(collision)  # Explicitly convert to bool
                self.collision_pub.publish(collision_msg)
                
                # Process for DRL: normalize and select key sectors
                processed_data = self.process_for_navigation(ranges, msg)
                
                # Publish processed data
                processed_msg = Float32MultiArray()
                processed_msg.data = processed_data.tolist()
                self.processed_pub.publish(processed_msg)
                
                # Log occasionally
                self.count += 1
                if self.count % 50 == 0:
                    self.get_logger().info(f' LiDAR: Min={min_distance:.2f}m, Collision={collision}')
                    
        except Exception as e:
            self.get_logger().error(f'LiDAR processing error: {e}')
    
    def process_for_navigation(self, ranges, msg):
        """Process LiDAR data for navigation"""
        # Focus on important sectors:
        num_sectors = 8
        sector_size = len(ranges) // num_sectors
        
        processed = []
        for i in range(num_sectors):
            start_idx = i * sector_size
            end_idx = start_idx + sector_size
            sector_data = ranges[start_idx:end_idx]
            
            # Remove invalid values and get minimum in sector
            valid_sector = sector_data[sector_data > msg.range_min]
            if len(valid_sector) > 0:
                sector_min = np.min(valid_sector)
                # Normalize to [0, 1]
                normalized = min(sector_min / msg.range_max, 1.0)
            else:
                normalized = 1.0  # No obstacle
                
            processed.append(float(normalized))  # Explicitly convert to float
        
        return np.array(processed, dtype=np.float32)

def main():
    rclpy.init()
    node = SimpleLidarProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()