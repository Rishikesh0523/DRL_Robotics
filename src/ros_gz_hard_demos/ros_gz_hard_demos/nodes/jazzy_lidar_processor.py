#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool
import threading

class JazzyLidarProcessor:
    def __init__(self):
        rospy.init_node('jazzy_lidar_processor', anonymous=True)
        
        # LiDAR parameters
        self.lidar_range_min = 0.1
        self.lidar_range_max = 10.0
        self.num_lidar_beams = 360
        self.collision_threshold = 0.3
        
        # LiDAR data
        self.lidar_data = None
        self.lidar_lock = threading.Lock()
        
        # Publishers and Subscribers
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.processed_lidar_pub = rospy.Publisher('/processed_lidar', Float32MultiArray, queue_size=10)
        self.collision_pub = rospy.Publisher('/collision_warning', Bool, queue_size=10)
        
        # Initialize processed lidar data
        self.processed_lidar = np.zeros(self.num_lidar_beams, dtype=np.float32)
        
        rospy.loginfo("Jazzy LiDAR Processor initialized")

    def lidar_callback(self, data):
        """Callback for raw LiDAR data"""
        with self.lidar_lock:
            self.lidar_data = data
            self.process_lidar_data()

    def process_lidar_data(self):
        """Process LiDAR data for DRL consumption"""
        if self.lidar_data is None:
            return

        try:
            # Convert to numpy array and handle inf/nan values
            ranges = np.array(self.lidar_data.ranges)
            ranges = np.nan_to_num(ranges, nan=self.lidar_range_max, posinf=self.lidar_range_max, neginf=self.lidar_range_min)
            ranges = np.clip(ranges, self.lidar_range_min, self.lidar_range_max)
            
            # Downsample to fixed number of beams if necessary
            if len(ranges) != self.num_lidar_beams:
                if len(ranges) > self.num_lidar_beams:
                    # Downsample
                    indices = np.linspace(0, len(ranges)-1, self.num_lidar_beams, dtype=int)
                    ranges = ranges[indices]
                else:
                    # Upsample with interpolation
                    ranges = np.interp(
                        np.linspace(0, len(ranges)-1, self.num_lidar_beams),
                        np.arange(len(ranges)),
                        ranges
                    )
            
            # Normalize to [0, 1]
            normalized_ranges = (ranges - self.lidar_range_min) / (self.lidar_range_max - self.lidar_range_min)
            self.processed_lidar = normalized_ranges.astype(np.float32)
            
            # Publish processed lidar data
            lidar_msg = Float32MultiArray()
            lidar_msg.data = self.processed_lidar.tolist()
            self.processed_lidar_pub.publish(lidar_msg)
            
            # Check for collision
            self.check_collision(ranges)
            
        except Exception as e:
            rospy.logerr(f"Error processing LiDAR data: {e}")

    def check_collision(self, ranges):
        """Check for imminent collision"""
        min_distance = np.min(ranges)
        collision_warning = min_distance < self.collision_threshold
        
        collision_msg = Bool()
        collision_msg.data = collision_warning
        self.collision_pub.publish(collision_msg)
        
        if collision_warning:
            rospy.logwarn(f"Collision warning! Minimum distance: {min_distance:.2f}m")

    def get_processed_lidar(self):
        """Get the latest processed LiDAR data"""
        with self.lidar_lock:
            return self.processed_lidar.copy()

    def get_collision_risk(self, ranges=None):
        """Get collision risk assessment"""
        if ranges is None:
            with self.lidar_lock:
                if self.lidar_data is not None:
                    ranges = np.array(self.lidar_data.ranges)
                else:
                    return False, self.lidar_range_max
        
        ranges_clean = np.nan_to_num(ranges, nan=self.lidar_range_max)
        min_distance = np.min(ranges_clean)
        return min_distance < self.collision_threshold, min_distance

if __name__ == '__main__':
    try:
        lidar_processor = JazzyLidarProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass