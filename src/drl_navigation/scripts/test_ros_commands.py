#!/usr/bin/env python3
import rclpy
from geometry_msgs.msg import Twist
import time

def test_ros_commands():
    rclpy.init()
    node = rclpy.create_node('command_test')
    pub = node.create_publisher(Twist, '/model/vehicle_blue/cmd_vel', 10)
    
    print("🚀 Testing ROS 2 command publishing...")
    print("Make sure your robot simulation is running!")
    print("You should see the robot move in the simulation.")
    
    # Test 1: Move forward
    print("\n1. Testing forward movement...")
    cmd = Twist()
    cmd.linear.x = 0.3
    cmd.angular.z = 0.0
    pub.publish(cmd)
    print(f"   Command: linear.x={cmd.linear.x}, angular.z={cmd.angular.z}")
    time.sleep(3.0)
    
    # Test 2: Turn left
    print("\n2. Testing left turn...")
    cmd = Twist()
    cmd.linear.x = 0.1
    cmd.angular.z = 0.5
    pub.publish(cmd)
    print(f"   Command: linear.x={cmd.linear.x}, angular.z={cmd.angular.z}")
    time.sleep(2.0)
    
    # Test 3: Turn right
    print("\n3. Testing right turn...")
    cmd = Twist()
    cmd.linear.x = 0.1
    cmd.angular.z = -0.5
    pub.publish(cmd)
    print(f"   Command: linear.x={cmd.linear.x}, angular.z={cmd.angular.z}")
    time.sleep(2.0)
    
    # Test 4: Stop
    print("\n4. Stopping...")
    cmd = Twist()
    cmd.linear.x = 0.0
    cmd.angular.z = 0.0
    pub.publish(cmd)
    print(f"   Command: linear.x={cmd.linear.x}, angular.z={cmd.angular.z}")
    
    node.destroy_node()
    rclpy.shutdown()
    print("\n Test complete! Check if your robot moved in the simulation.")
    print("If the robot didn't move, check:")
    print("  1. is the simulation running?")
    print("  2. is the robot model named 'vehicle_blue'?")
    print("  3. Check topic names with: ros2 topic list")

if __name__ == "__main__":
    test_ros_commands()