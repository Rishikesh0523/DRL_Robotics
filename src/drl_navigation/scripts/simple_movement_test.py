#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class SimpleMovementTest(Node):
    def __init__(self):
        super().__init__('simple_movement_test')
        self.publisher = self.create_publisher(Twist, '/model/vehicle_blue/cmd_vel', 10)
        
    def move_robot(self):
        print("Testing robot movement...")
        
        # Move forward
        print("Moving forward for 3 seconds...")
        twist = Twist()
        twist.linear.x = 3
        self.publisher.publish(twist)
        time.sleep(3.0)
        
        # Turn
        print("Turning for 2 seconds...")
        twist = Twist()
        twist.linear.x = 0.1
        twist.angular.z = 0.5
        self.publisher.publish(twist)
        time.sleep(2.0)
        
        # Stop
        print("Stopping...")
        twist = Twist()
        self.publisher.publish(twist)
        
        print("Test complete!")

def main():
    rclpy.init()
    node = SimpleMovementTest()
    
    print("=" * 50)
    print("SIMPLE MOVEMENT TEST")
    print("Make sure:")
    print("1. teleop_twist_keyboard is STOPPED (Ctrl+C)")
    print("2. Your simulation is RUNNING")
    print("3. Robot model 'vehicle_blue' exists")
    print("=" * 50)
    
    input("Press Enter to start the movement test...")
    
    node.move_robot()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()