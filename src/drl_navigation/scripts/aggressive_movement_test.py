#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class AggressiveMovementTest(Node):
    def __init__(self):
        super().__init__('aggressive_test')
        self.publisher = self.create_publisher(Twist, '/model/vehicle_blue/cmd_vel', 10)
        
    def test_movement(self):
        print("🚀 AGGRESSIVE MOVEMENT TEST")
        print("Sending strong commands repeatedly...")
        
        # Test 1: Strong forward
        print("\n1. STRONG FORWARD (linear.x = 1.0)")
        twist = Twist()
        twist.linear.x = 1.0  # Maximum speed
        for i in range(50):  # Send 50 times
            self.publisher.publish(twist)
            time.sleep(0.1)
        print("   Sent 50 forward commands")
        
        # Test 2: Strong turn
        print("\n2. STRONG TURN (angular.z = 1.0)")
        twist = Twist()
        twist.angular.z = 1.0  # Maximum turn
        for i in range(30):
            self.publisher.publish(twist)
            time.sleep(0.1)
        print("   Sent 30 turn commands")
        
        # Stop
        twist = Twist()
        self.publisher.publish(twist)
        print("\n✅ Test complete - all commands sent")

def main():
    rclpy.init()
    tester = AggressiveMovementTest()
    
    print("=" * 60)
    print("AGGRESSIVE MOVEMENT TEST")
    print("This sends STRONG, REPEATED commands")
    print("Watch your simulation carefully!")
    print("=" * 60)
    
    input("Press Enter to start aggressive test...")
    
    tester.test_movement()
    tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()