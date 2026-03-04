#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import time

class PhysicsChecker(Node):
    def __init__(self):
        super().__init__('physics_checker')
        self.positions = []
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/model/vehicle_blue/odometry',
            self.odom_callback,
            10
        )
        
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.positions.append((x, y))
        print(f"Position: ({x:.3f}, {y:.3f})")
        
    def check_movement(self):
        print(" Checking if robot has physics...")
        print("Try manually PUSHING the robot in your simulation!")
        print("Watch the position values above - they should change if physics works.")
        print("Waiting 10 seconds for manual testing...")
        
        time.sleep(10.0)
        
        if len(self.positions) > 1:
            # Check if position changed
            first_pos = self.positions[0]
            last_pos = self.positions[-1]
            distance = ((last_pos[0] - first_pos[0])**2 + (last_pos[1] - first_pos[1])**2)**0.5
            
            if distance > 0.01:
                print(f" Physics WORKS! Robot moved {distance:.3f} meters")
                return True
            else:
                print(" Physics NOT WORKING - robot didn't move when pushed")
                return False
        else:
            print(" No position data received")
            return False

def main():
    rclpy.init()
    checker = PhysicsChecker()
    
    print("=" * 60)
    print("PHYSICS CHECK TOOL")
    print("This checks if your robot has proper physics")
    print("INSTRUCTIONS:")
    print("1. Keep this terminal running")
    print("2. Go to your simulation GUI")
    print("3. Try to MANUALLY PUSH/DRAG the robot with your mouse")
    print("4. Watch if the position values change here")
    print("=" * 60)
    
    input("Press Enter to start physics check...")
    
    has_physics = checker.check_movement()
    
    if has_physics:
        print("\n Physics is working! The issue is with command handling.")
    else:
        print("\n Physics issue detected! The robot model needs fixing.")
        print("   The robot might be 'static' or missing physics properties.")
    
    checker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()