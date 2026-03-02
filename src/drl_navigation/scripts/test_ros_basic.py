# test_ros_basic.py
#!/usr/bin/env python3
try:
    import rclpy
    print("rclpy imported successfully!")
    
    import yaml
    print("yaml imported successfully!")
    
    print("All basic imports working!")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Try: pip install pyyaml lxml defusedxml")