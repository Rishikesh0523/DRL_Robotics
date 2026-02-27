"""
Configuration parameters for SAC training
"""

class TrainingConfig:
    # Training parameters
    TOTAL_TIMESTEPS = 100_000
    BATCH_SIZE = 128
    BUFFER_SIZE = 50000
    LEARNING_STARTS = 1000
    
    # SAC parameters
    LEARNING_RATE = 3e-4
    DISCOUNT = 0.99
    TAU = 0.005
    
    # Network architecture
    HIDDEN_DIM = 256
    HIDDEN_DEPTH = 2
    
    # Environment parameters
    MAX_EPISODE_STEPS = 300
    GOAL_TOLERANCE = 0.5
    COLLISION_THRESHOLD = 0.25
    MAX_LINEAR_SPEED = 0.3
    MAX_ANGULAR_SPEED = 0.8
    
    # Directories
    LOG_DIR = "./sac_training_logs/"
    MODEL_DIR = "./sac_models/"

class RobotConfig:
    # LiDAR parameters (from your actual data)
    LIDAR_MAX_RANGE = 10.0
    LIDAR_MIN_RANGE = 0.1
    LIDAR_BINS = 20
    
    # State dimensions
    STATE_DIM = 25  # 20 LiDAR bins + 5 features
    ACTION_DIM = 2  # [linear_vel, angular_vel]
    
    # Map boundaries (adjust based on your world)
    MAP_LIMITS = (-8.0, 8.0)
    SAFETY_MARGIN = 1.0
    
    # Goal parameters
    MIN_GOAL_DISTANCE = 2.0
    MAX_GOAL_DISTANCE = 6.0