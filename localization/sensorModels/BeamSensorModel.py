#!/usr/bin/env python
import rospy
import math
import range_libc
import numpy as np
import matplotlib.pyplot as plt

from threading import Lock
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid


MAP_TOPIC = 'static_map'
THETA_DISCRETIZATION = 112  # Discretization of scanning angle
INV_SQUASH_FACTOR = 0.2  # Factor for helping the weight distribution to be less peaked


class SensorModel:
  def __init__(self, 
      scan_topic = "/car/scan",
      laser_ray_step = 20, max_range_meters = 13.0,
      Z_HIT = 0.60, Z_RAND = 0.10,
      Z_MAX = 0.01, Z_SHORT = 0.29,
      SIGMA_HIT = 10.5, LAMBDA_SHORT = 0.005,
      map_msg = None,
      particles = None, weights = None,
      state_lock = None):
    '''Initializes the Sensor Model.
    
    Args:
      scan_topic: str = "/car/scan",
      laser_ray_step: int = 20, max_range_meters: float = 13.0,
      Z_HIT: float = 0.60, Z_RAND: float = 0.10,
      Z_MAX: float = 0.01, Z_SHORT: float = 0.29,
      SIGMA_HIT: float = 10.5, LAMBDA_SHORT: float = 0.005,
      map_msg: OccupancyGrid = None,
      particles: np.ndarray = None, weights: np.ndarray = None,
      state_lock: Lock = None

      scan_topic: The topic containing laser scans
      laser_ray_step: Step for downsampling laser scans
      exclude_max_range_rays: Whether to exclude rays that are beyond the max range
      max_range_meters: The max range of the laser
      map_msg: A nav_msgs/MapMetaData msg containing the map to use
      particles: The particles to be weighted
      weights: The weights of the particles
      state_lock: Used to control access to particles and weights
    '''
    print("[SensorModel] Initialization...")
    if state_lock is None:
      self.state_lock = Lock()
    else:
      self.state_lock = state_lock

    self.particles = particles
    self.weights = weights
    self.last_laser = None

    self.LASER_RAY_STEP = laser_ray_step  # Step for downsampling laser scans
    self.MAX_RANGE_METERS = max_range_meters  # The max range of the laser
    # Sensor Model Intrinsic Parameters. Z_SHORT + Z_MAX + Z_RAND + Z_HIT = 1
    self.Z_HIT = Z_HIT
    self.Z_RAND = Z_RAND      
    self.Z_SHORT = Z_SHORT
    self.Z_MAX = Z_MAX
    self.SIGMA_HIT = SIGMA_HIT
    self.LAMBDA_SHORT = LAMBDA_SHORT    

    oMap = range_libc.PyOMap(map_msg)  # A version of the map that range_libc can understand
    max_range_px = int(self.MAX_RANGE_METERS / map_msg.info.resolution)  # The max range in pixels of the laser
    self.range_method = range_libc.PyCDDTCast(oMap, max_range_px, THETA_DISCRETIZATION)  # The range method that will be used for ray casting
    # self.range_method = range_libc.PyRayMarchingGPU(oMap, max_range_px) # The range method that will be used for ray casting
    self.range_method.set_sensor_model(self.precompute_sensor_model(max_range_px))  # Load the sensor model expressed as a table
    self.queries = None  # Do not modify this variable
    self.ranges = None  # Do not modify this variable
    self.downsampled_ranges = None  # The ranges of the downsampled rays
    self.downsampled_angles = None  # The angles of the downsampled rays
    self.do_resample = False  # Set so that outside code can know that it's time to resample
    self.is_rays_downsampled = False  # Downsample laser rays once at the first message instance

    # Subscribe to laser scans
    self.laser_sub = rospy.Subscriber(scan_topic, LaserScan, self.lidar_callback, queue_size=1)
    print("[SensorModel] Initialization complete")

  def lidar_callback(self, msg):
    '''Downsamples laser measurements and applies Sensor Model.
    Arg:
        msg: A sensor_msgs/LaserScan
    '''    
    self.state_lock.acquire()

    """
    Compute the observation obs
    obs is a a two element tuple
    obs[0] is the downsampled ranges
    obs[1] is the downsampled angles
    Note it should be the case that obs[0].shape[0] == obs[1].shape[0]
    Each element of obs must be a numpy array of type np.float32
    Use self.LASER_RAY_STEP as the downsampling step
    Keep efficiency in mind, including by caching certain things that won't change across future iterations of this callback
      and vectorizing computations as much as possible
    Set all range measurements that are NAN or 0.0 to self.MAX_RANGE_METERS
    You may choose to use self.downsampled_ranges and self.downsampled_angles here
    """
    if self.downsampled_angles == None:
      self.downsample_size = len(msg.ranges) // self.LASER_RAY_STEP
      self.downsampled_angles = []
      for angle_idx in xrange(self.downsample_size):
        self.downsampled_angles.append(msg.angle_min + msg.angle_increment * angle_idx * self.LASER_RAY_STEP)
    
    self.downsampled_ranges = []
    for idx in xrange(self.downsample_size):
      obs_range = msg.ranges[idx * self.LASER_RAY_STEP]
      if np.isnan(obs_range) or obs_range == 0.0:
        self.downsampled_ranges.append(self.MAX_RANGE_METERS)
      else:
        self.downsampled_ranges.append(obs_range)

    obs = np.zeros((2, self.downsample_size), dtype=np.float32)
    obs[0,:] = np.array(self.downsampled_ranges, dtype=np.float32)
    obs[1,:] = np.array(self.downsampled_angles, dtype=np.float32)
    # print "[SensorModel] Weights: %s" % str(self.weights)
    self.apply_sensor_model(self.particles, obs, self.weights)
    # print "[SensorModel] Weights: %s" % str(self.weights)
    self.weights /= np.sum(self.weights)

    self.last_laser = msg
    self.do_resample = True
    self.state_lock.release()

  def precompute_sensor_model(self, max_range_px):
    '''Compute table enumerating the probability of observing a measurement given the expected measurement.
      Element (r,d) of the table is the probability of observing (true) measurement r (in pixels)
      when the expected (simulated) measurement is d (in pixels)
      
      Args:
        max_range_px: int, The maximum range in pixels
      
      Returns:
        np.ndarray: the table (which is a numpy array with dimensions [max_range_px+1, max_range_px+1]) 
    '''
    print("[SensorModel] Sensor model generation as table...")
    # import pdb
    table_width = int(max_range_px) + 1
    sensor_model_table = np.zeros((table_width, table_width))

    # Populate sensor_model_table according to the laser beam model specified
    # in CH 6.3 of Probabilistic Robotics
    # Note: no need to use any functions from utils.py to compute between world
    #       and map coordinates here
    variance = np.power(self.SIGMA_HIT, 2)
    norm = 1
    # ideal - noise-free sensor measurements
    # real - noisy sensor measurements
    for ideal in range(table_width):
      for real in range(table_width):
        # Point-mass Distribution to account 'Max-range Measurments' such as specular (mirror) reflections (sonars) and black, light-absorbing, objects or measuring in bright-sunlight (lasers)
        p_max = float(real == int(max_range_px))
        # Uniform Distribution to account 'Unexplainable Measurements' such as cross-talk between different sensors or phantom readings bouncing off walls (sonars)
        p_rand = float(ideal < int(max_range_px)) * 1 / int(max_range_px)
        if ideal == 0:
          p_short = 0.0
          norm = math.erf(float(max_range_px) / (math.sqrt(2) * self.SIGMA_HIT)) / 2
        else:
          norm = 1 / -np.expm1(-self.LAMBDA_SHORT * ideal) 
          # Exponential Distribution to account 'Unexpected Objects' such as people not contained in the static map
          p_short = float(real <= ideal) * norm * self.LAMBDA_SHORT * np.exp(-self.LAMBDA_SHORT * real)
        # Gaussian Distribution to account 'Measurement Noise'
        p_hit = float(ideal <= int(max_range_px)) * 1 / (np.sqrt(2 * np.pi * variance)) * np.exp(-1 / 2 * np.power(real - ideal, 2) / variance)

        sensor_model_table[real, ideal] = self.Z_HIT * p_hit + self.Z_SHORT * p_short + self.Z_MAX * p_max + self.Z_RAND * p_rand
    # Normalization over all probabilities suchas as below is not correct
    sensor_model_table /= sensor_model_table.sum(axis=0)
    return sensor_model_table

  def apply_sensor_model(self, proposal_dist, obs, weights):
    '''Updates the particle weights in-place based on the observed laser scan.
      
      Args:
        proposal_dist: np.ndarray, The particles
        obs: np.ndarray, The most recent observation
        weights: np.ndarray, The weights of each particle
    '''
    obs_ranges = obs[0]
    obs_angles = obs[1]
    num_rays = self.downsample_size

    # Only allocate buffers once to avoid slowness
    if not isinstance(self.queries, np.ndarray):
      self.queries = np.zeros((proposal_dist.shape[0], 3), dtype=np.float32)
      self.ranges = np.zeros(num_rays * proposal_dist.shape[0], dtype=np.float32)

    self.queries[:,:] = proposal_dist[:,:]

    # Raycasting to get expected measurements
    self.range_method.calc_range_repeat_angles(self.queries, obs_angles, self.ranges)

    # Evaluate the sensor model
    self.range_method.eval_sensor_model(obs_ranges, self.ranges, weights, num_rays, proposal_dist.shape[0])

    # Squash weights to prevent too much peakiness
    np.power(weights, INV_SQUASH_FACTOR, weights)
