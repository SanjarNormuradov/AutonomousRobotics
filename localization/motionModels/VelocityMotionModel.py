#!/usr/bin/env python
import rospy
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal as Gauss

from threading import Lock
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from vesc_msgs.msg import VescStateStamped

# Set these values and use them in motion_callback
KM_V_NOISE = 0.01 # Kinematic car velocity noise, which is normally distributed with standard deviation
KM_DELTA_NOISE = 0.05 # Kinematic car delta noise, which is normally distributed with standard deviation
KM_X_FIX_NOISE = 0.01 # Kinematic car x position constant noise, which is normally distributed with standard deviation
KM_Y_FIX_NOISE = 0.01 # Kinematic car y position constant noise, which is normally distributed with standard deviation
KM_THETA_FIX_NOISE = 0.02 # Kinematic car theta constant noise, which is normally distributed with standard deviation


class KinematicMotionModel:
  '''Propagates the particles forward based on the velocity and steering angle of the car.'''
  def __init__(self,
      motor_state_topic = "/car/vesc/sensors/core",
      servo_state_topic = "/car/vesc/sensors/servo_position_command",
      speed_to_erpm_offset = 0.0, speed_to_erpm_gain = 4350,
      steer_to_servo_offset = 0.5, steer_to_servo_gain = -1.2135, 
      car_length = 0.33,
      particles = None,
      state_lock = None):
    '''Initializes Kinematic Motion Model.
    
    Args:
      motor_state_topic: str = "/car/vesc/sensors/core",
      servo_state_topic: str = "/car/vesc/sensors/servo_position_command",
      speed_to_erpm_offset: float = 0.0, speed_to_erpm_gain: int = 4350,
      steer_to_servo_offset: float = 0.5, steer_to_servo_gain: float = -1.2135, 
      car_length: float = 0.33,
      particles: np.ndarray = None,
      state_lock: Lock = None

      motor_state_topic: The topic containing motor state information
      servo_state_topic: The topic containing servo state information    
      speed_to_erpm_offset: Offset conversion param from rpm to speed
      speed_to_erpm_gain: Gain conversion param from rpm to speed
      steering_angle_to_servo_offset: Offset conversion param from servo position to steering angle
      steering_angle_to_servo_gain: Gain conversion param from servo position to steering angle 
      car_length: The length of the car
      particles: The particles to propagate forward
      state_lock: Controls access to particles    
    '''
    print("[MotionModel] Initialization...")
    self.last_servo_cmd = None # The most recent servo command
    self.last_vesc_stamp = None # The time stamp from the previous vesc_msg
    self.particles = particles
    self.num_particles = particles.shape[0]
    self.SPEED_TO_ERPM_OFFSET = speed_to_erpm_offset # Offset conversion param from rpm to speed
    self.SPEED_TO_ERPM_GAIN   = speed_to_erpm_gain # Gain conversion param from rpm to speed  
    self.STEER_TO_SERVO_OFFSET = steer_to_servo_offset # Offset conversion param from servo position to steering angle
    self.STEER_TO_SERVO_GAIN = steer_to_servo_gain # Gain conversion param from servo position to steering angle
    self.CAR_LENGTH = car_length # The length of the car
    self.incr = np.zeros((self.num_particles, 3))
    self.theta = np.zeros((self.num_particles, 2))
    self.theta_lmt = np.zeros(self.num_particles)
    self.delta = np.zeros(self.num_particles)
    self.speed = np.zeros(self.num_particles)

    # This just ensures that two different threads are not changing the particles
    # array at the same time. You should not have to deal with this.
    if state_lock is None:
      self.state_lock = Lock()
    else:
      self.state_lock = state_lock
      
    # This subscriber just caches the most recent servo position command
    self.servo_pos_sub  = rospy.Subscriber(servo_state_topic, Float64, self.servo_callback, queue_size=1)
    # Subscribe to the state of the vesc
    self.motion_sub = rospy.Subscriber(motor_state_topic, VescStateStamped, self.motion_callback, queue_size=1)
    print("[MotionModel] Initialization complete")                                     

  def servo_callback(self, msg):
    '''Caches the most recent servo command.
    Args:
      msg: A std_msgs/Float64 message
    '''
    self.last_servo_cmd = msg.data # Update servo command

  def motion_callback(self, msg):
    '''Converts messages to controls and applies the kinematic car model to the particles.
    Args:
      msg: a message
    '''
    # print("[MotionModel] Future pose estimation...")
    self.state_lock.acquire()
    if self.last_servo_cmd is None:
      self.state_lock.release()
      return

    if self.last_vesc_stamp is None:
      self.last_vesc_stamp = msg.header.stamp
      self.state_lock.release()
      return
    
    # Convert raw msgs to controls
    # Note that control_val = (raw_msg_val - offset_param) / gain_param
    self.speed[:] = Gauss((msg.state.speed - self.SPEED_TO_ERPM_OFFSET) / self.SPEED_TO_ERPM_GAIN, KM_V_NOISE, self.num_particles)
    self.delta[:] = Gauss((self.last_servo_cmd - self.STEER_TO_SERVO_OFFSET) / self.STEER_TO_SERVO_GAIN, KM_DELTA_NOISE, self.num_particles)
    beta = np.arctan(np.tan(self.delta) / 2)
    # Propagate particles forward in place
      # Sample control noise and add to nominal control
      # Make sure different control noise is sampled for each particle
      # Propagate particles through kinematic model with noisy controls
      # Sample model noise for each particle
      # Limit particle theta to be between -pi and pi
      # Vectorize your computations as much as possible
      # All updates to self.particles should be in-place
    self.theta[:,0] = self.particles[:,2]
    dt = msg.header.stamp.to_sec() - self.last_vesc_stamp.to_sec()
    self.theta_lmt[:] = self.theta[:,0] + self.speed / self.CAR_LENGTH * np.sin(2 * beta) * dt
    self.last_vesc_stamp = msg.header.stamp
    for idx, theta_idx in enumerate(self.theta_lmt):
      if np.abs(theta_idx) > np.pi:
        self.theta_lmt[idx] = -np.sign(theta_idx) * (2*np.pi - np.abs(theta_idx))
    self.theta[:,1] = self.theta_lmt[:]

    self.incr[:,0] = Gauss(self.particles[:,0] + self.CAR_LENGTH / np.sin(2 * beta) * (np.sin(self.theta[:,1]) - np.sin(self.theta[:,0])), KM_X_FIX_NOISE)
    self.incr[:,1] = Gauss(self.particles[:,1] + self.CAR_LENGTH / np.sin(2 * beta) * (-np.cos(self.theta[:,1]) + np.cos(self.theta[:,0])), KM_Y_FIX_NOISE)
    self.incr[:,2] = Gauss(self.theta[:,1], KM_THETA_FIX_NOISE) 

    self.particles[:,:] = self.incr[:,:]

    self.last_vesc_stamp = msg.header.stamp    
    self.state_lock.release()
    # print "[MotionModel] Future poses estimated" 
