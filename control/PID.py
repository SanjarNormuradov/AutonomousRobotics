#!/usr/bin/env python
from __future__ import print_function
import collections
import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.srv import GetMap
from utils import utils

CTRL_TOPIC = '/car/mux/ackermann_cmd_mux/input/navigation'  # Topic to publish control commands


class PID:
  '''Follows a given plan using PID controller of velocity and steering angle.'''
  def __init__(self,
    plan, pose_topic, plan_lookahead, 
    crosstrack_weight, rotation_weight,
    Kp, Ki, Kd, error_buff_length,
    min_speed, max_speed,
    map_service_name
  ):
    '''Initializes PID Controller.

    Args:
      plan (List[np.ndarray]): List of N np.ndarray of shape (3). Path coordinates (x,y,theta) that the robot should follow.
      pose_topic (string): Topic that provides the current robot pose (PoseStamped).
      plan_lookahead (int): Navigate towards the (i + plan_lookahead)-th pose in the plan, if i-th pose is the closest to the robot.
      crosstrack_weight | rotation_weight (float): Cross-track | Rotation error weight in the total error.
      Kp | Ki | Kd (float): Proportional | Integral | Derivative parameters.
      error_buff_length (int): Length of the buffer that stores past error (values, timestamp in sec).
      min_speed | max_speed (float): Min | Max speed at which the robot could travel.
    '''
    print("[PID] Initialization...")
    # Store the passed parameters
    self.plan = plan
    self.plan_lookahead = plan_lookahead
    # Normalize cross-track and rotation weights, if they are not
    self.crosstrack_weight = crosstrack_weight / (crosstrack_weight + rotation_weight)
    self.rotation_weight = rotation_weight / (crosstrack_weight + rotation_weight)
    self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
    self.error_buff = collections.deque(maxlen=error_buff_length)
    self.min_speed = min_speed
    self.max_speed = max_speed

    self.go_forward = False
    rospy.wait_for_service(map_service_name)
    self.map_msg = rospy.ServiceProxy(map_service_name, GetMap)().map
    self.keypoint = utils.map_to_world_xy([2600, 660], self.map_msg.info)

    self.cmd_pub = rospy.Publisher(CTRL_TOPIC, AckermannDriveStamped, queue_size=10)
    self.pose_sub = rospy.Subscriber(pose_topic, PoseStamped, self.pose_callback)
    # Uncomment below if you want to visualize where is the (i + plan_lookahead)-th pose in the plan.
    # self.pla_pub = rospy.Publisher("plan_look_ahead", PoseStamped, queue_size=1)
    print("[PID] Initialization complete")
 
  def pose_callback(self, msg):
    '''Callback for the current pose of the car.

    Args:
        msg (PoseStamped): Current pose of the car.
    '''
    cur_pose = np.array(
      [msg.pose.position.x, msg.pose.position.y, utils.quaternion_to_angle(msg.pose.orientation)]
    )
    success, error = self.compute_error(cur_pose)
    if success:
      print("We have reached the target")
      self.pose_sub.unregister()  # Kill the subscriber to stop entering this funcion next time
      self.speed = 0.0  # Set speed to zero so car stops
      delta = 0.0  # Set steering angle to default
    else:
      delta = self.compute_steering_angle(error[0])

    ads = AckermannDriveStamped()
    ads.header = utils.make_header("/map")
    ads.drive.steering_angle = delta
    # If you need to change the speed depending on the along-track error (error[1]), uncomment below
    ads.drive.speed = self.min_speed + (self.max_speed - self.min_speed) / (1 + np.exp(-5 * (error[1] - 1/2)))
    # If you need to constant speed (self.min_speed)
    # ads.drive.speed = self.min_speed
    self.cmd_pub.publish(ads)

  def compute_error(self, cur_pose):
    '''Computes translation(cross-track and along-track) and rotation errors.

    Args:
      cur_pose (np.ndarray): np.ndarray of shape (3). Current robot pose (x,y,theta).
    
    Returns: 
      Tuple[bool, float]: (True, 0.0), if the end of the plan has been reached. (False, E), otherwise - where E is the computer error.
    '''
    # Find the first point of the plan that is in front of the robot, and remove any other that are behind. 
    # Loop over each point of the plan:
    #   remove it if it's behind the robot, i.e. dot product of the car's heading vector and the vector from the car to the point is negative
    #   break the loop otherwise
    while self.plan:
      dot = (self.plan[0][0] - cur_pose[0]) * np.cos(cur_pose[2]) + (self.plan[0][1] - cur_pose[1]) * np.sin(cur_pose[2])
      if dot > 0:
        break
      self.plan.pop(0)
      
    # If the plan is empty, then we reached the target
    if not self.plan:
      return True, 0.0

    # At this point, we have removed points from the plan that are behind the robot. 
    # Therefore, element 0 is the first points in the plan that is in front of the robot. 
    # To allow the robot to have some amount of 'look ahead', we choose to have the robot head towards the (0 + self.plan_lookahead)-th point
    goal_idx = min(0 + self.plan_lookahead, len(self.plan) - 1)
    
    v = self.plan[goal_idx][:2] - cur_pose[:2]  # Vector from the car to the goal point in the path
    v /= np.linalg.norm(v)  # If not normalized, crosstrack_weight/rotation_weight wouldn't have much effect in case of big vector's norm

    # Compute translation errors:
    ct_err = np.cos(cur_pose[2]) * v[1] - np.sin(cur_pose[2]) * v[0]  # Cross-track (ct_err) and . Both errors <= 1 because of each vector product component <= 1
    at_err = np.cos(cur_pose[2]) * v[0] + np.sin(cur_pose[2]) * v[1]  # Along-track (at_err). If you need to control the speed based on how far the robot is from the given path

    # Rotation error (rot_err) should also be <= 1. Use sin() with angles wrapped between [-np.pi, np.pi)
    rot_angle = (self.plan[goal_idx][2] - cur_pose[2] + np.pi) % (2 * np.pi) - np.pi
    rot_err = np.sin(rot_angle)  # abs(sin) is always <= 0

    # Debugging, ignore
    # print("[PID] plan_length: %d" % len(self.plan))
    # print("[PID] plan_angle: (%.3f rad, %.3f deg), cur_angle: (%.3f rad, %.3f deg)" % (self.plan[goal_idx][2], np.degrees(self.plan[goal_idx][2]), cur_pose[2], np.degrees(cur_pose[2])))
    # print("[PID] CT: %.3f, AT: %.3f, Rot: %.3f, Angle: (%.3f rad, %.3f deg)" % (ct_err, at_err, rot_err, rot_angle, np.degrees(rot_angle)))
    
    # If you need to control the speed depending on the along-track error, uncomment below. Otherwise, leave as it is
    error = (self.crosstrack_weight * ct_err + self.rotation_weight * rot_err, at_err)
    # print("[PID] SteerError: %.3f, SpeedError: %.3f" % (error[0], error[1]))

    # If you need to control the steering angle only depending on cross-track and rotation errors, uncomment below. Otherwise, leave as it is
    # error = self.crosstrack_weight * ct_err + self.rotation_weight * rot_err
    # print("[PID] total_error: %.3f, trans_error: %.3f, rot_error: %.3f, angle_diff: (%.3f rad, %.3f deg)" % (error, ct_err, rot_err, rot_angle, np.degrees(rot_angle)))
    
    # Uncomment below if you need to see the visualization of the plan_lookahead point
    # pla = PoseStamped()
    # pla.header = utils.make_header("map")
    # pla.pose.position.x = self.plan[goal_idx][0]
    # pla.pose.position.y = self.plan[goal_idx][1]
    # pla.pose.orientation = utils.angle_to_quaternion(self.plan[goal_idx][2])
    # self.pla_pub.publish(pla)

    return False, error

  def compute_steering_angle(self, error):
    '''Uses a PID control policy to generate a steering angle from the passed error.
    
    Args:
        error (float): Current error.
    
    Returns:
        float: Steering angle.
    '''
    now = rospy.Time.now().to_sec()
    
    # Compute the derivative error using the passed error, the current time,
    # the most recent error stored in self.error_buff, and the most recent time stored in self.error_buff
    deriv_error = (error - self.error_buff[-1][0]) / (now - self.error_buff[-1][1]) if len(self.error_buff) != 0 else 0.0

    self.error_buff.append((error, now))
    
    # Compute the integral error by applying rectangular integration to the elements
    # of self.error_buff: https://chemicalstatistician.wordpress.com/2014/01/20/rectangular-integration-a-k-a-the-midpoint-rule/
    integ_error = 0
    for i in range(len(self.error_buff) - 1):
      integ_error += (self.error_buff[i + 1][1] - self.error_buff[i][1]) * (self.error_buff[i][0] + self.error_buff[i + 1][0]) / 2

    # Compute the steering angle as the sum of the PID errors
    delta = self.Kp * error + self.Ki * integ_error + self.Kd * deriv_error
    # print("[PID] Steering angle: %f" % delta)
    return delta
