#!/usr/bin/env python
import rospy
import numpy as np

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from utils import utils

CTRL_TOPIC = '/car/mux/ackermann_cmd_mux/input/navigation'  # Topic to publish controls
MAP_TOPIC = '/static_map'  # Service topic that provides the map
VIZ_TOPIC = '/mpcontroller/rollouts'  # NOTE THAT THIS IS ONLY NECESSARY FOR TRAJECTORY ROLLOUT VISUALIZATION

REWARD = 10 # The reward to apply when a point in a rollout is close to a point in the plan
                    

class MPC:
  '''Follows a given plan using PID controller of velocity and steering angle.'''
  def __init__(self, 
      plan, pose_topic, 
      min_speed, max_speed,
      min_delta, max_delta, 
      traj_nums, dt, T, compute_time,
      car_length, car_width,
      visual_sim=True
  ):
    '''Initializes MPC Controller.
    
    Args:
      min_speed | max_speed (float): Min | Max speed at which the robot could travel.
      min_delta | max_delta (float): Min | Max allowed steering angle (radians).
      traj_nums (int): Number of trajectory rollouts.
      dt (float): Amount of time step to apply control.
      T (int): Number of points along each trajectory to compute the corresponding cost.
      compute_time (float): Amount of time (in seconds) allowed to compute the cost along each trajectory.
      car_length | car_width (float): Car length | width.
      visual_sim (bool): Visualize trajectory rollout in RViz.
    '''
    # Store the passed parameters
    print("[MPC] Initialization...")
    self.plan = plan
    self.T = T
    self.compute_time = compute_time
    self.car_length = car_length
    self.car_width = car_width
    self.visual_sim = visual_sim
    self.success = False
    self.first_indx = 0

    print("[MPC]  Rollouts, steering angles, and speed computation...")
    self.rollouts, self.deltas, self.speeds = self.generate_mpc_rollouts(
      min_speed, max_speed, min_delta, max_delta, 
      traj_nums, dt, T, car_length
    )
    print("[MPC] Rollouts and deltas computation complete")
    print("[MPC] Initialization complete")
    self.cmd_pub = rospy.Publisher(CTRL_TOPIC, AckermannDriveStamped, queue_size=10) # Publisher to send control commands
    if self.visual_sim:
      # NOTE THAT THIS VISUALIZATION WORKS ONLY IN SIMULATION.
      self.viz_pub = rospy.Publisher(VIZ_TOPIC, PoseArray, queue_size=10) # Publisher to vizualize trajectories
    input("[MPC] Press Enter to start following the path with MPC...\n")
    self.pose_sub = rospy.Subscriber(pose_topic, PoseStamped, self.pose_callback) # Subscriber to the current robot pose

  def generate_mpc_rollouts(self,
      min_speed, max_speed,
      min_delta, max_delta,
      traj_nums, dt, T,
      car_length
  ):
    '''Generate trajectory rollouts and deltas.

    Args:
      min_speed | max_speed (float): Min | Max speed at which the car could travel. More straight line trajectory is, higher speed.
      min_delta | max_delta (float): Min | Max allowed steering angle (radians).
      traj_nums (int): Number of trajectory rollouts.
      dt (float): Amount of time step to apply control.
      T (int): Number of points along each trajectory to compute the corresponding cost.
      car_length | car_width (float): Car length | width.

    Returns:
      rollouts (np.ndarray): np.ndarray of shape (N,T,3). N trajectories, each containing T poses (x,y,theta) of the robot at time t+1.
      deltas (np.ndarray): np.ndarray of shape (N). N steering angles where n-th angle would result in the n-th trajectory in rollouts.
      speeds (np.ndarray): np.ndarray of shape (N). N speeds where n-th speed would be applied along the n-th trajectory in rollouts.
    '''
    delta_step = (max_delta - min_delta - 0.001) / (traj_nums - 1)
    deltas = np.arange(min_delta, max_delta, delta_step)
    N = deltas.shape[0]
    speeds = np.zeros_like(deltas, dtype=np.float)
    rollouts = np.zeros((N, T, 3), dtype=np.float)
    init_pose = np.array([0.0, 0.0, 0.0], dtype=np.float)
    speed_step = (N - 1) / 2
    for i in range(N):
      speeds[i] = max_speed - ((max_speed - min_speed) / speed_step) * abs(speed_step - i)
      controls = np.zeros((T, 3), dtype=np.float)
      controls[:,0] = speeds[i]
      controls[:,1] = deltas[i]
      controls[:,2] = min_speed / speeds[i] * dt
      rollouts[i,:,:] = self.generate_rollout(init_pose, controls, car_length)

    return rollouts, deltas, speeds

  def generate_rollout(self, init_pose, controls, car_length):
    '''Repeatedly apply Ackermann kinematic model to produce a trajectory for the robot.

    Args:
      init_pose (np.ndarray): Initial pose of the robot (x,y,theta).
      controls (np.ndarray): np.ndarray of shape (T,3). Control commands for each trajectory point (v,delta,dt).
      car_length (float): Car length.

    Returns:
      np.ndarray of shape (T,3): Trajectory rollout (x,y,theta), where each element is the hypothetical robot's pose along this trajectory.
    '''
    rollout_mat = np.zeros((controls.shape[0], 3), dtype=np.float)
    for pose_id, ctrl_cmd in enumerate(controls):
      current_pose = init_pose.copy()
      if pose_id > 0:
        current_pose += rollout_mat[pose_id - 1]
      rollout_mat[pose_id] = self.kinematic_model_step(current_pose, ctrl_cmd, car_length)

    return rollout_mat

  def kinematic_model_step(self, current_pose, control, car_length):
    '''Apply Ackermann kinematic model to a given pose using given control.

    Args:
      pose (np.ndarray): Current robot pose (x,y,theta).
      control (np.ndarray): Control command to be applied (v,delta,dt).
      car_length (float): Car length.

    Returns:
      np.ndarray: Resulting robot pose.
    '''
    next_pose = current_pose.copy()
    v, delta, dt = control
    theta = next_pose[2]

    next_pose[0] += v * np.cos(theta) * dt
    next_pose[1] += v * np.sin(theta) * dt
    next_pose[2] += v * np.tan(delta) * dt / car_length

    return next_pose

  def pose_callback(self, msg):
    '''Computes steering angle and speed in response to the received laser scan.
    Uses approximately self.compute_time amount of time to compute the control command.
    If self.visual_sim == True, vizualizes the last pose (transformed to the world-frame) of each rollout to prevent lagginess.
    
    Args:
      msg (PoseStamped): Current robot pose. 
    '''
    start = rospy.Time.now().to_sec()
    cur_pose = np.array(
      [msg.pose.position.x, msg.pose.position.y, utils.quaternion_to_angle(msg.pose.orientation)]
    )
    # Find index of the first point in the plan that is in front of the robot,
    # DO NOT REMOVE points behind the robot as in PID controller, because we compute cost for each trajectory
    # Loop over the plan (starting from the index that was found in the previous step) 
    # For each point in the plan:
    #   If the point is behind the robot, increase the index
    #     Perform a coordinate transformation to determine if the point is in front or behind the robot
    #   If the point is in front of the robot, break out of the loop
    while self.first_indx < len(self.plan):
      # print("[MPC] First front point index: %d" % self.first_indx
      # Compute inner product of car's heading vector (cos(theta), sin(theta)) and pointing vector (plan_x[0] - car_x, plan_y[0] - car_y)
      pntVect = self.plan[self.first_indx][0:2] - cur_pose[0:2] 
      pntVectNorm = np.sqrt(np.power(pntVect, 2).sum())
      pntVect /= pntVectNorm
      vect_inn_prod = pntVect[0] * np.cos(cur_pose[2]) + pntVect[1] * np.sin(cur_pose[2])
      # If this inner product is: negative, then car passed by the 1st element of the plan; zero, then car is starting its movement 
      if vect_inn_prod <= 0:
        # print("[MPC] Plan first front point is behind"
        self.first_indx += 1
      else: break
    # Check if the index is over the plan's last index. If so, stop movement
    if self.first_indx == len(self.plan):
      # print("[MPC] Target is achieved!"
      self.success = True
      self.pose_sub.unregister() # Kill the subscriber

    print("[MPC] First index: %d" % self.first_indx)
    if not self.success:
      # N-dimensional matrix that should be populated with the costs of each trajectory up to time t <= T
      delta_costs = np.zeros(self.deltas.shape[0], dtype=np.float) 
      traj_depth = 0
      # Evaluate cost of each trajectory. Each iteration of the loop should calculate
      # the cost of each trajectory at time t = traj_depth and add those costs to delta_costs as appropriate
      while (rospy.Time.now().to_sec() - start <= self.compute_time) and (traj_depth < self.T):
        for traj_num, delta in enumerate(self.deltas):
          delta_costs[traj_num] += self.compute_cost(delta, traj_depth, self.rollouts[traj_num, traj_depth], cur_pose)
        traj_depth += 1
      print("[MPC] Trajectory depth: %d" % traj_depth)
      # Find index of delta that has the smallest cost and execute it by publishing
      min_cost_delta_idx = np.argmin(delta_costs, axis=0)
      print("[MPC] Delta_costs: %s" % str(delta_costs))
      print("[MPC] Min cost trajectory index: %d" % min_cost_delta_idx)

    # ads.header.frame_id = '/laser_link'
    # Setup the control message
    ads = AckermannDriveStamped()
    ads.header = utils.make_header("/map")
    ads.drive.steering_angle = 0.0 if self.success else self.deltas[min_cost_delta_idx]
    ads.drive.speed = 0.0 if self.success else self.speeds[min_cost_delta_idx]
    # Send the control message
    self.cmd_pub.publish(ads)

    if self.visual_sim:
      # Create the PoseArray to publish. Will contain N poses, where the n-th pose represents the last pose in the n-th trajectory
      pose_array = PoseArray()
      pose_array.header = utils.make_header("/map")
      # Transform the last pose of each trajectory to be w.r.t the world and insert into the pose array
      traj_pose = Pose()
      for trj_num, trj in enumerate(self.rollouts):
        vect = np.array([[trj[-1, 0]], [trj[-1, 1]]]) 
        theta_point = trj[-1, 2]
        theta_car = cur_pose[2]
        traj_pose.position.x = utils.rotation_matrix(theta_car).dot(vect)[0] + cur_pose[0]
        traj_pose.position.y = utils.rotation_matrix(theta_car).dot(vect)[1] + cur_pose[1]
        traj_pose.orientation = utils.angle_to_quaternion(theta_point + theta_car)
        pose_array.poses.append(traj_pose)
      self.viz_pub.publish(pose_array)


  '''
  Compute the cost of one step in the trajectory. It should penalize the magnitude
  of the steering angle. It should also heavily penalize crashing into an object
  (as determined by the laser scans)
    delta: The steering angle that corresponds to this trajectory
    rollout_pose: The pose in the trajectory 

  '''  
  def compute_cost(self, delta, traj_depth, rollout_pose, cur_pose):
    # NOTE THAT NO COORDINATE TRANSFORMS ARE NECESSARY INSIDE OF THIS FUNCTION

    # Computation without car's physical dimensions, because A* planner accounted the dimensions during the path generation
    # Initialize the cost to be the magnitude of delta
    cost = np.abs(delta)

    # Compute the cost of following each trajectory, 
    # i.e. inner product of two vectors directed from current robot's pose to:
    #     1. trajectory point with traj_depth
    #     2. plan point with index (self.first_indx + traj_depth)
    planVect = self.plan[self.first_indx + traj_depth][:2] - cur_pose[:2]
    planVectNorm = np.sqrt(np.power(planVect, 2).sum())
    planVect /= planVectNorm

    trajVect = rollout_pose[:2] - cur_pose[:2]
    trajVectNorm = np.sqrt(np.power(trajVect, 2).sum())
    trajVect /= trajVectNorm
      
    # print("[MPC] Cost computation..."
    cost = -(trajVect[0] * planVect[0] + trajVect[1] * planVect[1]) * REWARD
    # Return the resulting cost
    return cost
