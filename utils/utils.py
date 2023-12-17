#!/usr/bin/env python
from __future__ import print_function
import rospy
import numpy as np
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Quaternion, Point32
import tf


def angle_to_quaternion(angle):
    '''Convert yaw angle in radians into a quaternion message.

    Args:
        angle (float): Yaw angle.

    Returns: 
        Quaternion: Equivalent quaternion message.
    '''
    return Quaternion(*tf.transformations.quaternion_from_euler(0, 0, angle))


def quaternion_to_angle(q):
    '''Convert a quaternion message into a yaw angle in radians.
    Args:
        q (Quaternion): Quaternion message.
    
    Returns:
        float: Equivalent yaw in radians.
    '''
    x, y, z, w = q.x, q.y, q.z, q.w
    roll, pitch, yaw = tf.transformations.euler_from_quaternion((x, y, z, w))
    return yaw


def rotation_matrix(theta):
    '''Constructs a rotation matrix from a given angle in radians.

    Args:
        theta (float): Angle in radians.

    Returns:
        numpy.ndarray: Equivalent numpy.ndarray of shape (2, 2). Rotation matrix.
    '''
    c, s = np.cos(theta), np.sin(theta)
    return np.matrix([[c, -s], [s, c]])


def particle_to_pose(particle):
    '''Converts a particle to a pose message.
    
    Args:
        particle (Tuple[float, float, float]): Particle to convert as a list of [x,y,theta]
    
    Returns:
        Pose: An equivalent pose message.
    '''
    pose = Pose()
    pose.position.x = particle[0]
    pose.position.y = particle[1]
    pose.orientation = angle_to_quaternion(particle[2])
    return pose


def particles_to_poses(particles):
    '''Converts a list of particles to a list of pose messages.

    Args:
        particles (List[Tuple[float, float, float]]): A list of particles, where each element is itself a list of the form [x,y,theta]
    
    Returns:
        List[Pose]: List of equivalent pose messages.
    '''
    return list(map(particle_to_pose, particles))


def make_header(frame_id, stamp=None):
    '''Creates a header with the given frame_id and stamp. 
    Default value of stamp is None, which results in a stamp denoting the time at which this function was called.

    Args:
        frame_id (string): Desired coordinate frame.
        stamp (rospy.Time): Desired time stamp.
    
    Returns:
        Header: Resulting header.
    '''
    if stamp == None:
        stamp = rospy.Time.now()
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return header


def point(npt):
    '''Converts a list with coordinates into a point message.
    
    Args:
        npt (Tuple[float, float]): Tuple of coordinates (x,y).

    Returns:
        Point32: Point message.
    '''
    pt = Point32()
    pt.x = npt[0]
    pt.y = npt[1]
    return pt


def points(arr):
    '''Converts a list of coordinates into a list of equivalent point messages.

    Args:
        arr (List[Tuple[float, float]]): List of coordinates (x,y).

    Returns:
        List[Point32]: List of point messages.
    '''
    return list(map(point, arr))


def map_to_world(poses, map_info, mode=False):
    '''Convert (in-place) array of coordinates from map-frame to world-frame.

    Args:
        poses (numpy.ndarray): numpy.ndarray of shape (n, 3). Coordinates (x,y,theta) in the map-frame.
        map_info (nav_msgs.msg.MapMetaData): Info about the map (returned by nav_msgs/GetMap service).
    '''
    scale = map_info.resolution
    angle = quaternion_to_angle(map_info.origin.orientation)

    # Rotation matrix
    c, s = np.cos(angle), np.sin(angle)
    temp = np.copy(poses[:,0])
    # Rotation
    poses[:,0] = c * poses[:,0] - s * poses[:,1]
    poses[:,1] = c * poses[:,1] + s * temp
    # Scale
    poses[:,:2] *= float(scale)
    # Offset origin (translation, rotation)
    poses[:,0] += map_info.origin.position.x
    poses[:,1] += map_info.origin.position.y
    poses[:,2] += angle

 
def world_to_map(poses, map_info):
    '''Convert (in-place) array of coordinates from world-frame to map-frame.

    Args:
        pose (numpy.ndarray): numpy.ndarray of shape (n, 3). Coordinates (x,y,theta) in the world-frame.
        map_info (nav_msgs.msg.MapMetaData): Info about the map (returned by nav_msgs/GetMap service).
    '''
    scale = map_info.resolution
    angle = quaternion_to_angle(map_info.origin.orientation)

    # Offset origin (translation, rotation)
    poses[:,0] -= map_info.origin.position.x
    poses[:,1] -= map_info.origin.position.y
    poses[:,2] -= angle
    # Scale
    poses[:,:2] /= float(scale)
    # Rotation matrix
    c, s = np.cos(-angle), np.sin(-angle)
    temp = np.copy(poses[:,0])
    # Rotation
    poses[:,0] = c * poses[:,0] - s * poses[:,1]
    poses[:,1] = c * poses[:,1] + s * temp


def map_to_world_xy(pose, map_info, inv=True):
    '''Convert robot coordinates from map-frame to world-frame.

    Args:
        pose (numpy.ndarray, Tuple[float, float]): numpy.ndarray of shape (2) or tuple of 2 elements. Coordinates (x,y,theta) in the map-frame.
        map_info (nav_msgs.msg.MapMetaData): Info about the map (returned by nav_msgs/GetMap service).
        inv (bool): Invert y coordinate.
    
    Returns:
        List[float, float]: Coordinates in the world-frame.
    '''
    scale = map_info.resolution
    angle = quaternion_to_angle(map_info.origin.orientation)
    config = np.array([pose[0], map_info.height - pose[1]]) if inv else np.array([pose[0], pose[1]])

    # Rotation matrix
    c, s = np.cos(angle), np.sin(angle)
    temp = np.copy(pose[0])
    # Rotation
    config[0] = c * config[0] - s * config[1]
    config[1] = c * config[1] + s * temp
    # Scale
    config *= float(scale)
    # Offset origin (translation)
    config[0] += map_info.origin.position.x
    config[1] += map_info.origin.position.y

    return config


def world_to_map_xy(pose, map_info):
    '''Convert robot coordinates from world-frame to map-frame.

    Args:
        pose (numpy.ndarray, Tuple[float, float]): numpy.ndarray of shape (2) or tuple of 2 elements. Coordinates (x,y,theta) in the world-frame.
        map_info (nav_msgs.msg.MapMetaData): Info about the map (returned by nav_msgs/GetMap service).
    
    Returns:
        List[float, float]: Coordinates in the map-frame.
    '''
    scale = map_info.resolution
    angle = quaternion_to_angle(map_info.origin.orientation)
    if isinstance(pose, list):
        assert len(pose) == 2, "Pose should contain 2 elements"
        config = np.array([pose])
    elif isinstance(pose, np.ndarray):
        config = np.copy(pose)
    else:
        raise ValueError(f"Unsupported type {type(pose)} for pose")

    # Offset origin (translation, rotation)
    config[0] -= map_info.origin.position.x
    config[1] -= map_info.origin.position.y
    # Scale
    config /= float(scale)
    # Rotation matrix
    c, s = np.cos(-angle), np.sin(-angle)
    temp = np.copy(pose[0])
    # Rotation
    config[0] = c * config[0] - s * config[1]
    config[1] = c * config[1] + s * temp

    return config
