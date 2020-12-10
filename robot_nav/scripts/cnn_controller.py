#!/usr/bin/env python  

import utils
import sys
import numpy as np
import tensorflow
import rospy
import tf
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PolygonStamped
from sensor_msgs.msg import LaserScan

tensorflow.logging.set_verbosity(tensorflow.logging.INFO)


def scan_callback(scan):
    global latest_scan
    latest_scan = scan.ranges


def goal_callback(pose):
    global latest_goal, found_goal
    latest_goal = pose
    found_goal = False


def pose_callback(poseArray):
    # whenever the robot's pose has changed, we publish the updated footprint polygon
    polygon = PolygonStamped()
    polygon.header.frame_id = '/map'
    for coords in footprint:
        x, y = coords
        point = PointStamped()
        point.header.frame_id = '/base_link'
        point.point.x = x
        point.point.y = y
        new_point = listener.transformPoint('/map', point)
        polygon.polygon.points.append(new_point.point)
    footprint_publisher.publish(polygon)


def publish_move():
    global latest_goal, found_goal
    if latest_goal is None or latest_scan is None or found_goal is True:
        return

    try:
        latest_goal.header.stamp = rospy.get_rostime()
        robot_goal = listener.transformPose('/base_link', latest_goal)
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        return
    
    # robot_goal is PoseStamped in robot reference frame
    qtemp = robot_goal.pose.orientation
    q = [qtemp.x, qtemp.y, qtemp.z, qtemp.w]
    euler_angles = euler_from_quaternion(q)
    heading = euler_angles[2]
    
    (r, phi) = utils.cart_to_pol(robot_goal.pose.position.x, robot_goal.pose.position.y)
    target = (r, phi, heading)
    
    # insert latest_scan and target to CNN
    # get back twist bits trans and rot
    pred = predictor({'laser': np.asarray(latest_scan), 'goal': np.asarray(target)})
    cmd_vel = pred['cmd_vel']
    trans = cmd_vel[0][0]
    rot = cmd_vel[0][1]

    if trans < 0.01 and abs(rot) < 0.03:
        print "target is at r: " + str(round(r, 2)) + ", phi: " + str(round(phi, 2))
        print "I think I'm at the target!"
        found_goal = True
        return

    twist = Twist()
    twist.linear.x = trans
    twist.angular.z = rot
    
    # print "cnn_controller: target is at r: " + str(round(r, 2)) + ", phi: " + str(round(phi, 2))
    # print "cnn_controller: publishing move: " + str(round(trans, 2)) + ", rot: " + str(round(rot, 2))
    move_publisher.publish(twist)


if __name__ == '__main__':
    rospy.init_node('cnn_controller')    
    
    found_goal = False
    latest_goal = None

    model_dir = sys.argv[1]
    
    listener = tf.TransformListener()

    predictor = tensorflow.contrib.predictor.from_saved_model(model_dir)

    rospy.Subscriber('/move_base_simple/goal', PoseStamped, goal_callback)
    rospy.Subscriber('/base_scan', LaserScan, scan_callback)
    move_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    # needed so that rviz can draw the robot's footprint
    footprint = rospy.get_param("/cnn_controller/local_costmap/footprint")
    rospy.Subscriber('/particlecloud', PoseArray, pose_callback)
    footprint_publisher = rospy.Publisher('/cnn_controller/footprint', PolygonStamped, queue_size=1)

    rate = rospy.Rate(5)

    while not rospy.is_shutdown():
        if latest_goal is not None:
            publish_move()

