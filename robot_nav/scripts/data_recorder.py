#!/usr/bin/env python  

import utils
import os
import sys
import rospy
import tf
import datetime
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan


def scan_callback(scan):
    global latest_scan
    latest_scan = scan.ranges


def goal_callback(pose):
    global latest_goal, trajectory
    trajectory += 1
    latest_goal = pose


def move_callback(twist):
    global latest_goal, latest_scan, line, filename, file_num
    if latest_goal is None or latest_scan is None:
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
    
    r, phi = utils.cart_to_pol(robot_goal.pose.position.x, robot_goal.pose.position.y)
    target = r, phi, heading
    command = twist.linear.x, twist.angular.z
    
    line += 1
    # start a new file if we're at 100,000 lines
    if line > 100000:
        dir = '/home/data'
        filename = dir + '/' + map_name + '_' + today + '_' + str(file_num) + '.csv'
        file_num += 1
        line = 1

    if line % 100 == 0:
        print 'writing file ' + str(file_num) + ', line ' + str(line)
    with open(filename, 'a+') as f:
        f.write('{:.5f}, '.format(trajectory))
        for val in latest_scan:
            f.write('{:.5f}, '.format(val))
        for val in target:
            f.write('{:.5f}, '.format(val))
        f.write('{:.5f}, {:.5f}\n'.format(command[0], command[1]))


if __name__ == '__main__':
    rospy.init_node('data_recorder')    

    latest_goal = None
    latest_scan = None

    map_name = sys.argv[1]
    today = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    dir = '/home/data'
    if not os.path.exists(dir):
        os.makedirs(dir)
    filename = dir + '/' + map_name + '_' + today + '_0.csv'

    line = 0
    file_num = 1
    trajectory = 0

    listener = tf.TransformListener()

    rospy.Subscriber('/move_base_simple/goal', PoseStamped, goal_callback)
    rospy.Subscriber('/base_scan', LaserScan, scan_callback)
    rospy.Subscriber('/cmd_vel', Twist, move_callback)

    rospy.spin()

