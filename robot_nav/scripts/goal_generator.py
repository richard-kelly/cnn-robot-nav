#!/usr/bin/env python  

import random
import math
import rospy
import tf
import datetime
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid


def publish_goal():
    if goal_calc_complete:
        print 'Making a new goal'
        y, x = random.choice(goal_candidates)
        angle = random.random() * 2 * math.pi - math.pi
        q = quaternion_from_euler(0, 0, angle)
        pose = PoseStamped()
        pose.header.frame_id = '/map'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = q[2] 
        pose.pose.orientation.w = q[3]
        print 'publishing:'
        print pose
        goal_publisher.publish(pose)
    else:
        print 'goals not calculated yet'
        

def goal_callback(pose):
    global latest_goal
    latest_goal = pose


def move_callback(twist):
    global latest_move_time    
    latest_move_time = datetime.datetime.now()


def map_callback(grid):
    global min_space, goal_calc_complete
    map_sub.unregister()

    w = grid.info.width
    h = grid.info.height
    clear = int(math.ceil(min_space / grid.info.resolution))
    for row in xrange(clear, h - clear):
        for col in xrange(clear, w - clear):
            if grid.data[w * row + col] == 0 and clear_space(grid, row, col):
                goal_candidates.append((row * grid.info.resolution, col * grid.info.resolution))
    
    goal_calc_complete = True


def clear_space(grid, row, col):
    global min_space
    w = grid.info.width
    h = grid.info.height
    clear = int(math.ceil(min_space / grid.info.resolution))
    for x in xrange(row - clear, row + clear + 1):
        for y in xrange(col - clear, col + clear + 1):
            if grid.data[w * x + y] != 0:
                return False
    return True


if __name__ == '__main__':
    rospy.init_node('goal_generator')

    listener = tf.TransformListener()

    latest_goal = None
    latest_move_time = None

    goal_candidates = []
    goal_calc_complete = False

    goal_publisher = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)

    rospy.Subscriber('/move_base_simple/goal', PoseStamped, goal_callback)
    rospy.Subscriber('/cmd_vel', Twist, move_callback)
    map_sub = rospy.Subscriber('/map', OccupancyGrid, map_callback)

    rate = rospy.Rate(1)
    min_dist = 1.2
    min_angle = 1
    min_space = 1.0 # meters square of free space around goals

    while not rospy.is_shutdown():
        if latest_goal is None or latest_move_time is None:
            publish_goal()
        else:
            #determine distance to goal
            trans, rot = listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            euler_angles = euler_from_quaternion(rot)
            x1, y1, _ = trans
            theta1 = euler_angles[2]
            
            q = latest_goal.pose.orientation
            euler_angles = euler_from_quaternion([q.x, q.y, q.z, q.w])
            x2 = latest_goal.pose.position.x
            y2 = latest_goal.pose.position.y
            theta2 = euler_angles[2]
            
            d_trans = math.sqrt((abs(x1 - x2))**2 + (abs(y1 - y2))**2)
            if theta1 < 0:
                theta1 += 2 * math.pi
            if theta2 < 0:
                theta2 += 2 * math.pi            
            d_rot = abs(theta1 - theta2)
            # if the robot is near the goal AND the controller has stopped sending twist commands
            # if d_trans < min_dist:
            if True:
                current_time = datetime.datetime.now()
                elapsed = current_time - latest_move_time
                #if elapsed.seconds > 0 or elapsed.microseconds > 500 * 1000:
                if elapsed.seconds >= 2:
                    publish_goal()

        rate.sleep()


