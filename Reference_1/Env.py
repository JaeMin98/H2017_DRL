#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import rospy
import moveit_commander
import random
import math
import csv
from moveit_commander import MoveGroupCommander
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import Config
import pandas as pd
import numpy as np
import time

class RobotArmControl:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface', anonymous=True)
        self.move_group = MoveGroupCommander("h2017")
        rospy.loginfo("RobotArmControl initialized successfully")
        
        file_path = 'DataCSV/datapoints.csv'
        self.data = pd.read_csv(file_path)
        self.initialize_parameters()

    def initialize_parameters(self):
        self.Limit_joint = [[-180.0, 180.0], [-110.0, 110.0], [-140.0, 140.0]]
        self.goalDistance = 0.1
        self.prev_state = []
        self.prev_distance = None
        self.time_step = 0
        self.apply_collision = True
        self.collision_detected = False
        self.weight = Config.action_weight

    def degree_to_radian(self, degree_input):
        return [math.radians(d) for d in degree_input]

    def radian_to_degree(self, radian_input):
        return [math.degrees(r) for r in radian_input]

    def calc_distance(self, point1, point2):
        # 각 좌표의 차이를 제곱한 후 더한 값을 제곱근한다.
        distance = math.sqrt((point1[0] - point2[0]) ** 2 +
                            (point1[1] - point2[1]) ** 2 +
                            (point1[2] - point2[2]) ** 2)
        return distance
    
    def get_end_effector_linear_velocity(self, current_pose, previous_pose, dt):
        # 위치 변화 계산
        dx = current_pose[0] - previous_pose[0]
        dy = current_pose[1] - previous_pose[1]
        dz = current_pose[2] - previous_pose[2]

        # 속도 계산
        vx = dx / dt
        vy = dy / dt
        vz = dz / dt

        return [vx, vy, vz]
    
    def get_angle_between_velocity_and_target(self, current_pose, linear_velocity):
        # 현재 위치에서 목표 위치까지의 방향 벡터 계산
        direction_vector = [
            self.target[0] - current_pose[0],
            self.target[1] - current_pose[1],
            self.target[2] - current_pose[2]
        ]

        if(linear_velocity == [0, 0, 0]): linear_velocity = self.prev_linear_velocity
        else : self.prev_linear_velocity = linear_velocity

        # 벡터 정규화
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        linear_velocity = linear_velocity / np.linalg.norm(linear_velocity)
        
        # 두 벡터 사이의 각도 계산 (라디안)
        dot_product = np.dot(direction_vector, linear_velocity)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # 라디안을 도로 변환
        angle_degrees = np.degrees(angle)
        
        return angle_degrees

    def action(self, angle):
        
        joint = self.get_state()[:6]
        for i in range(3):
            joint[i] += angle[i] * self.weight
        joint[3:6] = [0, 0, 0]

        for i in range(len(self.Limit_joint)):
            joint[i] = max(self.Limit_joint[i][0], min(joint[i], self.Limit_joint[i][1]))

        try:
            plan = self.move_group.go(self.degree_to_radian(joint), wait=True)
        except:
            plan = False

        self.collision_detected = not plan
        self.time_step += 1
            
    def reset(self):
        self.time_step = 0
        self.move_group.go([0, 0, 0, 0, 0, 0], wait=True)
        self.set_random_target()

    def set_random_target(self):
        self.target = self.data.sample(n=1).values.tolist()[0]
        self.target_reset()

    def get_joint2_position(self):
        joint3_pose = self.move_group.get_current_pose(end_effector_link="link2").pose
        joint3_position = [joint3_pose.position.x, joint3_pose.position.y, joint3_pose.position.z]
        return joint3_position

    def get_joint3_position(self):
        joint3_pose = self.move_group.get_current_pose(end_effector_link="link3").pose
        joint3_position = [joint3_pose.position.x, joint3_pose.position.y, joint3_pose.position.z]
        return joint3_position
    
    def get_endeffector_position(self):
        pose = self.move_group.get_current_pose().pose
        pose_value = [pose.position.x,pose.position.y,pose.position.z]
        return pose_value
    
    def get_state(self):
        joint2_pos = self.get_joint2_position()
        joint3_pos = self.get_joint3_position()
        endeffector_pos = self.get_endeffector_position()
        
        relative_joint2 = [joint2_pos[0] - self.target[0], joint2_pos[1] - self.target[1], joint2_pos[2] - self.target[2]]
        relative_joint3 = [joint3_pos[0] - self.target[0], joint3_pos[1] - self.target[1], joint3_pos[2] - self.target[2]]
        relative_endeffector = [endeffector_pos[0] - self.target[0], endeffector_pos[1] - self.target[1], endeffector_pos[2] - self.target[2]]
        
        state = joint3_pos + endeffector_pos + relative_joint2 + relative_joint3 + relative_endeffector

        if(len(state) == 15): self.prev_state = state
        else : state = self.prev_state

        return state

    def get_pose(self):
        pose = self.move_group.get_current_pose().pose
        return [pose.position.x, pose.position.y, pose.position.z]
    
    def get_reward(self, distance, angle_difference):
        # R(position)

        # R(theta)
        if(angle_difference >= 90): R_theta = -0.001
        elif(90 > angle_difference >= 22.5): R_theta = 0.1
        elif(22.5 > angle_difference >= 11.25): R_theta = 0.3
        elif(11.25 > angle_difference >= 0): R_theta = 0.6

        # R(dinstance)
        df = 1.5
        if(distance >= df): R_distance = 0.0
        elif(df > distance >= df*0.7): R_distance = 0.01
        elif(df*0.7 > distance >= df*0.5): R_distance = 0.05
        elif(df*0.5 > distance >= df*0.2): R_distance = 0.2
        elif(df*0.2 > distance >= 0): R_distance = 1.0

        isDone, IsSuccess = False, False
        if(self.time_step >= 200):
            isDone,IsSuccess = True, False

        if(distance <= self.goalDistance):
            R_distance += 60
            isDone,IsSuccess = True,True
        elif(self.collision_detected):
            R_distance -= 60
            isDone,IsSuccess = True,False

        totalReward = R_theta + R_distance
        return totalReward, isDone,IsSuccess

    def observation(self):
        totalReward, isFinished, isComplete = self.get_reward()
        current_state = self.get_state()
        return current_state, totalReward, isFinished, isComplete
    
    def step(self, angle):
        distance = self.calc_distance(self.target, self.get_endeffector_position())

        df = 1.5
        if(distance >= df): self.weight = 10
        elif(df > distance >= df*0.7): self.weight = 5
        elif(df*0.7 > distance >= df*0.5): self.weight = 3
        elif(df*0.5 > distance >= df*0.2): self.weight = 2
        elif(df*0.2 > distance >= 0): self.weight = 1


        previous_pose = self.move_group.get_current_joint_values()
        start_time = time.time()
        self.action(angle)
        end_time = time.time()
        current_pose = self.move_group.get_current_joint_values()
        elapsed_time = (end_time - start_time)*23

        current_distance = self.calc_distance(self.target, self.get_endeffector_position())

        linear_velocity = self.get_end_effector_linear_velocity(current_pose, previous_pose, elapsed_time)
        angle_difference = self.get_angle_between_velocity_and_target(current_pose, linear_velocity)
        totalReward,isDone,IsSuccess = self.get_reward(current_distance, angle_difference)
        current_state = self.get_state()

        return current_state,totalReward,isDone, IsSuccess
    
    def target_reset(self):
        state_msg = ModelState()
        state_msg.model_name = 'cube'
        state_msg.pose.position.x, state_msg.pose.position.y, state_msg.pose.position.z = self.target
        state_msg.pose.orientation.x = state_msg.pose.orientation.y = state_msg.pose.orientation.z = state_msg.pose.orientation.w = 0

        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        for _ in range(500):
            set_state(state_msg)