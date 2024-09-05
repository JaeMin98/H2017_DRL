#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import rospy
import moveit_commander #Python Moveit interface를 사용하기 위한 모듈
import moveit_msgs.msg
import geometry_msgs.msg
import random
import math
from moveit_commander.conversions import pose_to_list
from moveit_commander import PlanningSceneInterface, RobotCommander, MoveGroupCommander
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import os
from datetime import datetime
import Config

# alpha shape #
import math
import csv
# alpha shape #

class RobotArmControl:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface', anonymous=True)
        self.move_group = MoveGroupCommander("h2017")
        rospy.loginfo("RobotArmControl initialized successfully")
        
        # CSV 파일 열기
        self.level_point = []
        for i in range(1,6):
            with open('./DataCSV/UoC_'+str(i)+'.csv', 'r') as file:
                reader = csv.reader(file)

                # 각 행들을 저장할 리스트 생성
                rows = []

                for row in reader:
                    row_temp = row[:3]
                    rows.append(row_temp)
                self.level_point.append(rows)

        self.MAX_Level_Of_Point = 4
        self.Level_Of_Point = 0

        Config.Clustering_K = self.MAX_Level_Of_Point+1

        self.isLimited = False 
        self.Limit_joint=[[-180.0, 180.0],   [-110.0,110.0],   [-140.0,140.0]]

        self.Iswait = True

        ## reward weight ##
        self.goalDistance = 0.05
        self.farDistance = 2.999

        self.prev_state = []
        self.prev_distance = None
        self.joint_error_count = 0
        self.time_step = 0
        self.MAX_time_step = Config.max_episode_steps


        self.weight = Config.action_weight

    def Degree_to_Radian(self,Dinput):
        Radian_list = []
        for i in Dinput:
            Radian_list.append(i* (math.pi/180.0))
        return Radian_list

    def Radian_to_Degree(self,Rinput):
        Degree_list = []
        for i in Rinput:
            Degree_list.append(i* (180.0/math.pi))
        return Degree_list

    def action(self,angle):  # angle 각도로 이동 (angle 은 크기 6의 리스트 형태)
        joint = self.get_state()[0:6]

        joint[0] += angle[0] * self.weight
        joint[1] += angle[1] * self.weight
        joint[2] += angle[2] * self.weight
        joint[3] = 0
        joint[4] = 0
        joint[5] = 0

        for i in range(len(self.Limit_joint)):
            if(self.Limit_joint[i][1] < joint[i]):
                joint[i] = self.Limit_joint[i][1]
                # self.isLimited = True
            elif(self.Limit_joint[i][0] > joint[i]):
                joint[i] = self.Limit_joint[i][0]
                # self.isLimited = True

        try:
            self.move_group.go(self.Degree_to_Radian(joint), wait=self.Iswait)
        except:
            print("move_group.go EXCEPT, ", joint)
            self.isLimited = True

        self.time_step += 1
            
    def reset(self):
        # print("Go To Home pose")
        self.time_step = 0
        self.move_group.go([0,0,0,0,0,0], wait=True)
        
        if random.random() < Config.Current_Data_Selection_Ratio:
            random_index_list = random.choice(range(len(self.level_point[self.Level_Of_Point])))
            self.target = self.level_point[self.Level_Of_Point][random_index_list]
            self.target = [float(element) for element in self.target] # 목표 지점 위치
            self.target_reset()
        else :
            temp_range = max(0,self.Level_Of_Point - 1)
            temp_index = random.randint(0, temp_range)
            random_index_list = random.choice(range(len(self.level_point[temp_index])))
            self.target = self.level_point[temp_index][random_index_list]
            self.target = [float(element) for element in self.target] # 목표 지점 위치
            self.target_reset()
        

    def get_state(self): #joint 6축 각도
        joint = self.move_group.get_current_joint_values()
        state = self.Radian_to_Degree(joint)[0:3] + self.target + self.get_pose()
        if(len(state) == 9):
            self.prev_state = state
        else:
            state = self.prev_state
            self.joint_error_count += 1
            print(self.joint_error_count)
        return state


    def get_pose(self):
        pose = self.move_group.get_current_pose().pose
        pose_value = [pose.position.x,pose.position.y,pose.position.z]
        return pose_value
    
    def get_reward(self):
        # self.move_group.stop()
        end_effector = self.get_pose()
        
        d = math.sqrt(abs((end_effector[0]-self.target[0])**2 + (end_effector[1]-self.target[1])**2 + (end_effector[2]-self.target[2])**2 ))

        # reward parameter
        rewardS = 0 # 도달 성공 시 부여
        rewardD = -1 * d # 거리가 가까울수록 부여
        R_extra = 0 # 로봇팔 동작 가능 범위(각도)를 벗어나면 부여

        if (self.prev_distance != None ): R_extra = -1 * (d - self.prev_distance)
        self.prev_distance = d

        isFinished = False
        isComplete = 0

        if(self.time_step >= self.MAX_time_step):
            isFinished = True
        
        # 목표 지점 도달 시
        if (d <= self.goalDistance):
            isFinished = True
            isComplete = 1  
            rewardS = 50

        totalReward = (rewardS)
        totalReward += (rewardD)
        totalReward += (R_extra)

        return totalReward,isFinished,isComplete

    def observation(self):
        totalReward,isFinished,isComplete = self.get_reward()
        current_state = self.get_state()

        return current_state,totalReward,isFinished, isComplete
    
    def step(self, actions):
        self.action(actions)
        return self.observation()
    
    def target_reset(self):
        state_msg = ModelState()
        state_msg.model_name = 'cube'
        state_msg.pose.position.x = self.target[0]
        state_msg.pose.position.y = self.target[1]
        state_msg.pose.position.z = self.target[2]
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0

        rospy.wait_for_service('/gazebo/set_model_state')
        for i in range(500):
            set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            resp = set_state(state_msg)    