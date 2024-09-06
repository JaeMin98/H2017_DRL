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

class RobotArmControl:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface', anonymous=True)
        self.move_group = MoveGroupCommander("h2017")
        rospy.loginfo("RobotArmControl initialized successfully")
        
        self.load_level_points()
        self.initialize_parameters()

    def load_level_points(self):
        self.level_point = []
        for i in range(1, 6):
            with open(f'./DataCSV/UoC_{i}.csv', 'r') as file:
                reader = csv.reader(file)
                self.level_point.append([row[:3] for row in reader])

    def initialize_parameters(self):
        self.MAX_Level_Of_Point = 4
        self.Level_Of_Point = 0
        self.Limit_joint = [[-180.0, 180.0], [-110.0, 110.0], [-140.0, 140.0]]
        self.goalDistance = 0.1
        self.prev_state = []
        self.prev_distance = None
        self.time_step = 0

    def degree_to_radian(self, degree_input):
        return [math.radians(d) for d in degree_input]

    def radian_to_degree(self, radian_input):
        return [math.degrees(r) for r in radian_input]

    def action(self, angle):
        
        joint = self.get_state()[:6]
        for i in range(3):
            joint[i] += angle[i] * Config.action_weight
        joint[3:6] = [0, 0, 0]

        for i in range(len(self.Limit_joint)):
            joint[i] = max(self.Limit_joint[i][0], min(joint[i], self.Limit_joint[i][1]))

        try:
            plan = self.move_group.go(self.degree_to_radian(joint), wait=True)
        except:
            plan = False

        collision_detected = not plan

        self.time_step += 1
        print(collision_detected)
        return collision_detected
            
    def reset(self):
        self.time_step = 0
        self.move_group.go([0, 0, 0, 0, 0, 0], wait=True)
        
        if random.random() < Config.Current_Data_Selection_Ratio:
            level = self.Level_Of_Point
        else:
            level = random.randint(0, max(0, self.Level_Of_Point - 1))

        random_index = random.choice(range(len(self.level_point[level])))
        self.target = [float(element) for element in self.level_point[level][random_index]]
        self.target_reset()

    def get_state(self):
        try:
            joint = self.move_group.get_current_joint_values()
            state = self.radian_to_degree(joint)[:3] + self.target + self.get_pose()
            if len(state) == 9:
                self.prev_state = state
            else:
                state = self.prev_state
        except:
            state = self.prev_state
        return state

    def get_pose(self):
        pose = self.move_group.get_current_pose().pose
        return [pose.position.x, pose.position.y, pose.position.z]
    
    def get_reward(self):
        end_effector = self.get_pose()
        d = math.sqrt(sum((e - t)**2 for e, t in zip(end_effector, self.target)))

        rewardS = 0
        rewardD = -d
        R_extra = -1 * (d - self.prev_distance) if self.prev_distance is not None else 0
        self.prev_distance = d

        isFinished = self.time_step >= Config.max_episode_steps

        if d <= self.goalDistance:
            isFinished = True
            rewardS = 50

        totalReward = rewardS + rewardD + R_extra
        isComplete = int(d <= self.goalDistance)

        return totalReward, isFinished, isComplete

    def observation(self):
        totalReward, isFinished, isComplete = self.get_reward()
        current_state = self.get_state()
        return current_state, totalReward, isFinished, isComplete
    
    def step(self, actions):
        self.action(actions)
        return self.observation()
    
    def target_reset(self):
        state_msg = ModelState()
        state_msg.model_name = 'cube'
        state_msg.pose.position.x, state_msg.pose.position.y, state_msg.pose.position.z = self.target
        state_msg.pose.orientation.x = state_msg.pose.orientation.y = state_msg.pose.orientation.z = state_msg.pose.orientation.w = 0

        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        for _ in range(500):
            set_state(state_msg)