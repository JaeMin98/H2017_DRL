import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
from moveit_commander.conversions import pose_to_list
from moveit_msgs.msg import RobotState
from tf.transformations import quaternion_matrix
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import time

class Ned2_control:
    def __init__(self) -> None:
        """
        Initialize the Ned2 robot control environment.
        Sets up ROS nodes, move groups, and initializes various parameters for the robot control.
        """
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface', anonymous=True)
        self.move_group = moveit_commander.MoveGroupCommander("ned2")
        self.robot = moveit_commander.RobotCommander()

        self.target = np.zeros(3)
        self.isLimited = False
        self.Iswait = False
        self.Limit_joint = np.array([[-171.88, 171.88],
                                     [-105.0, 34.96],
                                     [-76.78, 89.96],
                                     [-119.75, 119.75],
                                     [-110.01, 110.17],
                                     [-144.96, 144.96]])
        self.weight = np.array([6.8, 3, 3.32, 4.8, 4.4, 5.8])

        self.time_step = 0
        self.MAX_time_step = 200
        self.prev_distance = None

    @staticmethod
    def Degree_to_Radian(Dinput) -> float:
        """Convert degrees to radians."""
        return np.deg2rad(Dinput)

    @staticmethod
    def Radian_to_Degree(Rinput) -> float:
        """Convert radians to degrees."""
        return np.rad2deg(Rinput)
    
    @staticmethod
    def calc_distance(point1, point2) -> float:
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def action(self, angle) -> None:
        """
        Execute an action by moving the robot's joints.
        angle (array-like): Change in joint angles to be applied
        """
        joint = np.array(self.move_group.get_current_joint_values())
        angle = self.Degree_to_Radian(angle)

        joint[:3] += angle[:3] * self.weight[:3]
        joint[3:] = 0

        joint = np.clip(joint, self.Limit_joint[:, 0], self.Limit_joint[:, 1])

        try:
            self.move_group.go(joint.tolist(), wait=self.Iswait)
        except:
            self.isLimited = True

        self.time_step += 1
            
    def reset(self) -> None:
        """
        Reset the environment to its initial state.
        This includes resetting the robot position, time step, and target.
        """
        self.time_step = 0
        self.prev_distance = None
        self.isLimited = False
        self.move_group.go([0]*6, wait=True)
        self.set_random_target()
    
    def get_endeffector_position(self) -> list[float]:
        """Get the current position of the robot's end effector."""
        pose = self.move_group.get_current_pose().pose
        return [pose.position.x, pose.position.y, pose.position.z]
    
    def get_state(self) -> list[float]:
        """Get the current state of the environment."""
        joint = self.move_group.get_current_joint_values()
        return joint[:3] + self.target.tolist()

    def get_reward(self) -> tuple[float, bool, bool]:
        """Calculate the reward based on the current state of the environment."""
        distance = self.calc_distance(self.get_endeffector_position(), self.target)

        R_basic = -distance
        R_done = 0
        R_extra = -1 * (distance - self.prev_distance) if self.prev_distance is not None else 0
        self.prev_distance = distance

        isDone, isSuccess = False, False

        if self.get_endeffector_position()[2] < 0.1 or self.isLimited:
            R_done = -10
            isDone, isSuccess = True, False
     
        if self.time_step >= self.MAX_time_step:
            isDone, isSuccess = True, False

        if distance <= 0.05:
            R_done = 50
            isDone, isSuccess = True, True

        totalReward = R_basic + R_done + R_extra
        return totalReward, isDone, isSuccess
    
    def step(self, angle) -> tuple[list[float], float, bool, bool]:
        """Take a step in the environment by executing an action and observing the result."""
        time_interver = 0.05
        self.action(angle)
        time.sleep(time_interver)
        self.move_group.stop()

        totalReward, isDone, isSuccess = self.get_reward()
        current_state = self.get_state()

        return current_state, totalReward, isDone, isSuccess
    
    def set_random_target(self) -> None:
        """Set a random target position for the robot to reach."""
        while True:
            random_pose = self.move_group.get_random_pose()
            if random_pose.pose.position.z > 0.1:
                break
        self.target = np.array([random_pose.pose.position.x, random_pose.pose.position.y, random_pose.pose.position.z])
        self.target_reset()

    def target_reset(self) -> None:
        """Reset the target object's position in the simulation environment."""
        state_msg = ModelState()
        state_msg.model_name = 'cube'
        state_msg.pose.position.x, state_msg.pose.position.y, state_msg.pose.position.z = self.target
        state_msg.pose.orientation.x = state_msg.pose.orientation.y = state_msg.pose.orientation.z = state_msg.pose.orientation.w = 0

        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        for _ in range(100):
            set_state(state_msg)