#!/usr/bin/env python

import rospy
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
import csv
import os

def read_csv_file(file_path):
    csv_content = ""
    try:
        with open(file_path, 'r') as file:
            csv_content = file.read()
    except IOError as e:
        rospy.logerr(f"Error reading CSV file: {e}")
    return csv_content

def normalize_data(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) if max_val != min_val else 0.5 for x in data]

def get_color(value):
    # value가 0에 가까울수록 빨간색, 1에 가까울수록 초록색
    return f"{1-value} {value} 0"  # R G B

def spawn_gazebo_models(csv_file_path):
    rospy.init_node('spawn_model_node', anonymous=True)
    rospy.wait_for_service('gazebo/spawn_sdf_model')
    spawn_model_service = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)

    csv_content = read_csv_file(csv_file_path)
    
    if not csv_content:
        rospy.logerr("Failed to read CSV file or file is empty.")
        return

    csv_data = []
    color_data = []
    for row in csv.reader(csv_content.split('\n')):
        if row and len(row) >= 4:  # 최소 4개의 값이 있는지 확인
            try:
                csv_data.append([float(val) for val in row[:3]])  # 첫 3개 값은 위치
                color_data.append(float(row[3]))  # 4번째 값은 색상에 사용
            except ValueError as e:
                rospy.logwarn(f"Skipping invalid row: {row}. Error: {e}")

    normalized_color_data = normalize_data(color_data)

    for i, ((x, y, z), color_value) in enumerate(zip(csv_data, normalized_color_data)):
        model_name = f'sphere_{i}'
        color = get_color(color_value)
        
        model_xml = f"""
        <?xml version="1.0"?>
        <sdf version="1.4">
          <model name="{model_name}">
            <static>true</static>
            <link name="link">
              <visual name="visual">
                <geometry>
                  <sphere>
                    <radius>0.015</radius>
                  </sphere>
                </geometry>
                <material>
                  <ambient>{color} 1</ambient>
                  <diffuse>{color} 1</diffuse>
                  <specular>{color} 1</specular>
                </material>
              </visual>
            </link>
          </model>
        </sdf>
        """

        initial_pose = Pose()
        initial_pose.position.x = x
        initial_pose.position.y = y
        initial_pose.position.z = z

        try:
            spawn_model_service(model_name, model_xml, "", initial_pose, "world")
            rospy.loginfo(f"Successfully spawned model: {model_name}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to spawn model: {e}")

if __name__ == '__main__':
    try:
        csv_file_path = os.path.expanduser("matching_rows.csv")
        spawn_gazebo_models(csv_file_path)
    except rospy.ROSInterruptException:
        pass