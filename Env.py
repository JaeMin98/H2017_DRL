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

def spawn_gazebo_models(csv_file_path):
    rospy.init_node('spawn_model_node', anonymous=True)
    rospy.wait_for_service('gazebo/spawn_sdf_model')
    spawn_model_service = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)

    csv_content = read_csv_file(csv_file_path)
    if not csv_content:
        rospy.logerr("Failed to read CSV file or file is empty.")
        return

    csv_data = []
    for row in csv.reader(csv_content.split('\n')):
        if row and len(row) >= 3:  # Ensure we have at least X, Y, Z values
            try:
                # Skip the first row if it's a header
                if row[0] == "Timestamp":
                    continue
                csv_data.append([float(row[1]), float(row[2]), float(row[3])])  # X, Y, Z are in columns 1, 2, 3
            except ValueError as e:
                rospy.logwarn(f"Skipping invalid row: {row}. Error: {e}")

    for i, (x, y, z) in enumerate(csv_data):
        model_name = f'sphere_{i}'
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
                <ambient>1 0 0 1</ambient>
                <diffuse>1 0 0 1</diffuse>
                <specular>1 0 0 1</specular>
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
        csv_file_path = "job_66_virtual.csv"  # Adjust this path if needed
        spawn_gazebo_models(csv_file_path)
    except rospy.ROSInterruptException:
        pass