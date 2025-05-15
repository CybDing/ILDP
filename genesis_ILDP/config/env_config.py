import os

file_path = os.path.dirname(os.path.abspath(__file__)) # Return the directory path for current file
robot_path = os.path.join(file_path, '../assets/roboticArms/urdf/rizon44/flexiv_rizon4_kinematics.urdf')
plane_path = os.path.join(file_path, '../assets/roboticArms/urdf/plane.urdf')
cube_path = os.path.join(file_path, '../assets/roboticArms/urdf/cube.urdf')

env_path = {
    'robot': robot_path,
    'plane': plane_path,
    'cube': cube_path
}
target_folder = "./test/" 