import genesis as gs
import sys
import os
import numpy as np

file_path = os.path.dirname(os.path.abspath(__file__)) # Return the directory path for current file
urdf_path = os.path.join(file_path, 'genesis_ILDP/assets/roboticArms/urdf/rizon44/flexiv_rizon4_kinematics.urdf')
# urdf_path = os.path.join(file_path, 'roboticArms/urdf/rizon/dual_rizon_edited.urdf')

device = gs.gpu
# dtype = gs.tc_float # 32 precision
seed = None
gs.init(backend=device, seed=seed)

scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (3, -1, 1.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        res           = (960, 640),
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01
    ),
    show_viewer = True
)
plane = scene.add_entity(
    gs.morphs.Plane(
    ),
)
flexiv : gs.engine.entities.RigidEntity = scene.add_entity(
    gs.morphs.URDF(
        pos = (0, 0, 0.5),
        file = urdf_path,
        fixed=True,
        collision=True,
        links_to_keep=[
          # 'tcp',
            'grav_base_link',
            "flange_with_ori"
        ]
    )
)
cube = scene.add_entity(
    gs.morphs.Box(
        size = (0.07, 0.07, 0.07),
        pos = (0.65, 0.0, 0.035)
    )
)

scene.build()

jnt_names = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'joint7',
    'finger_width_joint' # grav mimic
]
# finger_names = [
#     'left_inner_finger_joint',
#     'right_inner_finger_joint',
# ]

dofs_idx = [flexiv.get_joint(name).dof_idx_local for name in jnt_names]
# print(dofs_idx) 0-6, 11

# fingers_idx = [flexiv.get_joint(name).dof_idx_local for name in finger_names]
# print(fingers_idx) 12, 13

end_effector = flexiv.get_link('flange_with_ori') # 设置一个与安装方向相同的flange虚拟link，可以使用grav_base_link代替

flexiv.set_dofs_kp(
    kp             = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100]),
    dofs_idx_local = dofs_idx,
)
flexiv.set_dofs_kv(
    kv             = np.array([450, 450, 350, 350, 200, 200, 200, 10]),
    dofs_idx_local = dofs_idx,
)

flexiv.control_dofs_position(np.zeros(shape=(len(dofs_idx) ,)), dofs_idx) # set back to initial frame
flexiv.control_dofs_position(np.array([0.15]), dofs_idx[-1])
for i in range(100):
    scene.step()

qpos = flexiv.inverse_kinematics(
    link = end_effector,
    pos = np.array([0.65, 0.0, 0.28]),
    quat = np.array([0, 1, 0, 0]),
    dofs_idx_local=dofs_idx[0:7]
)

path = flexiv.plan_path(
    qpos_goal = qpos,
    num_waypoints = 200
)
for waypoint in path:
    flexiv.control_dofs_position(waypoint[0:7], dofs_idx[0:7])
    scene.step()
for i in range(100):
    scene.step()

qpos = flexiv.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.22]),
    quat = np.array([0, 1, 0, 0]),
    dofs_idx_local=dofs_idx[0:7],
    damping=2,
    max_solver_iters=100
)

# grasp
flexiv.control_dofs_position(qpos[0:7], dofs_idx[0:7])
for i in range(150):
    scene.step()

flexiv.control_dofs_position(np.array([-0.05]), dofs_idx[-1]) # TODO 如何选择合适的作用力 这里是强迫一直到夹爪提供最大夹力为止
for i in range(250):
    scene.step()

# lift
qpos = flexiv.inverse_kinematics(
    link = end_effector,
    pos = np.array([0.65, 0.0, 0.4]),
    quat = np.array([0, 1, 0, 0]),
    dofs_idx_local=dofs_idx[0:7]
)

flexiv.control_dofs_position(qpos[0:7], dofs_idx[0:7])
for i in range(100):
    scene.step()