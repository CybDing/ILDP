import numpy as np
import genesis as gs

########################## 初始化 ##########################
gs.init(backend=gs.gpu)

########################## 创建场景 ##########################
scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (3, -1, 1.5),
        camera_lookat = (0.0, 0.0, 0.5), 
        camera_fov    = 30,
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    show_viewer = True,
)

########################## 创建实体 ##########################
# 添加地面
scene.add_entity(gs.morphs.Plane())

# 添加目标立方体
cube = scene.add_entity(
    gs.morphs.Box(
        size = (0.04, 0.04, 0.04),
        pos  = (0.65, 0.0, 0.02),
    )
)

# 添加Franka机械臂
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

########################## 构建场景 ##########################
scene.build()

# 定义关节索引
motors_dof = np.arange(7)     # 机械臂关节
fingers_dof = np.arange(7, 9) # 夹爪关节

# 设置控制器参数
# 注意：以下值是为实现Franka最佳行为而调整的。
# 有时高质量的URDF或XML文件也会提供这些参数，并会被解析。
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]), 
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
)

# 获取末端执行器链接
end_effector = franka.get_link('hand')

# 用IK求解预抓取位姿的关节角度
qpos = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.25]),
    quat = np.array([0, 1, 0, 0]),
)
qpos[-2:] = 0.04  # 夹爪打开

# 规划运动路径
path = franka.plan_path(
    qpos_goal     = qpos,
    num_waypoints = 200, # 2秒时长
)

# 执行规划路径
for waypoint in path:
    franka.control_dofs_position(waypoint)
    scene.step()

# 等待到达最后一个路径点
for i in range(100):
    scene.step()

# 向下移动到抓取位置
qpos = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.135]),
    quat = np.array([0, 1, 0, 0]),
)
franka.control_dofs_position(qpos[:-2], motors_dof)
for i in range(100):
    scene.step()

# 夹紧物体
franka.control_dofs_position(qpos[:-2], motors_dof)
franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)
for i in range(100):
    scene.step()

# 抬起物体
qpos = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.3]),
    quat = np.array([0, 1, 0, 0]),
)
franka.control_dofs_position(qpos[:-2], motors_dof)
for i in range(200):
    scene.step()