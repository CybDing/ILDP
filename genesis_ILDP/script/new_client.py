import pygame
import pymunk
import time
import threading
import requests
from pygame.locals import *

# 服务器配置
SERVER_IP = "127.0.0.1"  # 你的服务器IP
ACTION_PORT = 7100  # 环境服务器端口
DATA_PORT = 6100    # 数据服务器端口

latest_data = {
    "cur_keypoints": [],
    "target_keypoints": [],
    "agent_pos": [0.2, 0.2],
    "timestamp": time.time(),
    "intersection_ratio": 0.0,
    "reward": 0.0
}
data_lock = threading.Lock()

# 添加action发送相关的全局变量
action_sending = False
action_frequency = 10  # Hz
last_action_time = 0
current_mouse_pos = (0, 0)
last_action_sent = (0, 0)
action_count = 0

# 数据获取相关
data_fetching = True
data_fetch_frequency = 200  # Hz
last_data_fetch = 0

def send_action(x, y):
    """发送action到环境服务器"""
    global last_action_sent, action_count
    try:
        payload = {"action": [x, y]}
        url = f"http://{SERVER_IP}:{ACTION_PORT}/api/action"
        response = requests.post(url, json=payload, timeout=0.5)
        
        if response.status_code == 200:
            last_action_sent = (x, y)
            action_count += 1
            print(f"Action #{action_count} sent: ({x:.3f}, {y:.3f})")
        else:
            print(f"Failed to send action: {response.status_code}")
    except Exception as e:
        print(f"Error sending action: {e}")

def fetch_data():
    """从数据服务器获取最新数据"""
    global latest_data
    try:
        url = f"http://{SERVER_IP}:{DATA_PORT}/api/positions"
        response = requests.get(url, timeout=0.5)
        
        if response.status_code == 200:
            data = response.json()
            with data_lock:
                latest_data.update(data)
            return True
        else:
            print(f"Failed to fetch data: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error fetching data: {e}")
        return False

# def register_with_server():
#     """向服务器注册客户端"""
#     try:
#         url = f"http://{SERVER_IP}:{DATA_PORT}/api/register_client"
#         client_info = {
#             "client_type": "pygame_visualizer",
#             "version": "1.0"
#         }
#         response = requests.post(url, json=client_info, timeout=2.0)
        
#         if response.status_code == 200:
#             print("✅ Successfully registered with server")
#             return True
#         else:
#             print(f"❌ Failed to register: {response.status_code}")
#             return False
#     except Exception as e:
#         print(f"❌ Error registering with server: {e}")
#         return False

def data_fetch_thread():
    """数据获取线程"""
    global last_data_fetch
    
    while data_fetching:
        current_time = time.time()
        if current_time - last_data_fetch > (1.0 / data_fetch_frequency):
            fetch_data()
            last_data_fetch = current_time

# 初始化pygame (保持原有代码)
pygame.init()
width, height = 1000, 800
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 14)

# 颜色定义 (保持原有代码)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 50, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
SCALE = 800

# 保持原有的类定义
class SimpleT:
    def __init__(self, color, label):
        self.points = []
        self.color = color
        self.label = label
        
    def update(self, keypoints):
        if len(keypoints) >= 8:
            self.points = [(p[0] * SCALE, p[1] * SCALE) for p in keypoints]
    
    def draw(self, surface):
        if len(self.points) < 8:
            return
            
        connections = [(i, (i+1)%8) for i in range(8)]
        for i, j in connections:
            pygame.draw.line(surface, self.color, 
                           (int(self.points[i][0]), int(self.points[i][1])),
                           (int(self.points[j][0]), int(self.points[j][1])), 2)
        
        for x, y in self.points:
            pygame.draw.circle(surface, self.color, (int(x), int(y)), 4)
        
        if self.points:
            center_x = sum(p[0] for p in self.points) / len(self.points)
            center_y = sum(p[1] for p in self.points) / len(self.points)
            text = font.render(self.label, True, BLACK)
            surface.blit(text, (center_x + 10, center_y + 10))

class SimpleEEF:
    def __init__(self):
        self.pos = (200, 200)
        
    def update(self, agent_pos):
        if len(agent_pos) >= 2:
            self.pos = (agent_pos[0] * SCALE, agent_pos[1] * SCALE)
    
    def draw(self, surface):
        x, y = int(self.pos[0]), int(self.pos[1])
        pygame.draw.circle(surface, BLUE, (x, y), 6)
        pygame.draw.line(surface, BLUE, (x-10, y), (x+10, y), 2)
        pygame.draw.line(surface, BLUE, (x, y-10), (x, y+10), 2)

def draw_mouse_cursor(surface, pos):
    x, y = pos
    pygame.draw.circle(surface, YELLOW, (int(x), int(y)), 8, 2)
    pygame.draw.line(surface, YELLOW, (x-12, y), (x+12, y), 2)
    pygame.draw.line(surface, YELLOW, (x, y-12), (x, y+12), 2)

# 创建对象
t_current = SimpleT(RED, "Current")
t_target = SimpleT(GREEN, "Target")
eef = SimpleEEF()


data_thread = threading.Thread(target=data_fetch_thread, daemon=True)
data_thread.start()
print("Data fetching started")
# 注册客户端并启动数据获取线程
# print(f"Connecting to server: {SERVER_IP}")
# if register_with_server():

# else:
#     print("Failed to connect to server, running in offline mode")

# 主循环 (保持原有逻辑，但移除Flask相关代码)
running = True
last_update = 0
UPDATE_INTERVAL = 0.01

print("Pygame visualization started. Press ESC to quit, S for stats.")
print("Press SPACE to start/stop sending actions at mouse position")

while running:
    current_time = time.time()
    current_mouse_pos = pygame.mouse.get_pos()
    
    # 事件处理 (保持原有代码)
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            elif event.key == K_SPACE:
                action_sending = not action_sending
                if action_sending:
                    print("Started sending actions")
                else:
                    print("Stopped sending actions")
            elif event.key == K_s:
                # 显示服务器状态
                try:
                    url = f"http://{SERVER_IP}:{DATA_PORT}/api/status"
                    response = requests.get(url, timeout=1.0)
                    if response.status_code == 200:
                        status = response.json()
                        print(f"\n=== 服务器状态 ===")
                        print(f"连接的客户端: {status['connected_clients']}")
                        print(f"数据时间戳: {status['latest_data_timestamp']}")
                        print(f"Action发送次数: {action_count}")
                except Exception as e:
                    print(f"无法获取服务器状态: {e}")
    
    # 自动发送action
    if action_sending and (current_time - last_action_time) > (1.0 / action_frequency):
        mouse_x, mouse_y = current_mouse_pos
        action_x = max(0, min(1, mouse_x / SCALE))
        action_y = max(0, min(1, mouse_y / SCALE))
        
        threading.Thread(target=send_action, args=(action_x, action_y), daemon=True).start()
        last_action_time = current_time
    
    # 更新显示数据
    if current_time - last_update > UPDATE_INTERVAL:
        with data_lock:
            data = latest_data.copy()
        
        if "cur_keypoints" in data and data["cur_keypoints"]:
            t_current.update(data["cur_keypoints"])
        if "target_keypoints" in data and data["target_keypoints"]:
            t_target.update(data["target_keypoints"])
        if "agent_pos" in data:
            eef.update(data["agent_pos"])
            
        last_update = current_time
    
    # 绘制 (保持原有代码)
    screen.fill(WHITE)
    t_current.draw(screen)
    t_target.draw(screen)
    eef.draw(screen)
    draw_mouse_cursor(screen, current_mouse_pos)
    
    # 状态显示
    with data_lock:
        data = latest_data.copy()
    
    status_lines = [
        f"Server: {SERVER_IP}",
        f"FPS: {int(clock.get_fps())}",
        f"Data age: {current_time - data.get('timestamp', 0):.1f}s",
        f"Intersection: {data.get('intersection_ratio', 0):.3f}",
        f"Reward: {data.get('reward', 0):.3f}",
        "",
        f"Action sending: {'ON' if action_sending else 'OFF'}",
        f"Action count: {action_count}",
        f"Mouse pos: ({current_mouse_pos[0]}, {current_mouse_pos[1]})",
        f"Action pos: ({current_mouse_pos[0]/SCALE:.3f}, {current_mouse_pos[1]/SCALE:.3f})",
        f"Last action: ({last_action_sent[0]:.3f}, {last_action_sent[1]:.3f})",
        "",
        "Press SPACE to start/stop actions",
        "Press S for server status",
        "Press ESC to quit"
    ]
    
    for i, line in enumerate(status_lines):
        text = font.render(line, True, BLACK)
        screen.blit(text, (10, 10 + i * 18))
    
    pygame.display.flip()
    clock.tick(60)

# 清理
data_fetching = False
pygame.quit()
print("Visualization stopped.")