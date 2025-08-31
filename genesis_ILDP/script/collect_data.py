import pygame
import pymunk
import time
import threading
import requests
from pygame.locals import *
from flask import Flask, request, jsonify

# 初始化
pygame.init()
width, height = 1000, 800
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 14)

# 颜色定义
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 50, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
SCALE = 800

# 全局数据存储
latest_data = {
    "cur_keypoints": [],
    "target_keypoints": [],
    "agent_pos": [0.3, 0.3],
    "timestamp": time.time(),
    "intersection_ratio": 0.0,
    "reward": 0.0
}
data_lock = threading.Lock()

# 频率监控
request_stats = {
    "last_time": 0,
    "count": 0,
    "intervals": [],
    "start_time": time.time()
}

# 添加action发送相关的全局变量
action_sending = False
action_frequency = 10  # Hz
last_action_time = 0
current_mouse_pos = (0, 0)  # 当前鼠标位置
last_action_sent = (0, 0)   # 最后发送的action
action_count = 0            # action计数器

# 发送action到环境服务器
def send_action(x, y):
    global last_action_sent, action_count
    try:
        payload = {"action": [x, y]}
        response = requests.post("http://localhost:7100/api/action", 
                               json=payload, timeout=0.1)
        if response.status_code == 200:
            last_action_sent = (x, y)
            action_count += 1
            print(f"Action #{action_count} sent: ({x:.3f}, {y:.3f})")
        else:
            print(f"Failed to send action: {response.status_code}")
    except Exception as e:
        print(f"Error sending action: {e}")

# Flask应用
app = Flask(__name__)

@app.route('/api/positions', methods=['POST'])
def receive_data():
    global latest_data, request_stats
    
    current_time = time.time()
    
    try:
        data = request.get_json()
        if data:
            with data_lock:
                latest_data.update(data)
                latest_data["timestamp"] = current_time
                
                # 更新频率统计
                if request_stats["last_time"] > 0:
                    interval = current_time - request_stats["last_time"]
                    request_stats["intervals"].append(interval)
                    if len(request_stats["intervals"]) > 50:  # 只保留最近50个
                        request_stats["intervals"].pop(0)
                
                request_stats["last_time"] = current_time
                request_stats["count"] += 1
                
                # 每20个请求打印统计
                if request_stats["count"] % 20 == 0:
                    avg_interval = sum(request_stats["intervals"]) / len(request_stats["intervals"])
                    freq = 1.0 / avg_interval if avg_interval > 0 else 0
                    print(f"[频率] #{request_stats['count']}: {freq:.1f}Hz")
            
            return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"error": str(e)}), 400
    
    return jsonify({"error": "No data received"}), 400

def start_flask_server():
    print("Flask server starting on http://localhost:6000")
    app.run(host='0.0.0.0', port=6000, debug=False, threaded=True, use_reloader=False)

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
            
        # 绘制连接线
        connections = [(i, (i+1)%8) for i in range(8)]
        for i, j in connections:
            pygame.draw.line(surface, self.color, 
                           (int(self.points[i][0]), int(self.points[i][1])),
                           (int(self.points[j][0]), int(self.points[j][1])), 2)
        
        # 绘制关键点
        for x, y in self.points:
            pygame.draw.circle(surface, self.color, (int(x), int(y)), 4)
        
        # 显示标签
        if self.points:
            center_x = sum(p[0] for p in self.points) / len(self.points)
            center_y = sum(p[1] for p in self.points) / len(self.points)
            text = font.render(self.label, True, BLACK)
            surface.blit(text, (center_x + 10, center_y + 10))

class SimpleEEF:
    def __init__(self):
        self.pos = (300, 300)
        
    def update(self, agent_pos):
        if len(agent_pos) >= 2:
            self.pos = (agent_pos[0] * SCALE, agent_pos[1] * SCALE)
    
    def draw(self, surface):
        x, y = int(self.pos[0]), int(self.pos[1])
        pygame.draw.circle(surface, BLUE, (x, y), 6)
        # 十字标记
        pygame.draw.line(surface, BLUE, (x-10, y), (x+10, y), 2)
        pygame.draw.line(surface, BLUE, (x, y-10), (x, y+10), 2)

def draw_mouse_cursor(surface, pos):
    """绘制鼠标光标位置"""
    x, y = pos
    pygame.draw.circle(surface, YELLOW, (int(x), int(y)), 8, 2)
    pygame.draw.line(surface, YELLOW, (x-12, y), (x+12, y), 2)
    pygame.draw.line(surface, YELLOW, (x, y-12), (x, y+12), 2)

t_current = SimpleT(RED, "Current")
t_target = SimpleT(GREEN, "Target")
eef = SimpleEEF()

flask_thread = threading.Thread(target=start_flask_server, daemon=True)
flask_thread.start()

running = True
last_update = 0
UPDATE_INTERVAL = 0.05

print("Pygame visualization started. Press ESC to quit, S for stats.")
print("Press SPACE to start/stop sending actions at mouse position")

while running:
    current_time = time.time()
    
    # 在主线程中更新鼠标位置
    current_mouse_pos = pygame.mouse.get_pos()
    
    # 事件处理
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            elif event.key == K_SPACE:
                # 切换action发送状态
                action_sending = not action_sending
                if action_sending:
                    print("Started sending actions")
                else:
                    print("Stopped sending actions")
            elif event.key == K_s:  # 显示详细统计
                with data_lock:
                    stats = request_stats.copy()
                if stats["intervals"]:
                    avg_interval = sum(stats["intervals"]) / len(stats["intervals"])
                    freq = 1.0 / avg_interval
                    elapsed = current_time - stats["start_time"]
                    overall_freq = stats["count"] / elapsed
                    print(f"\n=== 统计 ===")
                    print(f"总请求: {stats['count']}")
                    print(f"当前频率: {freq:.1f} Hz")
                    print(f"总体频率: {overall_freq:.1f} Hz")
                    print(f"运行时间: {elapsed:.1f}s")
                    print(f"Action发送次数: {action_count}")
    
    # 自动发送action
    if action_sending and (current_time - last_action_time) > (1.0 / action_frequency):
        mouse_x, mouse_y = current_mouse_pos
        # 将屏幕坐标转换为归一化坐标
        action_x = mouse_x / SCALE
        action_y = mouse_y / SCALE
        
        # 限制坐标范围到合理区间
        action_x = max(0, min(1, action_x))
        action_y = max(0, min(1, action_y))
        
        # 在新线程中发送action，避免阻塞UI
        threading.Thread(target=send_action, args=(action_x, action_y), daemon=True).start()
        last_action_time = current_time
    
    if current_time - last_update > UPDATE_INTERVAL:
        with data_lock:
            data = latest_data.copy()
        
        if "cur_keypoints" in data:
            t_current.update(data["cur_keypoints"])
        if "target_keypoints" in data:
            t_target.update(data["target_keypoints"])
        if "agent_pos" in data:
            eef.update(data["agent_pos"])
            
        last_update = current_time
    
    screen.fill(WHITE)
    t_current.draw(screen)
    t_target.draw(screen)
    eef.draw(screen)
    
    # 绘制鼠标光标位置
    draw_mouse_cursor(screen, current_mouse_pos)
    
    with data_lock:
        data = latest_data.copy()
        stats = request_stats.copy()
    
    current_freq = 0
    if stats["intervals"]:
        avg_interval = sum(stats["intervals"]) / len(stats["intervals"])
        current_freq = 1.0 / avg_interval if avg_interval > 0 else 0
    
    elapsed_time = current_time - stats["start_time"]
    overall_freq = stats["count"] / elapsed_time if elapsed_time > 0 else 0
    
    # 状态信息
    status_lines = [
        f"FPS: {int(clock.get_fps())}",
        f"Requests: {stats['count']}",
        f"Frequency: {current_freq:.1f} Hz",
        f"Overall: {overall_freq:.1f} Hz",
        f"Intersection: {data.get('intersection_ratio', 0):.3f}",
        f"Reward: {data.get('reward', 0):.3f}",
        "",
        f"Action sending: {'ON' if action_sending else 'OFF'}",
        f"Action rate: {action_frequency} Hz",
        f"Action count: {action_count}",
        f"Mouse pos: ({current_mouse_pos[0]}, {current_mouse_pos[1]})",
        f"Action pos: ({current_mouse_pos[0]/SCALE:.3f}, {current_mouse_pos[1]/SCALE:.3f})",
        f"Last action: ({last_action_sent[0]:.3f}, {last_action_sent[1]:.3f})",
        "",
        "Press SPACE to start/stop actions",
        "Press S for detailed stats",
        "Press ESC to quit"
    ]
    
    for i, line in enumerate(status_lines):
        text = font.render(line, True, BLACK)
        screen.blit(text, (10, 10 + i * 18))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
print("Visualization stopped.")