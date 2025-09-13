import pygame
import time
import threading
import requests
import numpy as np
import base64
import cv2
import sys
import os
from pathlib import Path
from pygame.locals import *
from flask import Flask, request, jsonify

# sys.path.append('/Users/ding/Desktop/DesktopAir/Research/RoboArm/proj_gene')
from diffusion_policy.common.replay_buffer import ReplayBuffer

pygame.init()
width, height = 1000, 800
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 16)

WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 50, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
SCALE = 800

latest_data = {
    "cur_keypoints": [],
    "target_keypoints": [],
    "agent_pos": [0.3, 0.3],
    "intersection_ratio": 0.0,
    "reward": 0.0,
    "image": None
}
data_lock = threading.Lock()

class SmartRecordingController:
    def __init__(self):
        self.state = 'IDLE'  # IDLE, WAITING, RECORDING, COMPLETING
        self.movement_threshold = 0.02
        self.success_threshold = 0.85
        self.last_mouse_pos = None
        
    def should_start_recording(self, mouse_pos):
        if self.state != 'WAITING':
            return False
        if self.last_mouse_pos is None:
            self.last_mouse_pos = mouse_pos
            return False
        movement = np.linalg.norm(np.array(mouse_pos) - np.array(self.last_mouse_pos))
        return movement > self.movement_threshold
    
    def should_end_episode(self, intersection_ratio):
        return self.state == 'RECORDING' and intersection_ratio > self.success_threshold

class DataCollector:
    def __init__(self):
        self.episodes = []
        self.current_episode = []
        self.last_agent_pos = np.array([0.3, 0.3])
        
    def decode_image(self, img_data):
        if not img_data or img_data.get('format') != 'jpeg_base64':
            return None
        try:
            img_bytes = base64.b64decode(img_data['data'])
            img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            return cv2.resize(img_array, (96, 96)).astype(np.uint8)
        except:
            return None
    
    def record_timestep(self, obs_data, mouse_pos):
        image = self.decode_image(obs_data.get('image'))
        if image is None:
            return False
            
        current_agent_pos = np.array(obs_data.get('agent_pos', [0.3, 0.3])[:2], dtype=np.float32)
        target_pos = np.array([mouse_pos[0]/SCALE, mouse_pos[1]/SCALE], dtype=np.float32)
        
        # Position delta action
        action = target_pos - self.last_agent_pos
        
        # Combined state (compatible with original format)
        state = np.concatenate([
            current_agent_pos,  # [0:2] agent_pos for compatibility
            [obs_data.get('intersection_ratio', 0.0)],
            [obs_data.get('reward', 0.0)]
        ]).astype(np.float32)
        
        timestep = {
            'img': image,
            'state': state,
            'action': action,
            'cur_keypoints': np.array(obs_data.get('cur_keypoints', []), dtype=np.float32),
            'target_keypoints': np.array(obs_data.get('target_keypoints', []), dtype=np.float32)
        }
        
        self.current_episode.append(timestep)
        self.last_agent_pos = current_agent_pos
        return True
    
    def finish_episode(self):
        if not self.current_episode:
            return False
        
        episode_data = {}
        for key in self.current_episode[0].keys():
            episode_data[key] = np.stack([step[key] for step in self.current_episode])
        
        self.episodes.append(episode_data)
        self.current_episode = []
        return True
    
    def save_to_zarr(self, filepath):
        if not self.episodes:
            return False
        
        try:
            import zarr
            # Create ReplayBuffer using zarr backend for file storage
            store = zarr.DirectoryStore(filepath)
            replay_buffer = ReplayBuffer.create_empty_zarr(storage=store)
            
            # Define optimal compressors for each data type
            compressors = {
                'img': 'disk',      # zstd compression for images
                'state': 'default', # lz4 compression for state vectors  
                'action': 'default', # lz4 compression for actions
                'cur_keypoints': 'default', # lz4 compression for keypoints
                'target_keypoints': 'default' # lz4 compression for keypoints
            }
            
            # Define optimal chunks for each data type based on data shapes
            chunks = {
                'img': (32, 96, 96, 3),        # 32 timesteps per chunk for images (96x96x3)
                'state': (512, 4),              # 512 timesteps per chunk for state (4 values)
                'action': (512, 2),             # 512 timesteps per chunk for action (2D position)
                'cur_keypoints': (128, 8, 2),   # 128 timesteps per chunk for keypoints (8 points, 2D)
                'target_keypoints': (128, 8, 2) # 128 timesteps per chunk for keypoints (8 points, 2D)
            }
            
            # Add all episodes to replay buffer with proper chunking and compression
            for episode_idx, episode_data in enumerate(self.episodes):
                print(f"Adding episode {episode_idx + 1}/{len(self.episodes)}")
                replay_buffer.add_episode(
                    data=episode_data, 
                    chunks=chunks,
                    compressors=compressors
                )
            
            print(f"Successfully saved {len(self.episodes)} episodes to {filepath}")
            print(f"Total timesteps: {replay_buffer.n_steps}")
            
            # Print data structure info
            print("📁 Data structure:")
            for key in replay_buffer.data.keys():
                arr = replay_buffer.data[key]
                print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
            
            return True
            
        except Exception as e:
            print(f"Error saving to zarr: {e}")
            import traceback
            traceback.print_exc()
            return False

recording_controller = SmartRecordingController()
data_collector = DataCollector()
action_sending = False
current_mouse_pos = (400, 400)

def send_action(x, y):
    try:
        payload = {"action": [x, y]}
        response = requests.post("http://localhost:7100/api/action", 
                               json=payload, timeout=0.1)
        return response.status_code == 200
    except:
        return False

app = Flask(__name__)

@app.route('/api/positions', methods=['POST'])
def receive_data():
    global latest_data
    try:
        data = request.get_json()
        if data:
            with data_lock:
                latest_data.update(data)
            return jsonify({"status": "success"}), 200
    except:
        pass
    return jsonify({"error": "failed"}), 400

def start_flask_server():
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
        for i in range(8):
            j = (i + 1) % 8
            pygame.draw.line(surface, self.color, 
                           (int(self.points[i][0]), int(self.points[i][1])),
                           (int(self.points[j][0]), int(self.points[j][1])), 2)
        for x, y in self.points:
            pygame.draw.circle(surface, self.color, (int(x), int(y)), 3)

class SimpleEEF:
    def __init__(self):
        self.pos = (300, 300)
        
    def update(self, agent_pos):
        if len(agent_pos) >= 2:
            self.pos = (agent_pos[0] * SCALE, agent_pos[1] * SCALE)
    
    def draw(self, surface):
        x, y = int(self.pos[0]), int(self.pos[1])
        pygame.draw.circle(surface, BLUE, (x, y), 6)
        pygame.draw.line(surface, BLUE, (x-8, y), (x+8, y), 2)
        pygame.draw.line(surface, BLUE, (x, y-8), (x, y+8), 2)

t_current = SimpleT(RED, "Current")
t_target = SimpleT(GREEN, "Target")  
eef = SimpleEEF()

flask_thread = threading.Thread(target=start_flask_server, daemon=True)
flask_thread.start()

running = True
last_action_time = 0
action_frequency = 10

print("Genesis Data Collector")
print("SPACE: Start recording session")
print("R: Force new episode") 
print("S: Save and exit")
print("ESC: Exit without saving")

while running:
    current_time = time.time()
    current_mouse_pos = pygame.mouse.get_pos()
    
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            elif event.key == K_SPACE:
                if recording_controller.state == 'IDLE':
                    recording_controller.state = 'WAITING'
                    recording_controller.last_mouse_pos = current_mouse_pos
                    print("Waiting for movement to start recording...")
                elif recording_controller.state == 'WAITING':
                    recording_controller.state = 'IDLE'
                    print("Recording cancelled")
                elif recording_controller.state == 'RECORDING':
                    data_collector.finish_episode()
                    recording_controller.state = 'WAITING'
                    recording_controller.last_mouse_pos = current_mouse_pos
                    print("Episode finished, waiting for next...")
            elif event.key == K_r:
                if recording_controller.state == 'RECORDING':
                    data_collector.finish_episode()
                    print("Forced episode finish")
                recording_controller.state = 'RECORDING' 
                print("Force started recording")
            elif event.key == K_s:
                if recording_controller.state == 'RECORDING':
                    data_collector.finish_episode()
                
                output_dir = Path("data/genesis_collected")
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filepath = output_dir / f"genesis_data_{timestamp}.zarr"
                
                if data_collector.save_to_zarr(str(filepath)):
                    print(f"Saved {len(data_collector.episodes)} episodes to {filepath}")
                else:
                    print("Save failed")
                running = False
    
    # Auto action sending
    if action_sending and (current_time - last_action_time) > (1.0 / action_frequency):
        action_x = max(0, min(1, current_mouse_pos[0] / SCALE))
        action_y = max(0, min(1, current_mouse_pos[1] / SCALE))
        threading.Thread(target=send_action, args=(action_x, action_y), daemon=True).start()
        last_action_time = current_time
    
    # Recording logic
    with data_lock:
        data = latest_data.copy()
    
    if recording_controller.should_start_recording(current_mouse_pos):
        recording_controller.state = 'RECORDING'
        action_sending = True
        print("Auto-started recording")
    
    if recording_controller.should_end_episode(data.get('intersection_ratio', 0)):
        data_collector.finish_episode()
        recording_controller.state = 'WAITING'
        recording_controller.last_mouse_pos = current_mouse_pos
        action_sending = False
        print(f"Episode completed ({len(data_collector.episodes)} total)")
    
    if recording_controller.state == 'RECORDING' and action_sending:
        data_collector.record_timestep(data, current_mouse_pos)
    
    # Update display
    if "cur_keypoints" in data:
        t_current.update(data["cur_keypoints"])
    if "target_keypoints" in data:
        t_target.update(data["target_keypoints"])
    if "agent_pos" in data:
        eef.update(data["agent_pos"])
    
    screen.fill(WHITE)
    t_current.draw(screen)
    t_target.draw(screen)
    eef.draw(screen)
    
    # Mouse cursor
    pygame.draw.circle(screen, YELLOW, current_mouse_pos, 8, 2)
    
    # Status display
    status_lines = [
        f"State: {recording_controller.state}",
        f"Episodes: {len(data_collector.episodes)}",
        f"Current steps: {len(data_collector.current_episode)}",
        f"Intersection: {data.get('intersection_ratio', 0):.3f}",
        f"Action sending: {'ON' if action_sending else 'OFF'}",
        "",
        "SPACE: Start/Control recording",
        "R: Force new episode", 
        "S: Save and exit"
    ]
    
    for i, line in enumerate(status_lines):
        text = font.render(line, True, BLACK)
        screen.blit(text, (10, 10 + i * 20))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
print("Data collection stopped.")