import time
import os
import sys

import gym
import torch
from gym import spaces
import numpy as np

import cv2
import pickle
import genesis as gs
import glfw
import requests
import json
import threading
from flask import Flask, request, jsonify

from genesis_ILDP.utils.cuda import *
from genesis_ILDP.config.env_config import *
from shapely.geometry import Polygon
from genesis_ILDP.env.pushT_env import PushTEnv
global reset_requested
class PushTEnvServer(PushTEnv):
    def __init__(self, render_size=(96, 96), xlim=0.1, ylim=0.1, seed=None, 
                 model_path=env_path, fps=30, show_fps=True):
        super().__init__(
            render_size=render_size, 
            xlim=xlim, 
            ylim=ylim, 
            seed=seed,
            model_path=model_path, 
            fps=fps, 
            show_fps=show_fps
        )

    def _publish_keypoints(self, envs_idx=None):
        
       
        if envs_idx is None:
            envs_idx = torch.arange(1, device=gs.device, dtype=torch.int16)
        
        if self.poses is None:
            self._get_poses(envs_idx)
        if self.keypoints is None:
            self.calculate_all_keypoints()
        if self.render_cache is None:
            self._get_obs(rgb=True, envs_idx=envs_idx)
        
        
        data = {
            "timestamp": time.time(),
            # "env_id": 0,
            
            "cur_keypoints": self.keypoints['cur_keypoints'][0,:,:2].tolist(),
            
            "target_keypoints": self.keypoints['target_keypoints'][0,:, :2].tolist(),
            
            "cur_pose": {
                "position": to_numpy(self.poses['cur_Tpose'][0, :2]).tolist(),  # [x, y, z]
                "angle": float(to_numpy(self.poses['cur_Tpose'][0, 5]))  # z轴旋转角度
            },
            
            "target_pose": {
                "position": to_numpy(self.poses['target_Tpose'][0, :2]).tolist(),  # [x, y, z]
                "angle": float(to_numpy(self.poses['target_Tpose'][0, 5]))  # z轴旋转角度
            },
            
            "agent_pos": to_numpy(self.poses['agent_pos'][0, :2]).tolist(),  # [x, y]
            
            "image": self._encode_image(self.render_cache[0][0]) if self.render_cache is not None else None,
            
            "intersection_ratio": float(self._cal_intersection()[0]),
            "reward": float(self._cal_rewards()[0]),
        }
        
        try:
            # 发送到pygame接收器
            API_URL = "http://localhost:6000/api/positions"
            response = requests.post(API_URL, 
                                json=data, 
                                headers={'Content-Type': 'application/json'},
                                timeout=0.1)  # 短超时，避免阻塞仿真
            
            if response.status_code != 200:
                print(f"Warning: Failed to send data to pygame. Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not connect to pygame receiver: {e}")
        except Exception as e:
            print(f"Error in _publish_keypoints: {e}")


    def _encode_image(self, img_array):
        try:
            # 确保图像数据格式正确
            img_np = np.array(img_array)
            
            # 转换为 (H, W, 3) 用于编码
            if len(img_np.shape) == 3 and img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            
            import base64
            _, buffer = cv2.imencode('.jpg', img_np)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "format": "jpeg_base64",
                "data": img_base64,
                "shape": img_np.shape
            }
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
        

latest_action = {
    "action": [0.0, 0.0],
    "timestamp": time.time(),
    "received": False
}
action_lock = threading.Lock()
reset_requested = False

app = Flask(__name__)

@app.route('/api/action', methods=['POST'])
def receive_action():
    global latest_action
    
    print(f"Received request: {request.method} {request.url}")
    print(f"Content-Type: {request.content_type}")
    print(f"Raw data: {request.get_data()}")
    
    try:
        data = request.get_json()
        print(f"Parsed JSON data: {data}")
        
        if data and "action" in data:
            with action_lock:
                latest_action["action"] = data["action"]
                latest_action["timestamp"] = data['timestamp']
                latest_action["received"] = True
                
            # print(f"Action received and stored: {data['action']}")
            return jsonify({"status": "success"}), 200
        else:
            print("No action data in request")
            return jsonify({"error": "No action data"}), 400
            
    except Exception as e:
        print(f"Error receiving action: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": time.time()}), 200

@app.route('/api/status', methods=['GET'])
def get_status():
    with action_lock:
        return jsonify({
            "latest_action": latest_action,
            "server_time": time.time()
        }), 200
    

# @app.route('/api/received_action', methods=['POST'])
# def publish_received_action(latest_action):
#     return jsonify({
#         # "action": latest_action["action"], # it is enough to use action for tracing back to its real action and obs
#         "timestamp": latest_action["timestamp"]
#     }, 200)


@app.route('/api/reset', methods=['POST'])
def reset_environment():
    try:
        # Reset the environment - this will be set by the main loop
        global reset_requested
        reset_requested = True
        return jsonify({"status": "reset_requested", "timestamp": time.time()}), 200
    except Exception as e:
        print(f"Error in reset endpoint: {e}")
        return jsonify({"error": str(e)}), 500

def get_latest_action():
    with action_lock:
        if latest_action["received"]:
            action = latest_action["action"].copy()
            latest_action["received"] = False  # 标记已读取
            # print(f"Retrieved action: {action}")
            return np.array(action)
    return None

def start_flask_server():
    print("Flask server starting on http://localhost:7100")
    app.run(host='0.0.0.0', port=7100, debug=False, threaded=True, use_reloader=False)

if __name__ == '__main__':
    flask_thread = threading.Thread(target=start_flask_server, daemon=True)
    flask_thread.start()
    
    time.sleep(1)
    print("Flask server should be running now")
    
    env = PushTEnvServer(show_fps=False)
    env.start(n_envs=1, show_camera=False, show_interact_viewer=False, env_separate=False, seed=[0])
    env.seed(np.arange(1))
    env.reset()
    action_count = 0
    
    env.start_recording()
    for i in range(10000):

        current_time = time.time()
        # Check if reset is requested
        if reset_requested:
            print("Reset requested, resetting environment...")
            env.reset()
            reset_requested = False
            print("Environment reset completed")
            action = get_latest_action() # eat up the last action 
        # TODO change the logic of grasping the newest action into grasping 
        # the newest action and also check if the action is enough new to be executed
        action = get_latest_action()

        if action is not None:
            action_count += 1

            # Notify client that this action was executed
            try:
                requests.post("http://localhost:6000/api/received_action",
                             json={"timestamp": latest_action["timestamp"]},
                             timeout=0.1)
            except:
                pass  # Don't block if client not available

            action_tensor = torch.tensor([[*action, 0.2]], dtype=torch.float32)
            print(f"Step {i}: Executing action {action_count}: {action}")
            print(action_tensor)
            env.step(action=action_tensor)
        else:
            env.step()

        # Control simulation speed for time alignment with control signal
        env._publish_keypoints()
        finishing_time = time.time()
        executing_time = finishing_time - current_time
        time_to_wait = 0.1 - executing_time

        if i % 50 == 0:
            print(f"Step {i}: Total actions received: {action_count}")

        if(time_to_wait > 0):
            time.sleep(time_to_wait)
               
    env.stop_recording()