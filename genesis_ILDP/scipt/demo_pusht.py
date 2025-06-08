import torch
from genesis_ILDP.utils.pytorch_util import *
from genesis_ILDP.env.pushT_env import PushTEnv
from genesis_ILDP.config.env_config import *
import time
import traceback
import numpy as np
import requests
import json

class KeyboardClient:
    def __init__(self, server_url="http://127.0.0.1:5000"):
        self.server_url = server_url
        
    def get_pressed_keys(self):
        try:
            response = requests.get(f"{self.server_url}/status", timeout=0.1)
            if response.status_code == 200:
                data = response.json()
                return data['pressed_keys'], data['should_exit']
            return [], False
        except Exception as e:
            print(f"获取按键状态失败: {e}")
            return [], False
    
    def get_recent_events(self):
        try:
            response = requests.get(f"{self.server_url}/events", timeout=0.1)
            if response.status_code == 200:
                return response.json()['events']
            return []
        except Exception as e:
            print(f"获取按键事件失败: {e}")
            return []
    
    def stop_server(self):
        try:
            requests.post(f"{self.server_url}/stop", timeout=1.0)
        except:
            pass

class PushTDemoEnv(PushTEnv):
    def __init__(self, render_size=(96, 96),
                xlim=0.5, ylim=0.5,
                seed=None, model_path=env_path, 
                fps=30, show_fps=True):
        super().__init__(render_size, xlim, ylim, 
                         seed, model_path, fps, show_fps)
        self.keyboard_client = KeyboardClient()
        self.demo_states = 'Resume'
        self.planned_pos = None
        print("[DEBUG] PushTDemoEnv 初始化完成")
        
    def update_demo_state(self, pressed_keys):
        if 'esc' in pressed_keys:
            self.demo_states = 'Stop'
        elif 'space' in pressed_keys:
            self.demo_states = 'Pause'
        elif 'backspace' in pressed_keys:
            self.demo_states = 'Resume'
    def reset(self, ):
        super().reset()
        self.planned_pos = self.robot.get_links_pos(self.eef_idx, torch.tensor([0], device=self.device))[0,:,:]
    def step(self):
        try:
            pressed_keys, should_exit = self.keyboard_client.get_pressed_keys()
            
            self.update_demo_state(pressed_keys)
            
            print(f"[DEBUG] 当前按键: {pressed_keys}")
            
            move_increment = 0.02 / 1.414
            
            if 'left' in pressed_keys or 'a' in pressed_keys:
                print("[DEBUG] 向左移动")
                self.planned_pos = self.planned_pos + torch.tensor([move_increment, -move_increment, 0], device=self.device)
                
            if 'right' in pressed_keys or 'd' in pressed_keys:
                print("[DEBUG] 向右移动") 
                self.planned_pos = self.planned_pos + torch.tensor([-move_increment, move_increment, 0], device=self.device)
                
            if 'up' in pressed_keys or 'w' in pressed_keys:
                print("[DEBUG] 向上移动")
                self.planned_pos = self.planned_pos + torch.tensor([-move_increment, -move_increment, 0], device=self.device)
                
            if 'down' in pressed_keys or 's' in pressed_keys:
                print("[DEBUG] 向下移动")
                self.planned_pos = self.planned_pos + torch.tensor([move_increment, move_increment, 0], device=self.device)

            print(f"[DEBUG] 最终 action: {self.planned_pos}")
            
            # 调用父类 step
            result = super().step(self.planned_pos, torch.tensor([0], device=self.device))
            return result
            
        except Exception as e:
            print(f"[ERROR] step 方法错误: {e}")
            traceback.print_exc()
            return None

def wait_for_keyboard_server(client, max_attempts=30):
    """等待键盘服务器启动"""
    print("等待键盘服务器启动...")
    for i in range(max_attempts):
        try:
            response = requests.get(f"{client.server_url}/status", timeout=1.0)
            if response.status_code == 200:
                print("键盘服务器已就绪")
                return True
        except:
            pass
        time.sleep(1)
        print(f"等待中... ({i+1}/{max_attempts})")
    
    print("键盘服务器启动超时")
    return False

def main():
    DemoEnv = None
    try:
        # 等待键盘服务器启动
        keyboard_client = KeyboardClient()
        if not wait_for_keyboard_server(keyboard_client):
            print("请先运行 python keyboard_server.py")
            return
        
        print("[DEBUG] 创建 DemoEnv...")
        DemoEnv = PushTDemoEnv()
        
        print("[DEBUG] 启动环境...")
        DemoEnv.start(show_interact_viewer=False)
        DemoEnv.seed(np.arange(1))
        
        print("[DEBUG] 重置环境...")
        DemoEnv.reset()
        
        print("[DEBUG] 开始主循环...")
        print("控制说明: WASD或方向键移动, ESC退出, Space暂停, Backspace恢复")
        Time = time.time()
        step_count = 0
        while True:
            step_count += 1
            
            if DemoEnv.demo_states == 'Stop': 
                print("[DEBUG] 收到停止信号")
                break 
            if DemoEnv.demo_states == 'Pause': 
                print("[DEBUG] 暂停中...")
                time.sleep(0.1)
                continue

            result = DemoEnv.step()
            print(time.time()-Time)
            Time = time.time()
            if result is None:
                break
                            
    except KeyboardInterrupt:
        print("[INFO] 程序被用户中断")
    except Exception as e:
        print("[ERROR] 主函数发生错误: {e}")
        traceback.print_exc()
    finally:
        print("[DEBUG] 进入清理阶段...")
        if DemoEnv:
            DemoEnv.keyboard_client.stop_server()
        print("[DEBUG] 程序结束")

if __name__ == '__main__':
    main()