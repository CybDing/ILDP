from flask import Flask, jsonify
from pynput import keyboard
import threading
import time
from datetime import datetime

class KeyboardServer:
    def __init__(self, port=5000):
        self.app = Flask(__name__)
        self.port = port
        self.key_states = {}  # 当前按下的键
        self.key_events = []  # 最近的按键事件
        self.max_events = 100  # 保留最近100个事件
        self.should_exit = False
        self.listener = None
        
        self._setup_routes()
        
    def _setup_routes(self):
        @self.app.route('/status', methods=['GET'])
        def get_status():
            """获取当前按键状态"""
            return jsonify({
                'pressed_keys': list(self.key_states.keys()),
                'should_exit': self.should_exit,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/events', methods=['GET'])
        def get_events():
            """获取最近的按键事件"""
            return jsonify({
                'events': self.key_events[-10:],  # 返回最近10个事件
                'total_events': len(self.key_events)
            })
        
        @self.app.route('/clear', methods=['POST'])
        def clear_events():
            """清空事件历史"""
            self.key_events.clear()
            return jsonify({'message': 'Events cleared'})
        
        @self.app.route('/stop', methods=['POST'])
        def stop_server():
            """停止服务器"""
            self.should_exit = True
            return jsonify({'message': 'Server stopping'})
    
    def _key_to_string(self, key):
        """将按键转换为字符串"""
        if hasattr(key, 'char') and key.char:
            return key.char
        else:
            return str(key).replace('Key.', '')
    
    def start_keyboard_listener(self):
        """启动键盘监听器"""
        def on_press(key):
            try:
                key_str = self._key_to_string(key)
                self.key_states[key_str] = True
                
                event = {
                    'type': 'press',
                    'key': key_str,
                    'timestamp': datetime.now().isoformat()
                }
                self.key_events.append(event)
                
                # 保持事件列表大小
                if len(self.key_events) > self.max_events:
                    self.key_events.pop(0)
                
                print(f"按下: {key_str}")
                
                # ESC 键设置退出标志
                if key == keyboard.Key.esc:
                    self.should_exit = True
                    
            except Exception as e:
                print(f"按键处理错误: {e}")
        
        def on_release(key):
            try:
                key_str = self._key_to_string(key)
                if key_str in self.key_states:
                    del self.key_states[key_str]
                
                event = {
                    'type': 'release',
                    'key': key_str,
                    'timestamp': datetime.now().isoformat()
                }
                self.key_events.append(event)
                
                if len(self.key_events) > self.max_events:
                    self.key_events.pop(0)
                
                print(f"释放: {key_str}")
                
            except Exception as e:
                print(f"按键释放处理错误: {e}")
        
        self.listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )
        self.listener.start()
        print("键盘监听器已启动")
    
    def run(self):
        """运行服务器"""
        print(f"启动键盘监听服务器，端口: {self.port}")
        print("支持的按键: WASD, 方向键, ESC退出, Space暂停, Backspace恢复")
        
        # 在单独线程中启动键盘监听
        keyboard_thread = threading.Thread(target=self.start_keyboard_listener)
        keyboard_thread.daemon = True
        keyboard_thread.start()
        
        try:
            # 启动Flask服务器
            self.app.run(host='127.0.0.1', port=self.port, debug=False)
        except KeyboardInterrupt:
            print("服务器被中断")
        finally:
            if self.listener:
                self.listener.stop()
            print("键盘监听服务器已停止")

if __name__ == '__main__':
    server = KeyboardServer(port=5000)
    server.run()