# 本程序由Lobo开发，直接使用cursor开发，直接以来大语言模型编写。程序可能会有一些奇怪的逻辑，请见谅。
# 2025-03-26 

# 使用方法：
# 1. 安装依赖：pip install PyQt6 PyQt6-WebEngine requests markdown
# 2. 运行：python OllamaAIChatTool.py

# 注意：
# 1. 请确保Ollama服务已启动（运行 'ollama serve'）
# 2. 请点击\"选择模型\"按钮选择一个已安装的模型

import sys
import requests
import markdown
import time
import json
import logging
from typing import List, Dict, Optional, Any
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTextEdit, QPushButton, QLabel, QMessageBox,
                            QComboBox, QDialog, QDialogButtonBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtGui import QIcon

# 常量定义
DEFAULT_SERVER_URL = "http://localhost:11434"
DEFAULT_WINDOW_WIDTH = 800
DEFAULT_WINDOW_HEIGHT = 600
DEFAULT_WINDOW_X = 100
DEFAULT_WINDOW_Y = 100
INPUT_MAX_HEIGHT = 100
SEND_BUTTON_MIN_WIDTH = 150
MODELS_DIALOG_MIN_WIDTH = 400
SCROLL_DELAY = 500  # 滚动延迟时间（毫秒）

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelsDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None, server_url: str = DEFAULT_SERVER_URL) -> None:
        super().__init__(parent)
        self.server_url = server_url
        self.setWindowTitle("选择模型")
        self.setMinimumWidth(MODELS_DIALOG_MIN_WIDTH)
        
        # 设置对话框图标
        self.setWindowIcon(QIcon("ollamaICO.png"))
        
        self._init_ui()
        self.load_models()
        
    def _init_ui(self) -> None:
        """初始化UI组件"""
        layout = QVBoxLayout(self)
        
        label = QLabel("请选择一个模型:")
        layout.addWidget(label)
        
        self.model_combo = QComboBox()
        self.model_combo.addItem("正在加载模型列表...", "")
        layout.addWidget(self.model_combo)
        
        self.refresh_btn = QPushButton("刷新模型列表")
        self.refresh_btn.clicked.connect(self.load_models)
        layout.addWidget(self.refresh_btn)
        
        self.info_label = QLabel("")
        layout.addWidget(self.info_label)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def load_models(self) -> None:
        """加载可用的模型列表"""
        try:
            self.info_label.setText("正在加载模型列表...")
            self.model_combo.clear()
            self.model_combo.addItem("正在加载...", "")
            
            response = requests.get(f'{self.server_url}/api/tags')
            response.raise_for_status()  # 抛出HTTP错误
            
            models = response.json().get("models", [])
            self.model_combo.clear()
            
            if not models:
                self.model_combo.addItem("未找到模型", "")
                self.info_label.setText("未找到任何可用模型，请先使用 'ollama pull' 命令安装模型")
                return
                
            for model in models:
                self.model_combo.addItem(
                    f"{model['name']} ({model.get('size', '未知大小')})", 
                    model['name']
                )
            self.info_label.setText(f"找到 {len(models)} 个模型")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"加载模型列表时出错: {str(e)}")
            self.model_combo.clear()
            self.model_combo.addItem("连接错误", "")
            self.info_label.setText(f"连接 Ollama 服务时出错: {str(e)}")
            
    def get_selected_model(self) -> str:
        """获取选中的模型名称"""
        return self.model_combo.currentData()

class OllamaThread(QThread):
    response_ready = pyqtSignal(str)
    response_chunk = pyqtSignal(str)
    response_complete = pyqtSignal(str, float)
    
    def __init__(self, prompt: str, model: str, server_url: str) -> None:
        super().__init__()
        self.prompt = prompt
        self.model = model
        self.server_url = server_url
        self.start_time: Optional[float] = None
        
    def run(self) -> None:
        try:
            # 检查服务可用性
            self._check_service()
            
            # 发送生成请求
            self._send_generate_request()
            
        except Exception as e:
            logger.error(f"生成响应时出错: {str(e)}")
            self.response_ready.emit(f"错误：{str(e)}\n请确保已安装所需的模型（使用 'ollama pull {self.model}' 命令）")
            
    def _check_service(self) -> None:
        """检查Ollama服务是否可用"""
        try:
            response = requests.get(f'{self.server_url}/api/tags')
            if response.status_code != 200:
                raise Exception("Ollama服务未正常运行，请确保已启动ollama serve命令")
        except requests.exceptions.ConnectionError:
            raise Exception("无法连接到Ollama服务，请确保已启动ollama serve命令")
            
    def _send_generate_request(self) -> None:
        """发送生成请求并处理响应"""
        self.start_time = time.time()
        response = requests.post(
            f'{self.server_url}/api/generate',
            json={
                'model': self.model,
                'prompt': self.prompt,
                'stream': True
            },
            stream=True
        )
        
        if response.status_code != 200:
            self._handle_error_response(response)
            return
            
        self._process_response_stream(response)
        
    def _process_response_stream(self, response: requests.Response) -> None:
        """处理流式响应"""
        full_response = ""
        for line in response.iter_lines():
            if not line:
                continue
                
            try:
                chunk = line.decode('utf-8')
                if chunk.startswith('data: '):
                    chunk = chunk[6:]
                if chunk == '[DONE]':
                    break
                    
                response_data = json.loads(chunk)
                if 'response' in response_data:
                    chunk_text = response_data['response']
                    full_response += chunk_text
                    self.response_chunk.emit(chunk_text)
                    
            except Exception as e:
                logger.error(f"处理响应块时出错: {str(e)}")
                continue
                
        elapsed_time = time.time() - self.start_time
        self.response_complete.emit(full_response, elapsed_time)
        
    def _handle_error_response(self, response: requests.Response) -> None:
        """处理错误响应"""
        error_msg = f"错误：服务器返回状态码 {response.status_code}"
        try:
            error_detail = response.json().get('error', '未知错误')
            error_msg += f"\n详细信息：{error_detail}"
            
            if "model not found" in error_detail.lower():
                error_msg += f"\n\n请使用 'ollama pull {self.model}' 命令来安装此模型"
        except:
            pass
        self.response_ready.emit(error_msg)

class ChatWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Ollama AI Chat Tool")
        self.setGeometry(DEFAULT_WINDOW_X, DEFAULT_WINDOW_Y, 
                        DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        
        # 设置窗口图标
        self.setWindowIcon(QIcon("ollamaICO.png"))
        
        # 初始化属性
        self.current_model: str = ""
        self.current_response: str = ""
        self.current_response_start_time: Optional[float] = None
        self.server_url: str = DEFAULT_SERVER_URL
        self.chat_history: List[str] = []
        self.ollama_thread: Optional[OllamaThread] = None
        
        self._init_ui()
        self.show_startup_message()
        
    def _init_ui(self) -> None:
        """初始化UI组件"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QVBoxLayout(main_widget)
        
        # 创建顶部控制区
        self._init_top_controls(layout)
        
        # 创建聊天显示区域
        self._init_chat_display(layout)
        
        # 创建输入区域
        self._init_input_area(layout)
        
    def _init_top_controls(self, parent_layout: QVBoxLayout) -> None:
        """初始化顶部控制区域"""
        top_layout = QVBoxLayout()
        
        # 服务器地址输入
        server_layout = QHBoxLayout()
        server_label = QLabel("服务器地址:")
        server_layout.addWidget(server_label)
        
        self.server_input = QTextEdit()
        self.server_input.setMaximumHeight(30)
        self.server_input.setPlaceholderText("输入Ollama服务器地址...")
        self.server_input.setText(self.server_url)
        server_layout.addWidget(self.server_input)
        
        top_layout.addLayout(server_layout)
        
        # 模型选择器
        model_layout = QHBoxLayout()
        self.model_label = QLabel("当前模型:")
        model_layout.addWidget(self.model_label)
        
        self.model_button = QPushButton("选择模型")
        self.model_button.clicked.connect(self.select_model)
        model_layout.addWidget(self.model_button)
        
        model_layout.addStretch(1)
        
        top_layout.addLayout(model_layout)
        parent_layout.addLayout(top_layout)
        
    def _init_chat_display(self, parent_layout: QVBoxLayout) -> None:
        """初始化聊天显示区域"""
        self.chat_display = QWebEngineView()
        self.chat_display.setHtml(self.get_initial_html())
        parent_layout.addWidget(self.chat_display)
        
    def _init_input_area(self, parent_layout: QVBoxLayout) -> None:
        """初始化输入区域"""
        input_layout = QHBoxLayout()
        
        self.input_field = QTextEdit()
        self.input_field.setMaximumHeight(INPUT_MAX_HEIGHT)
        self.input_field.setPlaceholderText("输入你的问题...")
        self.input_field.keyPressEvent = self.handle_key_press
        input_layout.addWidget(self.input_field)
        
        self.send_button = QPushButton("发送")
        self.send_button.setMinimumHeight(INPUT_MAX_HEIGHT)
        self.send_button.setMinimumWidth(SEND_BUTTON_MIN_WIDTH)
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        
        parent_layout.addLayout(input_layout)
        
    def get_server_url(self) -> str:
        """获取用户输入的服务器地址"""
        url = self.server_input.toPlainText().strip()
        return url.rstrip('/') if url else DEFAULT_SERVER_URL
        
    def select_model(self) -> None:
        """打开模型选择对话框"""
        dialog = ModelsDialog(self, self.get_server_url())
        if dialog.exec():
            selected_model = dialog.get_selected_model()
            if selected_model:
                self.current_model = selected_model
                self.model_label.setText(f"当前模型: {self.current_model}")
                self.add_system_message(f"已选择模型: {self.current_model}")
        
    def add_system_message(self, message: str) -> None:
        """添加系统消息"""
        self.chat_history.append(f"**System:** {message}")
        self.update_chat_display()
        
    def show_startup_message(self) -> None:
        """显示启动消息"""
        self.chat_history.append("**System:** 欢迎使用AI聊天助手！\n\n使用前请确保：\n1. Ollama服务已启动（运行 'ollama serve'）\n2. 请点击\"选择模型\"按钮选择一个已安装的模型")
        self.update_chat_display()
        
        # 等待页面加载完成后再执行滚动
        self.chat_display.loadFinished.connect(
            lambda: self.chat_display.page().runJavaScript("scrollToBottom();")
        )
        
    def get_initial_html(self) -> str:
        """获取初始HTML模板"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                html {
                    height: 100%;
                    scroll-behavior: smooth;
                }
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    line-height: 1.6;
                    background-color: #1a1a1a;
                    color: #ffffff;
                    min-height: 100%;
                    display: flex;
                    flex-direction: column;
                }
                .message {
                    margin-bottom: 15px;
                    padding: 10px;
                    border-radius: 5px;
                }
                .user-message {
                    background-color: #2b2b2b;
                    margin-left: 20%;
                    border: 1px solid #3b3b3b;
                }
                .ai-message {
                    background-color: #2b2b2b;
                    margin-right: 20%;
                    border: 1px solid #3b3b3b;
                }
                .system-message {
                    background-color: #2b2b2b;
                    margin: 10px 0;
                    padding: 10px;
                    border-radius: 5px;
                    border: 1px solid #3b3b3b;
                }
                .time-info {
                    font-size: 0.8em;
                    color: #888;
                    margin-top: 5px;
                    text-align: right;
                }
                code {
                    background-color: #2d2d2d;
                    padding: 2px 5px;
                    border-radius: 3px;
                    border: 1px solid #3b3b3b;
                }
                pre {
                    background-color: #2d2d2d;
                    padding: 10px;
                    border-radius: 5px;
                    overflow-x: auto;
                    border: 1px solid #3b3b3b;
                }
                a {
                    color: #66b3ff;
                }
                blockquote {
                    border-left: 4px solid #3b3b3b;
                    margin: 0;
                    padding-left: 10px;
                    color: #cccccc;
                }
                #chat-container {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    gap: 15px;
                    margin-bottom: 20px;
                }
                #scroll-anchor {
                    height: 1px;
                    width: 100%;
                }
                .spacer {
                    flex-grow: 1;
                }
            </style>
            <script>
                function scrollToBottom() {
                    window.scrollTo({
                        top: document.body.scrollHeight,
                        behavior: 'smooth'
                    });
                }
                // 页面加载完成后执行滚动
                document.addEventListener('DOMContentLoaded', function() {
                    scrollToBottom();
                });
            </script>
        </head>
        <body>
            <div id="chat-container"></div>
            <div id="scroll-anchor"></div>
        </body>
        </html>
        """
        
    def send_message(self) -> None:
        """发送用户消息并获取AI响应"""
        user_message = self.input_field.toPlainText().strip()
        if not user_message:
            return
            
        if not self.current_model:
            self.add_system_message("请先选择一个模型")
            return
            
        # 添加用户消息到历史
        self.chat_history.append(f"**You:** {user_message}")
        self.update_chat_display()
        
        # 清空输入框
        self.input_field.clear()
        
        # 创建并启动Ollama线程
        self.ollama_thread = OllamaThread(user_message, self.current_model, self.get_server_url())
        self.ollama_thread.response_chunk.connect(self.handle_response_chunk)
        self.ollama_thread.response_complete.connect(self.handle_response_complete)
        self.ollama_thread.response_ready.connect(self.handle_error)
        self.ollama_thread.start()
        
        # 记录开始时间
        self.current_response_start_time = time.time()
        self.current_response = ""
        
    def handle_response_chunk(self, chunk: str) -> None:
        """处理AI响应的文本块"""
        if not self.chat_history or not self.chat_history[-1].startswith(f"**{self.current_model}:**"):
            self.chat_history.append(f"**{self.current_model}:** ")
        self.current_response += chunk
        self.chat_history[-1] = f"**{self.current_model}:** {self.current_response}"
        self.update_chat_display()
        
        # 使用节流的滚动函数，避免频繁跳动
        self.chat_display.page().runJavaScript("scrollToBottom();")
        
    def handle_response_complete(self, response: str, elapsed_time: float) -> None:
        """处理AI响应完成事件"""
        self.chat_history[-1] = f"**{self.current_model}:** {response}\n\n*生成时间: {elapsed_time:.2f}秒*"
        self.update_chat_display()
        
        # 在回答完成时滚动到底部
        self.chat_display.page().runJavaScript("scrollToBottom();")
        
    def handle_error(self, error_message: str) -> None:
        """处理错误消息"""
        self.chat_history.append(f"**System:** {error_message}")
        self.update_chat_display()
        # 错误信息也需要滚动到底部
        self.chat_display.page().runJavaScript("scrollToBottom();")
        
    def update_chat_display(self) -> None:
        """更新聊天显示区域的内容"""
        messages_html = self._generate_messages_html()
        html_content = self._generate_full_html(messages_html)
        self.chat_display.setHtml(html_content)
        
    def _generate_messages_html(self) -> str:
        """生成消息的HTML内容"""
        messages_html = ""
        for message in self.chat_history:
            # 转换markdown为HTML
            message_html = markdown.markdown(message)
            # 添加消息样式
            if message.startswith("**You:**"):
                messages_html += f'<div class="message user-message">{message_html}</div>'
            elif message.startswith("**System:**"):
                messages_html += f'<div class="message system-message">{message_html}</div>'
            else:  # AI模型的消息
                messages_html += f'<div class="message ai-message">{message_html}</div>'
        return messages_html
        
    def _generate_full_html(self, messages_html: str) -> str:
        """生成完整的HTML内容"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                html {{
                    height: 100%;
                    scroll-behavior: smooth;
                }}
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    line-height: 1.6;
                    background-color: #1a1a1a;
                    color: #ffffff;
                    min-height: 100%;
                    display: flex;
                    flex-direction: column;
                }}
                .message {{
                    margin-bottom: 15px;
                    padding: 10px;
                    border-radius: 5px;
                }}
                .user-message {{
                    background-color: #2b2b2b;
                    margin-left: 20%;
                    border: 1px solid #3b3b3b;
                }}
                .ai-message {{
                    background-color: #2b2b2b;
                    margin-right: 20%;
                    border: 1px solid #3b3b3b;
                }}
                .system-message {{
                    background-color: #2b2b2b;
                    margin: 10px 0;
                    padding: 10px;
                    border-radius: 5px;
                    border: 1px solid #3b3b3b;
                }}
                .time-info {{
                    font-size: 0.8em;
                    color: #888;
                    margin-top: 5px;
                    text-align: right;
                }}
                code {{
                    background-color: #2d2d2d;
                    padding: 2px 5px;
                    border-radius: 3px;
                    border: 1px solid #3b3b3b;
                }}
                pre {{
                    background-color: #2d2d2d;
                    padding: 10px;
                    border-radius: 5px;
                    overflow-x: auto;
                    border: 1px solid #3b3b3b;
                }}
                a {{
                    color: #66b3ff;
                }}
                blockquote {{
                    border-left: 4px solid #3b3b3b;
                    margin: 0;
                    padding-left: 10px;
                    color: #cccccc;
                }}
                #chat-container {{
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    gap: 15px;
                    margin-bottom: 20px;
                }}
                #scroll-anchor {{
                    height: 1px;
                    width: 100%;
                }}
                .spacer {{
                    flex-grow: 1;
                }}
            </style>
            <script>
                function scrollToBottom() {{
                    window.scrollTo({{
                        top: document.body.scrollHeight,
                        behavior: 'smooth'
                    }});
                }}
                // 确保页面加载完成后立即定义函数
                document.addEventListener('DOMContentLoaded', function() {{
                    window.scrollToBottom = scrollToBottom;
                    scrollToBottom();
                }});
            </script>
        </head>
        <body>
            <div id="chat-container">
                {messages_html}
            </div>
            <div id="scroll-anchor"></div>
        </body>
        </html>
        """

    def handle_key_press(self, event: Any) -> None:
        """处理键盘事件"""
        # 检查是否按下回车键且没有按住Shift键
        if (event.key() == Qt.Key.Key_Return and 
            not event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            # 阻止事件继续传播
            event.accept()
            # 发送消息
            self.send_message()
        else:
            # 其他按键正常处理
            QTextEdit.keyPressEvent(self.input_field, event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 设置应用图标
    app_icon = QIcon("ollamaICO.png")
    app.setWindowIcon(app_icon)
    
    window = ChatWindow()
    window.show()
    sys.exit(app.exec()) 