import numpy as np
from vispy import app, gloo
from vispy.gloo import Program
from PyQt5 import QtWidgets, QtCore
import sys
import subprocess

#pip3 install numpy
#pip3 install vispy
#pip3 install PyQt5

#app.use_app('pyqt5')  # PyInstaller와 호환성을 위해 pyqt5로 변경

# 셰이더 프로그램 (Vertex Shader와 Fragment Shader)
vertex_shader = """
attribute vec2 position;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

fragment_shader = """
void main() {
    gl_FragColor = vec4(0.2, 0.6, 1.0, 1.0);  // Blue color
}
"""

# Canvas 설정
class ComplexPlaneAnimation(app.Canvas):
    def __init__(self, frequency=1, frames_per_second=60):
        super().__init__(size=(800, 800), dpi=96, title='GPU Accelerated Complex Vector Rotation')

        # 클래스 속성 초기화
        self.frequency = frequency  # 주파수
        self.frames_per_second = frames_per_second  # 초당 프레임 수
        self.time = 0
        self.program = Program(vertex_shader, fragment_shader)

        # 단위 벡터 초기화
        self.vector = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        self.program['position'] = self.vector

        # GPU 버퍼에 데이터 업로드
        self.program.bind(gloo.VertexBuffer(self.vector))

        # 타이머 설정
        self.timer = app.Timer(interval=1 / frames_per_second, connect=self.on_timer, start=True)

        # OpenGL 설정
        gloo.set_state(clear_color='black', blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))

    def on_draw(self, event):
        gloo.clear()
        self.program.draw('lines')  # 벡터를 선으로 그리기

    def on_timer(self, event):
        # 현재 시간에 따른 회전 계산
        angle = 2 * np.pi * self.frequency * self.time
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                     [np.sin(angle),  np.cos(angle)]], dtype=np.float32)
        rotated_vector = rotation_matrix @ np.array([[1.0], [0.0]])
        
        # 벡터 업데이트
        self.vector[1] = rotated_vector.flatten()
        self.program['position'] = self.vector
        self.update()  # 화면 업데이트 요청

        # 시간 증가
        self.time += 1 / self.frames_per_second

    def update_frequency(self, frequency):
        self.frequency = frequency

# GUI 설정
class ControlPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.canvas = None
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()

        # 주사율 입력
        self.fps_input = QtWidgets.QSpinBox()
        self.fps_input.setRange(1, 240)  # 주사율 범위 설정 (1-240 Hz)
        self.fps_input.setValue(60)
        layout.addWidget(QtWidgets.QLabel('Monitor Refresh Rate (frames per second):'))
        layout.addWidget(self.fps_input)

        # 주파수 슬라이더 및 입력
        self.freq_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.freq_slider.setRange(0, 500)  # 초기 범위 설정 (0 to 500 Hz)
        self.freq_slider.setValue(1)
        self.freq_slider.valueChanged.connect(self.update_frequency)
        layout.addWidget(QtWidgets.QLabel('Frequency (Hz):'))
        layout.addWidget(self.freq_slider)

        self.freq_input = QtWidgets.QSpinBox()
        self.freq_input.setRange(0, 500)  # 주파수 범위 설정 (0-500 Hz)
        self.freq_input.setValue(1)
        self.freq_input.valueChanged.connect(self.update_frequency_slider)
        layout.addWidget(self.freq_input)

        # 시작 버튼
        self.start_button = QtWidgets.QPushButton('Start Animation')
        self.start_button.clicked.connect(self.start_animation)
        layout.addWidget(self.start_button)

        self.setLayout(layout)
        self.setWindowTitle('Complex Plane Animation Control Panel')

    def update_frequency(self, value):
        self.freq_input.setValue(value)
        if self.canvas is not None:
            self.canvas.update_frequency(value)

    def update_frequency_slider(self, value):
        self.freq_slider.setValue(value)
        if self.canvas is not None:
            self.canvas.update_frequency(value)

    def start_animation(self):
        frames_per_second = self.fps_input.value()
        frequency = self.freq_slider.value()
        self.canvas = ComplexPlaneAnimation(frequency=frequency, frames_per_second=frames_per_second)
        self.canvas.show()
        app.run()

# 애니메이션 실행
if __name__ == '__main__':
    qt_app = QtWidgets.QApplication([])
    control_panel = ControlPanel()
    control_panel.show()
    qt_app.exec()

# EXE 파일로 빌드하는 방법
# 이 파일을 exe로 만들기 위해서는 PyInstaller를 사용할 수 있습니다.
# 아래의 명령어를 사용하여 실행 파일을 생성할 수 있습니다.
# 터미널에서 다음 명령어를 실행하세요:
# pyinstaller --onefile --windowed your_script_name.py
# 'your_script_name.py'를 이 파일의 이름으로 변경하세요.
# '--onefile' 옵션은 단일 실행 파일을 생성하고, '--windowed' 옵션은 콘솔 창 없이 GUI만 실행되도록 합니다.
