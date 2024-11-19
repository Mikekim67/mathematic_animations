import numpy as np
from vispy import app, gloo
from vispy.gloo import Program

app.use_app('pyqt6')

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

# 애니메이션 실행
if __name__ == '__main__':
    canvas = ComplexPlaneAnimation(frequency=241, frames_per_second=240)
    canvas.show()
    app.run()

#frames_per_second = 모니터의 주사율, frequency = 사용자가 원하는 회전 주파수