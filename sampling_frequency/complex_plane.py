import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 회전 주파수 설정 (1초에 1바퀴 = 2π 라디안)
frequency = 1  # n Hz

# 애니메이션 설정
frames_per_second = 60  # 초당 n프레임
duration = 1  # 애니메이션 지속 시간 (초)
num_frames = frames_per_second * duration  # 전체 프레임 수

# 시간 배열 생성
time = np.linspace(0, duration, num_frames, endpoint=False)

# 복소수 생성
def complex_rotation(t, frequency):
    return np.exp(2j * np.pi * frequency * t)

# 복소수의 실수부와 허수부 계산
z = complex_rotation(time, frequency)
x = z.real  # NumPy 배열 형태
y = z.imag  # NumPy 배열 형태

# 애니메이션 생성
fig, ax = plt.subplots()
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.grid()

# 원형 경로 그리기
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
ax.add_artist(circle)

# 점과 궤적 초기화
point, = ax.plot([], [], 'ro', markersize=8)  # 회전하는 점
trajectory, = ax.plot([], [], 'b-', lw=0.5)  # 궤적

# 궤적 저장용 배열
trail_x = []
trail_y = []

def init():
    point.set_data([], [])
    trajectory.set_data([], [])
    return point, trajectory

def update(frame):
    # 현재 프레임의 데이터 계산
    current_x = x[frame]
    current_y = y[frame]

    # 궤적 업데이트
    #trail_x.append(current_x)
    #trail_y.append(current_y)

    # 점과 궤적 데이터 설정
    point.set_data([current_x], [current_y])
    trajectory.set_data(trail_x, trail_y)
    return point, trajectory

# 각 프레임 간 시간 간격 (밀리초)
interval = 1000 / frames_per_second

ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=interval)

plt.title(f"Complex Rotation at {frequency} Hz (2π rad/s)")
plt.show()
