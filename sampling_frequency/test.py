from vispy import app, gloo

canvas = app.Canvas(keys='interactive')

@canvas.events.draw.connect
def on_draw(event):
    gloo.clear(color='blue')  # 파란색 배경 설정

canvas.show()
app.run()
