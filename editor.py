from engine import Engine
import pyglet
from pyglet.window import key
import numpy as np

import os

S_STEP = 5 # shape step size
J_STEP = np.pi/10 # joint limit step size (radians)

print('Welcome to the experimental pyglet+pybox2d creature editor!')
print('- B to add a dynamic body')
print('- Drag to move bodies')
print('- P to pin bodies together at the current mouse position')
print('- Arrow keys to change the size of the selected body (width, height)')
print('- S to save models as model.json (or whatever filename you passed in load) in the current directory')

class Editor(object):
    def __init__(self):
        self.engine = Engine(width=1280, height=720, gravity=(0, 0), \
            linear_damping=10.0, angular_damping=10.0, caption='Editor', \
            joint_limit=True, lower_angle=-.2*np.pi, upper_angle=.2*np.pi)

    def load(self, filename='model.json'):
        self.filename = filename
        # load if there's a file
        if self.filename and os.path.isfile(self.filename):
            self.engine.load(self.filename)

    def run(self):
        e = self.engine
        while not e.exited():
            e.window.dispatch_events()
            e.window.clear()
            if e.window.mouse_pressed:
                e.create_mouse_joint()
            else:
                e.destroy_mouse_joint()
            self.handle_keys()
            e.update_mouse_joint()
            e.step_physics(1)
            e.render()
            pyglet.clock.tick()
        e.window.close()

    def handle_keys(self):
        e = self.engine
        if e.selected:
            s = e.selected.userData['size']
            s = (s[0]*e.ppm, s[1]*e.ppm)

            if e.window.pressed(key.RIGHT):
                s = (s[0] + S_STEP, s[1])
            if e.window.pressed(key.LEFT):
                s = (s[0] - S_STEP, s[1])
            if e.window.pressed(key.UP):
                s = (s[0], s[1] + S_STEP)
            if e.window.pressed(key.DOWN):
                s = (s[0], s[1] - S_STEP)
            s = (int(max(S_STEP, s[0])), int(max(S_STEP, s[1])))
            e.set_box(e.selected, s)
        else:
            # joint limit
            if e.window.pressed(key.UP):
                j = e.joint_at(e.window.mouse)
                self.change_joint_limit(j, J_STEP)
            if e.window.pressed(key.DOWN):
                j = e.joint_at(e.window.mouse)
                self.change_joint_limit(j, -J_STEP)

        if e.window.pressed(key.P):
            joint = e.pin_at(e.window.mouse)
        if e.window.pressed(key.S):
            e.save(self.filename)
        if e.window.pressed(key.B):
            e.add_dynamic_body(e.window.mouse, (S_STEP * 3, S_STEP * 2))
        e.window.reset_keys()

    def change_joint_limit(self, j, delta):
        if j:
            lim = j.upperLimit + delta
            lim = max(min(lim, np.pi * 2.0), 0)
            j.upperLimit = lim
            j.lowerLimit = -lim


if __name__ == "__main__":
    editor = Editor()

    editor.load('model.json')
    editor.run()