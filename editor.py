from engine import Engine
from pygame import (K_RIGHT, K_LEFT, K_UP, K_DOWN, K_r, K_s, K_p)

from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody, circleShape, fixtureDef, transform, revoluteJoint)
from Box2D.b2 import (pi, filter)

import os

S_STEP = 5 # shape step size

class Editor(object):
    def __init__(self):
        self.engine = Engine(width=1280, height=720, gravity=(0, 0), \
            linear_damping=10.0, angular_damping=10.0, caption='Editor')
        self.ground = self.engine.add_static_body(p=(self.engine.width/2, self.engine.height-10), \
            size=(self.engine.width, 30))
        self.selected = None # selected physics body
        self.mouse_joint = None

    def load(self, filename='model.json'):
        self.filename = filename
        # load if there's a file
        if self.filename and os.path.isfile(self.filename):
            self.engine.load(self.filename)

    def run(self):
        self.engine.run(callback=self.callback, mouse_pressed=self.mouse_pressed, \
            mouse_released=self.mouse_released, key_pressed=self.key_pressed)

    def callback(self):
        if self.mouse_joint:
            self.mouse_joint.target = self.engine.to_pybox2d(self.engine.mouse)

    def mouse_pressed(self):
        bodies = self.engine.bodies_at(self.engine.mouse)
        if len(bodies) > 0:
            self.selected = bodies[0]
            self.selected.awake = True
            self.mouse_joint = self.engine.world.CreateMouseJoint(bodyA=self.ground, bodyB=self.selected, \
                target=self.engine.to_pybox2d(self.engine.mouse), maxForce=1000.0 * self.selected.mass)
        else:
            self.engine.add_dynamic_body(self.engine.mouse, (S_STEP * 3, S_STEP * 2))

    def mouse_released(self):
        self.selected = None
        if self.mouse_joint:
            self.engine.world.DestroyJoint(self.mouse_joint)
            self.mouse_joint = None

    def key_pressed(self, keys):
        if self.selected:
            s = self.selected.userData['size']
            s = (s[0]*self.engine.ppm, s[1]*self.engine.ppm)
            if keys[K_RIGHT]:
                s = (s[0] + S_STEP, s[1])
            if keys[K_LEFT]:
                s = (s[0] - S_STEP, s[1])
            if keys[K_UP]:
                s = (s[0], s[1] + S_STEP)
            if keys[K_DOWN]:
                s = (s[0], s[1] - S_STEP)
            s = (int(max(S_STEP, s[0])), int(max(S_STEP, s[1])))
            self.engine.set_box(self.selected, s)
        if keys[K_p]:
            joint = self.engine.pin_at(self.engine.mouse)
        if keys[K_s]:
            self.engine.save(self.filename)


if __name__ == "__main__":
    print('Welcome to the experimental pybox2d editor!')
    print('- Click to add a dynamic body')
    print('- Drag to move bodies')
    print('- \'P\' to pin bodies together at the current mouse position')
    print('- Arrow keys to change the size of the selected body (width, height)')
    print('- \'S\' to save models as model.json in the current directory')

    editor = Editor()
    editor.load('model.json')
    editor.run()