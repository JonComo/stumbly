import numpy as np

import pyglet
from pyglet import gl
from pyglet.window import key
from pyglet.window import mouse

from Box2D import (b2Filter, b2_pi)
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody, circleShape, \
    fixtureDef, transform, revoluteJoint)

import json
from uuid import uuid4

# Disable error checking for increased performance
# pyglet.options['debug_gl'] = False


# http://stackoverflow.com/questions/9035712/numpy-array-is-shown-incorrect-with-pyglet
def tex_from_m(m, resize=4):
    #m = m.T
    shape = m.shape

    m = np.clip(m, -1, 1)
    m += 1
    m /= 2

    m *= 255

    # we need to flatten the array
    m.shape = -1

    # convert to GLubytes
    tex_data = (gl.GLubyte * m.size)( *m.astype('uint8') )

    # create an image
    # pitch is 'texture width * number of channels per element * per channel size in bytes'
    img = pyglet.image.ImageData(shape[1], shape[0], "I", tex_data, pitch = shape[1] * 1 * 1)

    texture = img.get_texture()   
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)   

    texture.width = shape[1] * resize                                                                                                                                                            
    texture.height = -shape[0] * resize                                                                                                                                                                                                                                                                                                                       
    return texture

class Window(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        self.reset_keys()
        self.mouse_pressed = False
        self.mouse = (0, 0)
        self.label = None
        self.texture_cache = None # used for rendering weight/grad matrices
        self.set_line_width(3.0)

    def set_line_width(self, width):
        pyglet.gl.glLineWidth(width)

    def pressed(self, key):
        if key in self.keys:
            return True
        return False

    def on_mouse_press(self, x, y, button, modifiers):
        if button & mouse.LEFT:
            self.mouse_pressed = True

    def on_mouse_release(self, x, y, button, modifiers):
        self.mouse_pressed = False

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.mouse = (x, y)
        self.dx = dx
        self.dy = dy

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse = (x, y)
        self.dx = dx
        self.dy = dy

    def on_key_press(self, symbol, modifiers):
        self.keys[symbol] = True

    def on_key_release(self, symbol, modifiers):
        pass

    def reset_keys(self):
        self.keys = {}

    def line_loop(self, vertices):
        out = []
        for i in range(len(vertices) - 1):
            # 0,1  1,2  2,3 ... len-1,len  len,0
            out.extend(vertices[i])
            out.extend(vertices[i + 1])

        out.extend(vertices[len(vertices) - 1])
        out.extend(vertices[0])

        return len(out) // 2, out

    def triangle_fan(self, vertices):
        out = []
        for i in range(1, len(vertices) - 1):
            # 0,1,2   0,2,3  0,3,4 ..
            out.extend(vertices[0])
            out.extend(vertices[i])
            out.extend(vertices[i + 1])
        return len(out) // 2, out

    def draw_poly(self, vertices, color):
        ll_count, ll_vertices = self.line_loop(vertices)

        pyglet.graphics.draw(ll_count, gl.GL_LINES,
                        ('v2f', ll_vertices),
                        ('c4f', [color[0], color[1], color[2], 1] * (ll_count)))

    def draw_poly_fill(self, vertices, color):
        tf_count, tf_vertices = self.triangle_fan(vertices)
        if tf_count == 0:
            return

        pyglet.graphics.draw(tf_count, gl.GL_TRIANGLES,
                        ('v2f', tf_vertices),
                        ('c4f', [0.5 * color[0], 0.5 * color[1], 0.5 * color[2], 0.5] * (tf_count)))

        """ll_count, ll_vertices = self.line_loop(vertices)

        pyglet.graphics.draw(ll_count, gl.GL_LINES,
                        ('v2f', ll_vertices),
                        ('c4f', [color[0], color[1], color[2], 1.0] * ll_count))"""
    
    def draw_rect(self, x, y, w, h, color, thickness=1):
        verts = ((x, y), (x+w, y), (x+w, y+h), (x, y+h), (x, y))
        if thickness > 0:
            # edges only
            self.draw_poly(verts, color)
        else:
            # full
            self.draw_poly_fill(verts, color)

    def draw_text(self, text, size=18, p=None):
        p = (10, 10) if not p else p
        self.label = pyglet.text.Label(text,
                          font_name='monospace',
                          font_size=size,
                          x=p[0], y=p[1],
                          anchor_x='left', anchor_y='bottom')
        
    def draw_matrices(self, M, x=10, y=710, recalc=True):
        if recalc:
            self.texture_cache = None

        if not self.texture_cache:
            self.texture_cache = [tex_from_m(m) for m in M]

        for t in self.texture_cache:
            t.blit(x, y)
            x += t.width + 10

class Engine(object):
    def __init__(self, ppm=20, fps=60, width=640, height=480, gravity=(0, 0), \
     caption="Window", joint_limit=False, lower_angle=-.5*b2_pi, upper_angle=.5*b2_pi, max_torque=10000, \
     linear_damping=0.0, angular_damping=0.0, enable_mouse_joint=True):
        self.window = Window(width=width, height=height, caption=caption)
        self.ppm = ppm # pixels per meter
        self.width = width
        self.height = height

        self.colors = [(255, 255, 255, 255), (255, 0, 0, 255)]
        self.fps = fps
        self.timestep = 1.0 / self.fps
        self.world = world(gravity=gravity, doSleep=False)
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        self.joint_limit = joint_limit
        self.lower_angle = lower_angle
        self.upper_angle = upper_angle
        self.max_torque = max_torque
        self.enable_mouse_joint = enable_mouse_joint
        self.selected = None
        self.mouse_joint = None

        self.joints = []
        self.bodies = []

        # create ground
        self.ground = self.add_static_body(p=(self.width/2, 10), \
            size=(self.width, 30))

    def to_pybox2d(self, p):
        return [p[0]/self.ppm, p[1]/self.ppm]

    def size_to_pybox2d(self, s):
        return (s[0]/self.ppm, s[1]/self.ppm)

    def to_window(self, p):
        return [p[0]*self.ppm, p[1]*self.ppm]

    def render(self):
        self.window.set_line_width(2.0)
        for body in self.world.bodies:
            for fixture in body.fixtures:
                shape = fixture.shape
                if isinstance(shape, polygonShape):
                    vertices = [(body.transform * v) * self.ppm for v in shape.vertices]
                    if body == self.selected:
                        self.window.draw_poly(vertices, self.colors[1])
                    else:
                        self.window.draw_poly(vertices, self.colors[0])

        self.window.set_line_width(2.0)
        for joint in self.joints:
            if joint.limitEnabled:
                p = self.to_window(joint.anchorA)
                l = self.ppm
                rot = -np.pi/2
                lower = joint.limits[0] - rot
                upper = joint.limits[1] - rot
                current = joint.angle - rot
                
                center = (p[0], p[1])

                tri = [center]
                tri += [(p[0] + np.cos(lower) * l, p[1] + np.sin(lower) * l)]
                tri += [center]
                tri += [(p[0] + np.cos(upper) * l, p[1] + np.sin(upper) * l)]

                cur = [tri[0]]
                cur += [(p[0] + np.cos(current) * l, p[1] + np.sin(current) * l)]
                self.window.draw_poly(tri, self.colors[1])
                self.window.draw_poly(cur, self.colors[0])
            else:
                p = self.to_window(joint.anchorA)
                l = self.ppm/4
                
                tri = [(p[0]-l, p[1]-l)]
                tri += [(p[0]+l, p[1]-l)]
                tri += [(p[0], p[1]+l)]

                self.window.draw_poly(tri, self.colors[1])
        
        if self.window.label:
            self.window.label.draw()
        self.window.flip()

    def step_physics(self, steps=1):
        for i in range(steps):
            self.world.Step(self.timestep, 10, 10)

    def close(self):
        self.window.close()

    def exited(self):
        return self.window.has_exit

    def clock_tick(self):
        pyglet.clock.tick()

    def create_mouse_joint(self):
        if self.selected:
            return

        bodies = self.bodies_at(self.window.mouse)
        if len(bodies) > 0:
            self.selected = bodies[0]
            self.selected.awake = True
            self.mouse_joint = self.world.CreateMouseJoint(bodyA=self.ground, bodyB=self.selected, \
                target=self.to_pybox2d(self.window.mouse), maxForce=1000.0 * self.selected.mass)

    def destroy_mouse_joint(self):
        self.selected = None
        if self.mouse_joint:
            self.world.DestroyJoint(self.mouse_joint)
            self.mouse_joint = None

    def update_mouse_joint(self):
        if self.mouse_joint:
            self.mouse_joint.target = self.to_pybox2d(self.window.mouse)

    def body_position(self):
        x = 0
        y = 0
        for b in self.bodies:
            x += b.position[0]
            y += b.position[1]
        cnt = len(self.bodies)
        return self.to_window([x/cnt, y/cnt])

    def set_position(self, p, zero_vel=True):
        c = self.to_pybox2d(self.body_position())
        p = self.to_pybox2d(p)
        shift = (p[0] - c[0], p[1] - c[1])
        for b in self.bodies:
            if zero_vel:
                b.linearVelocity = (0, 0)
                b.angularVelocity = 0
            b.position = (b.position[0] + shift[0], b.position[1] + shift[1])

    def add_static_body(self, p, size):
        return self.world.CreateStaticBody(position=self.to_pybox2d(p), \
            shapes=polygonShape(box=self.size_to_pybox2d(size), friction=1.0))

    def bodies_at(self, p):
        p = self.to_pybox2d(p)
        bodies = []
        for body in self.bodies:
            if body.fixtures[0].shape.TestPoint(body.transform, p):
                bodies.append(body)
        return bodies

    def joint_at(self, p, limit=1):
        p = self.to_pybox2d(p)
        dist = limit
        best_j = None
        for j in self.joints:
            jp = j.anchorA
            d = (p[0] - jp[0])**2 + (p[1] - jp[1])**2
            if d < dist:
                dist = d
                best_j = j
        return best_j

    def body_with_uuid(self, uuid):
        for body in self.bodies:
            if body.userData and isinstance(body.userData, dict):
                if body.userData['uuid'] == uuid:
                    return body
        return None

    def add_dynamic_body(self, p, size, angle=0, uuid=None):
        body = self.world.CreateDynamicBody(position=self.to_pybox2d(p), angle=angle)
        body.userData = {}
        body.linearDamping = self.linear_damping
        body.angularDamping = self.angular_damping
        uuid = uuid if uuid else str(uuid4())
        body.userData['uuid'] = uuid
        self.set_box(body, size)
        self.bodies.append(body)
        return body

    def set_box(self, body, size):
        while len(body.fixtures) > 0:
            body.DestroyFixture(body.fixtures[0])
        size = self.size_to_pybox2d(size)
        body.CreatePolygonFixture(box=size, density=1, friction=1.0, filter=b2Filter(groupIndex=-2))
        body.userData['size'] = size

    def add_static_body(self, p, size):
        return self.world.CreateStaticBody(position=self.to_pybox2d(p), \
            shapes=polygonShape(box=self.size_to_pybox2d(size)))

    def pin_at(self, p, a_uuid=None, b_uuid=None, ll=None, ul=None):
        bodies = []
        if a_uuid and b_uuid:
            bodies = [self.body_with_uuid(a_uuid), self.body_with_uuid(b_uuid)]
        else:
            bodies = self.bodies_at(p)

        if len(bodies) >= 2:
            b1 = bodies[0]
            b2 = bodies[1]
            joint = self.world.CreateRevoluteJoint(bodyA=b1, bodyB=b2, anchor=self.to_pybox2d(p), 
                maxMotorTorque = self.max_torque,
                motorSpeed = 0.0,
                enableMotor = True,
                upperAngle = ul if ul else self.upper_angle,
                lowerAngle = ll if ll else self.lower_angle,
                enableLimit = self.joint_limit
                )
            self.joints.append(joint)
            return joint
        return None

    def body_data(self, body):
        return {'p': (body.position[0], body.position[1]), \
            'size': body.userData['size'], 'angle': body.angle, 'uuid': body.userData['uuid']}

    def joint_data(self, joint):
        return {'p': (joint.anchorA[0], joint.anchorA[1]), 'a_uuid': joint.bodyA.userData['uuid'], \
            'b_uuid': joint.bodyB.userData['uuid'], 'll': joint.limits[0], 'ul': joint.limits[1]}

    def load_body(self, d):
        self.add_dynamic_body(self.to_window(d['p']), (d['size'][0] * self.ppm, \
            d['size'][1] * self.ppm), angle=d['angle'], uuid=d['uuid'])

    def load_joint(self, d):
        ul = None
        ll = None
        if 'ul' in d and 'll' in d:
            ul = d['ul']
            ll = d['ll']
        self.pin_at(self.to_window(d['p']), a_uuid=d['a_uuid'], b_uuid=d['b_uuid'], \
         ll=ll, ul=ul)

    def settings_data(self):
        return {}

    def load_settings(self, d):
        pass

    def save(self, filename='model.json'):
        data = {'_settings': self.settings_data(), 'bodies': [], 'joints': []}
        for body in self.bodies:
            if body.userData:
                data['bodies'].append(self.body_data(body))
        for joint in self.joints:
            data['joints'].append(self.joint_data(joint))

        with open(filename, 'w') as fp:
            json.dump(data, fp, sort_keys=True, indent=4)

        print('File saved as: {}'.format(filename))

    def load(self, filename='model.json'):
        with open(filename, 'r') as fp:
            data = json.load(fp)
            self.file_data = data
            self.load_settings(data['_settings'])
            for b in data['bodies']:
                self.load_body(b)
            for j in data['joints']:
                self.load_joint(j)

    def clear_all_but_ground(self):
        self.destroy_mouse_joint()
        for b in self.bodies:
            self.world.DestroyBody(b)
        self.bodies = []
        self.joints = []

if __name__ == "__main__":
    print('Welcome to the experimental pyglet + pybox2d creature engine!')
    print('- You most likely want to import this as a module')
    print('- and use it to do simple simulations.')

    engine = Engine(width=1280, height=720, caption='Built in run loop', gravity=(0, -30))
    for i in range(10):
        b = engine.add_dynamic_body((engine.width/2 + (i-5)*5.0, engine.height/2 + i * 10.0), (10, 10))
        b.angle = np.random.random_sample()
    
    while not engine.exited():
        engine.window.dispatch_events()
        engine.window.clear()
        if engine.window.mouse_pressed:
            engine.create_mouse_joint()
        else:
            engine.destroy_mouse_joint()
        engine.update_mouse_joint()
        engine.step_physics(1)
        engine.render()
        pyglet.clock.tick()