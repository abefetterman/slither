# based on https://github.com/openai/gym/blob/master/gym/envs/classic_control/rendering.py

import pyglet
from pyglet.gl import *

RAD2DEG = 57.29577951308232

class Viewer(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.window = pyglet.window.Window(width=width, height=height)
        self.window.on_close = self.window_closed_by_user
        self.isopen = True
        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width/(right-left)
        scaley = self.height/(top-bottom)
        self.transform = Transform(
            translation=(-left*scalex, -bottom*scaley),
            scale=(scalex, scaley))

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array=False):
        glClearColor(1,1,1,1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.draw()
        for geom in self.onetime_geoms:
            geom.draw()
        self.transform.disable()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1,:,0:3]
        self.window.flip()
        self.onetime_geoms = []
        return arr if return_rgb_array else self.isopen


class Transform:
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1,1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)
    def enable(self):
        glPushMatrix()
        glTranslatef(self.translation[0], self.translation[1], 0) # translate to GL loc ppint
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(self.scale[0], self.scale[1], 1)
    def disable(self):
        glPopMatrix()
    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))
    def set_rotation(self, new):
        self.rotation = float(new)
    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))


class Border:
    def __init__(self, window_w, window_h, border_w, color):
        tl_outer=(window_w,window_h)
        tl_inner=(window_w-border_w, window_h-border_w)
        tr_outer=(0,window_h)
        tr_inner=(border_w,window_h-border_w)
        bl_outer=(window_w,0)
        bl_inner=(window_w-border_w,border_w)
        br_outer=(0,0)
        br_inner=(border_w,border_w)
        data=tl_outer+tl_inner+tr_outer+tr_inner+br_outer+br_inner+bl_outer+bl_inner+tl_outer+tl_inner
        top_indices=[0,1,3,2]
        left_indices=[0,1,5,4]
        right_indices=[2,3,7,6]
        bottom_indices=[4,5,7,6]
        indices=top_indices+right_indices+bottom_indices+left_indices
        color_list=color*10
        self.vertex_list = pyglet.graphics.vertex_list(10,
            ('v2f', data),
            ('c3B', color_list)
        )
    def draw(self):
        self.vertex_list.draw(GL_QUAD_STRIP)

class Plotter:
    def __init__(self, window_w, window_h, border_w, state_w, state_h, point_list=[]):
        self.x0=border_w
        self.y0=border_w
        self.x_max=state_w-1
        self.dx=(window_w-2*border_w)//state_w
        self.dy=(window_h-2*border_w)//state_h
        self.point_list=point_list
    def _get_vertices(self,x,y):
        x0,y0 = self.x0, self.y0
        dx,dy = self.dx, self.dy
        tl=((x+1)*dx+x0, (y+1)*dy+y0)
        tr=(x*dx+x0, (y+1)*dy+y0)
        bl=((x+1)*dx+x0, y*dy+y0)
        br=(x*dx+x0, y*dy+y0)
        return tl+tr+br+bl
    def update_points(self, point_list):
        self.point_list=point_list
    def draw(self):
        if (self.point_list is None): return
        for x_inv,y,c in self.point_list:
            x=self.x_max-x_inv #invert x for sanity
            vertex_list=self._get_vertices(x,y)
            pyglet.graphics.draw(4, GL_QUADS,
                ('v2f',vertex_list),
                ('c3B',c*4)
            )
