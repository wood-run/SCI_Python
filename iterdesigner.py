# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:28:58 2013

@author: cjy
"""

from traitsui.api import *
from traitsui.menu import OKCancelButtons
from traits.api import *
from traitsui.wx.editor import Editor

import matplotlib
matplotlib.use("WXAgg")
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np
import thread
import time
import wx
import pickle

ITER_COUNT = 20000
ITER_TIMES = 10
VERY_BIG = 1000000

def triangle_area(tri):
    """
    Obtain the area of the triangle
    """
    A = tri[0]
    B = tri[1]
    C = tri[2]
    AB = A - B
    AC = A - C
    return np.abs(np.cross(AB, AC)) / 2.0
    
def solve_eq(tri1, tri2):
    """
    从tri1变换到tri2的变换系数
    tri1,2是二维数组
    x0, y0
    x1, y1
    x2, y2
    """
    x0, y0 = tri1[0]
    x1, y1 = tri1[1]
    x2, y2 = tri1[2]
    
    a = np.zeros((6, 6), dtype = np.float)
    b = tri2.reshape(-1)
    a[0, 0:3] = x0, y0, 1
    a[1, 3:6] = x0, y0, 1
    a[2, 0:3] = x1, y1, 1
    a[3, 3:6] = x1, y1, 1
    a[4, 0:3] = x2, y2, 1
    a[5, 3:6] = x2, y2, 1
    
    c = np.linalg.solve(a, b)
    c.shape = (2, 3)
    return c
    
def ifs(p, eq, init, n):
    """
    函数迭代，参数定义如下：
    p：函数选择概率列表
    eq：迭代函数列表
    init：迭代初始点
    n：迭代次数
    return：每次迭代所得的XY坐标数组，和计算所用函数下标
    """
    #初始化迭代向量
    pos = np.ones(3, dtype = np.float)
    pos[:2] = init
    
    #通过概率，计算函数的选择序列
    p = np.add.accumulate(p)
    rands = np.random.rand(n)
    select = np.ones(n, dtype = np.int) #* (n - 1)
    for i, x in enumerate(p[::-1]):
        select[rands < x] = len(p) - i - 1

    #initiate the result
    result = np.zeros((n, 2), dtype = np.float)
    #c = np.zeros(n, dtype = np.float)
    
    for i in xrange(n):
        #eqidx = select[i]
        tmp = np.dot(eq[select[i]], pos)
        pos[:2] = tmp
        result[i] = tmp
        #c[i] = eqidx
        
    return result[:, 0], result[:, 1], select
    
class _MPLFigureEditor(Editor):
    """
    use matplotlib figure's traits editor
    """
    scrollable = True
    
    def init(self, parent):
        self.control = self._create_canvas(parent)
        
    def update_editor(self):
        pass
    
    def _create_canvas(self, parent):
        panel = wx.Panel(parent, -1, style = wx.CLIP_CHILDREN)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)
        mpl_control = FigureCanvas(panel, -1, self.value)
        sizer.Add(mpl_control, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.value.canvas.SetMinSize((10, 10))
        return panel
        
class MPLFigureEditor(BasicEditorFactory):
    """
    return the class for creating controllers
    """
    klass = _MPLFigureEditor
    
class IFSTriangles(HasTraits):
    """
    Editor for triangles
    """
    #记录重绘次数
    version = Int(0)
    
    def __init__(self, ax):
        #某种初始化
        super(IFSTriangles, self).__init__()
        #设置默认颜色数组和points
        self.colors = ["r", "g", "b", "c", "m", "y", "k"]
        self.points = np.array([(0,0),(2,0),(2,4),(0,1),(1,1),(1,3),(1,1),(2,1),(2,3)], dtype=np.float)
        #得到变换函数
        self.equations = self.get_eqs()
        self.ax = ax
        self.ax.set_ylim(-10, 10)
        self.ax.set_xlim(-10, 10)
        canvas = ax.figure.canvas
        #绑定canvas上的鼠标事件
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.canvas = canvas
        self._ind = None
        self.background = None
        self.update_lines()
        
    def refresh(self):
        """
        repaint all triangles
        """
        self.update_lines()
        self.canvas.draw()
        self.version += 1
        
    def del_triangle(self):
        """
        delete the last triangle
        """
        self.points = self.points[:-3].copy()
        self.refresh()
        
    def add_triangle(self):
        """
        create a new triangle
        """
        self.points = np.vstack((self.points, np.array([(0,0),(1,0),(0,1)],dtype=np.float)))
        self.refresh()
        
    def set_points(self, points):
        """
        directly set the points of triangles
        """
        self.points = points.copy()
        self.refresh()
        
    def get_eqs(self):
        """
        calculate all the affine equations
        """
        eqs = []
        #每三个点为一组，从第1组开始计算
        for i in range(1, len(self.points) / 3):
            #相当于取第0组和第i组求解仿射函数
            eqs.append(solve_eq(self.points[:3, :], self.points[i*3:i*3+3, :]))
        return eqs
        
    def get_areas(self):
        """
        get the equation iterate probability based on triangle's area
        """
        areas = []
        for i in range(1, len(self.points) / 3):
            areas.append(triangle_area(self.points[i*3:i*3+3, :]))
        s = sum(areas)
        return [x/s for x in areas]
        
    def update_lines(self):
        """
        redraw all new triangles
        """
        del self.ax.lines[:]
        for i in xrange(0, len(self.points), 3):
            color = self.colors[i / 3 % len(self.colors)]
            x0, x1, x2 = self.points[i:i+3, 0]
            y0, y1, y2 = self.points[i:i+3, 1]
            linetype = color + "%so"
            if i == 0:
                linewidth = 3
            else:
                linewidth = 1
            self.ax.plot([x0, x1], [y0, y1], linetype % "-", linewidth = linewidth)
            self.ax.plot([x1, x2], [y1, y2], linetype % "--", linewidth = linewidth)
            self.ax.plot([x0, x2], [y0, y2], linetype % ":", linewidth = linewidth)
            
        self.ax.set_ylim(-10, 10)
        self.ax.set_xlim(-10, 10)
        
    def button_release_callback(self, event):
        """
        the event when mouse button releases
        """
        self._ind = None
        
    def button_press_callback(self, event):
        """
        the event when mouse button presses
        """
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event.xdata, event.ydata)
        
    def get_ind_under_point(self, mx, my):
        """
        find the nearest point to (mx, my)
        """
        for i, p in enumerate(self.points):#i代表序号，p代表列举的每一项
            if abs(mx - p[0]) < 0.5 and abs(my - p[1]) < 0.5:
                return i
        return None
        
    def motion_notify_callback(self, event):
        """
        the event when mouse moves
        """
        self.event = event
        if self._ind is None:
            return
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata
        
        #update the points
        self.points[self._ind, :] = [x, y]
        i = self._ind / 3 * 3
        #update the lines
        x0, x1, x2 = self.points[i:i+3, 0]
        y0, y1, y2 = self.points[i:i+3, 1]
        self.ax.lines[i].set_data([x0, x1], [y0, y1])
        self.ax.lines[i+1].set_data([x1, x2], [y1, y2])  
        self.ax.lines[i+2].set_data([x0, x2], [y0, y2])  
        
        #if the background is empty, capture it
        if self.background == None:
            self.ax.clear()
            self.ax.set_axis_off()
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
            self.update_lines()
            
        #redraw all triangles quickly
        self.canvas.restore_region(self.background)
        for line in self.ax.lines:
            self.ax.draw_artist(line)
        self.canvas.blit(self.ax.bbox)
        
        self.version += 1
        
class AskName(HasTraits):
    name = Str("")
    view = View(
        Item("name", label = u"Name"),
        kind = "modal",
        buttons = OKCancelButtons
    )
    
class IFSHandler(Handler):
    """
    initiate before the window appears
    """
    def init(self, info):
        info.object.init_gui_component()
        return True
        
class IFSDesigner(HasTraits):
    #规定Figure的实例
    figure = Instance(Figure)
    #规定IFSTriangles的实例    
    ifs_triangle = Instance(IFSTriangles)
    #规定几种按钮    
    add_button = Button(u"Add a triangle")
    del_button = Button(u"Del a triangle")
    save_button = Button(u"Save the IFS")
    unsave_button = Button(u"Del the IFS")
    #布尔型clear 表示
    clear = Bool(True)
    #布尔型exit 表示程序是否退出
    exit = Bool(False)
    #初始化，为名称和点集做准备
    ifs_names = List()
    ifs_points = List()
    current_name = Str
    
    #基础界面
    view = View(
        VGroup(
            #竖向分割，上面是一排控制，下面是图像
            HGroup(
                Item("add_button"),
                Item("del_button"),
                Item("current_name", editor = EnumEditor(name = "object.ifs_names")),
                Item("save_button"),
                Item("unsave_button"),
                show_labels = False
            ),
            Item("figure", editor = MPLFigureEditor(), show_label = False, width = 600)
        ),
        resizable = True,
        height = 350,
        width = 600,
        title = u"Iterate Function System Designer",
        handler = IFSHandler()
    )
    
    def _current_name_changed(self):
        self.ifs_triangle.set_points(self.ifs_points[self.ifs_names.index(self.current_name)])
        
    def _add_button_fired(self):
        self.ifs_triangle.add_triangle()
        
    def _del_button_fired(self):
        self.ifs_triangle.del_triangle()
        
    def _unsave_button_fired(self):
        if self.current_name in self.ifs_names:
            index = self.ifs_names.index(self.current_name)
            del self.ifs_names[index]
            del self.ifs_points[index]
            self.save_data()
            
    def _save_button_fired(self):
        ask = AskName(name = self.current_name)
        if ask.configure_traits():
            if ask.name not in self.ifs_names:
                self.ifs_names.append(ask.name)
                self.ifs_points.append(self.ifs_triangle.points.copy())
            else:
                index = self.ifs_names.index(ask.name)
                self.ifs_names[index] = ask.name
                self.ifs_points[index] = self.ifs_triangle.points.copy()
            self.save_data()
            
    def save_data(self):
        with file("IFS.data", "wb") as f:
            #turn ifs_names to a list
            pickle.dump(self.ifs_names[:], f)
            for data in self.ifs_points:
                np.save(f, data)
                
    def ifs_calculate(self):
        """
        calculate in another thread
        """
        def draw_points(x, y, c):
            if len(self.ax2.collections) < ITER_TIMES:
                try:
                    self.ax2.scatter(x, y, s = 1, c = c, marker = "s", linewidths = 0)
                    self.ax2.set_axis_off()
                    self.ax2.axis("equal")
                    self.figure.canvas.draw()
                except:
                    pass
                
        def clear_points():
            self.ax2.clear()
            
        while 1:
            try:
                if self.exit == True:
                    break
                if self.clear == True:
                    self.clear = False
                    self.initpos = [0, 0]
                    #don't draw the first 100 points
                    x, y, c = ifs(self.ifs_triangle.get_areas(), self.ifs_triangle.get_eqs(), self.initpos, 100)
                    self.initpos = [x[-1], y[-1]]
                    self.ax2.clear()              
                   
                x, y, c = ifs(self.ifs_triangle.get_areas(), self.ifs_triangle.get_eqs(), self.initpos, ITER_COUNT)
                if np.max(np.abs(x)) < VERY_BIG and np.max(np.abs(y)) < VERY_BIG:
                    self.initpos = [x[-1], y[-1]]
                    wx.CallAfter(draw_points, x, y, c)
                time.sleep(0.05)
            except:
                pass
            
    @on_trait_change("ifs_triangle.version")
    def on_ifs_version_changed(self):
        """
        When triangles update, redraw all iterate points
        """
        self.clear = True
        
    def _figure_default(self):
        """
        build a Figure object with default values
        """
        figure = Figure()
        self.ax = figure.add_subplot(121)
        self.ax2 = figure.add_subplot(122)
        self.ax2.set_axis_off()
        self.ax.set_axis_off()
        figure.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, wspace = 0, hspace = 0)
        figure.patch.set_facecolor("w")
        return figure
        
    def init_gui_component(self):
        self.ifs_triangle = IFSTriangles(self.ax)
        self.figure.canvas.draw()
        thread.start_new_thread(self.ifs_calculate, ())
        try:
            with file("ifs.data", "rb") as f:
                self.ifs_names = pickle.load(f)
                self.ifs_points = []
                for i in xrange(len(self.ifs_names)):
                    self.ifs_points.append(np.load(f))
                    
            if len(self.ifs_names) > 0:
                self.current_name = self.ifs_names[-1]
        except:
            pass
        
designer = IFSDesigner()
designer.configure_traits()
designer.exit = True
