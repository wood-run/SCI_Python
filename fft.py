# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:58:54 2013

@author: cjy
"""

from traits.api import Str, Float, HasTraits, Range, Instance, on_trait_change, Enum
from chaco.api import Plot, AbstractPlotData, ArrayPlotData, VPlotContainer
from traitsui.api import Item, View, VGroup, HSplit, ScrubberEditor, VSplit
from enable.api import Component, ComponentEditor
from chaco.tools.api import PanTool, ZoomTool
import numpy as np

#添加控件样式：鼠标拖动
scrubber = ScrubberEditor(
    hover_color = 0xFFFFFF,
    active_color = 0xA0CD9E,
    border_color = 0x808080
)

#将freqs的前n项进行合成，计算loops个周期的波形
def fft_combine(freqs, n, loops = 1):
    length = len(freqs) * loops
    data = np.zeros(length)
    index = loops * np.arange(0, length, 1.0) / length * (2 * np.pi)
    for k, p in enumerate(freqs[:n]):
        if k != 0:
            p *= 2
        data += np.real(p) * np.cos(k*index)
        data -= np.imag(p) * np.sin(k*index)
    return index, data

class TriangleWave(HasTraits):
    low = Float(0.02)
    hi = Float(1.0)
    wave_width = Range("low", "hi", 0.5)
    #顶点C坐标
    length_c = Range("low", "wave_width", 0.5)
    height_c = Range(-10.0, 10.0, 10.0)
    #FFT的取样点
    fftsize = Enum([(2**x) for x in range(6, 12)])
    #FFT频谱图x轴上限值
    fft_graph_up_limit = Range(0, 400, 20)
    #fft_parameters = []
    #用于显示FFT的结果
    peak_list = Str
    #采用多少个频率进行合成
    N = Range(1, 40, 4)
    #保存绘图数据的对象
    plot_data = Instance(AbstractPlotData)
    #绘制波形图的容器
    plot_wave = Instance(Component)
    #绘制FFT频谱图的容器
    plot_fft = Instance(Component)
    #两个容器的框架
    container = Instance(Component)
    #设置用户界面视图，一定要指定窗口大小
    view = View(
        HSplit(
            VSplit(
                VGroup(
                    Item("wave_width", editor = scrubber, label = u"波形宽度"),
                    Item("length_c", editor = scrubber, label = u"最高点x坐标"),
                    Item("height_c", editor = scrubber, label = u"最高点y坐标"),
                    Item("fft_graph_up_limit", label = u"频谱图范围"),
                    Item("fftsize", label = u"FFT点数"),
                    Item("N", label = u"合成波频率数")                
                ),
                Item("peak_list", style = "custom", show_label = False, width = 300, height = 250)
            ),
            VGroup(
                Item("container", editor = ComponentEditor(size = (600, 500)), show_label = False),            
                orientation = "vertical"            
            )
        ),
        resizable = True, width = 800, height = 600, title = u"三角波FFT演示"
    )
    
    #创建绘图辅助函数
    def _create_plot(self, data, name, type = "line"):
        p = Plot(self.plot_data)
        p.plot(data, name = name, title = name, type = type)
        p.tools.append(PanTool(p))
        zoom = ZoomTool(compoment = p, tool_mode = "box", always_on = False)
        p.overlays.append(zoom)
        p.title = name
        return p
        
    def __init__(self):
        #调用父类初始化函数
        super(TriangleWave, self).__init__()
        #创建绘图数据集
        self.plot_data = ArrayPlotData(x=[], y=[], f=[], p=[], x2=[], y2=[])
        #创建垂直排列绘图容器
        self.container = VPlotContainer()
        #创建波形图，绘制原始波形(x, y)和合成波形(x2, y2)
        self.plot_wave = self._create_plot(("x", "y"), "Triangle Wave")
        self.plot_wave.plot(("x2", "y2"), color = "red")
        #创建f与p的频谱图
        self.plot_fft = self._create_plot(("f", "p"), "FFT", type = "scatter")
        #添加到垂直容器中
        self.container.add(self.plot_wave)
        self.container.add(self.plot_fft)
        #标题设置
        self.plot_wave.x_axis.title = "Samples"
        self.plot_fft.x_axis.title = "Frequency pins"
        self.plot_fft.y_axis.title = "dB"
        
        #设fftsize为1024
        self.fftsize = 1024
        
    #FFT频谱图x轴上限值改变事件处理函数
    def _fft_graph_up_limit_changed(self):
        self.plot_fft.x_axis.mapper.range.high = self.fft_graph_up_limit
        
    def _N_changed(self):
        self.plot_sin_combine() #在后面定义
        
    #多个trait属性改变时进行相同的处理函数
    @on_trait_change("wave_width, length_c, height_c, fftsize")
    def update_plot(self):
        #三角波
        global y_data
        x_data = np.arange(0, 1.0, 1.0/self.fftsize)
        func = self.triangle_func()
        y_data = np.cast["float64"](func(x_data))
        #计算频谱
        fft_parameters = np.fft.fft(y_data) / len(y_data)
        #计算振幅
        fft_data = np.clip(20*np.log10(np.abs(fft_parameters))[:self.fftsize/2+1], -120, 120)
        #将结果写入数据集
        self.plot_data.set_data("x", np.arange(0, self.fftsize))
        self.plot_data.set_data("y", y_data)
        self.plot_data.set_data("f", np.arange(0, len(fft_data)))
        self.plot_data.set_data("p", fft_data)
        #显示两个周期
        self.plot_data.set_data("x2", np.arange(0, 2*self.fftsize))
        #更新频谱图x轴上限
        self._fft_graph_up_limit_changed()
        #输出功率大于-80dB的
        peak_index = (fft_data > -80)
        peak_value = fft_data[peak_index][:20]
        result = []
        for f, v in zip(np.flatnonzero(peak_index), peak_value):
            result.append("%s : %s" %(f, v))
        self.peak_list = "\n".join(result)
        #保存当前FFT结果，并计算正弦合成波
        self.fft_parameters = fft_parameters
        self.plot_sin_combine()
        
    #计算正弦合成的两个周期
    def plot_sin_combine(self):
        index, data = fft_combine(self.fft_parameters, self.N, 2)
        self.plot_data.set_data("y2", data)
        
    #返回一个ufunc计算指定参数的三角波
    def triangle_func(self):
        c = self.wave_width
        c0 = self.length_c
        hc = self.height_c
        
        def trifunc(x):
            x = x - int(x)
            if x >=c:
                r = 0.0
            elif x < c0:
                r = x / c0 * hc
            else:
                r = (c - x) / (c - c0) * hc
            return r
            
        return np.frompyfunc(trifunc, 1, 1)
        
if __name__ == "__main__":
    triangle = TriangleWave()
    triangle.configure_traits()