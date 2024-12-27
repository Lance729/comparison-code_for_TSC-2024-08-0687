'''数据可视化的顶层文件，被执行文件所调用'''
print("开始注入环境")
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
print("结束注入环境")


# latency_with_changed_times = pd.read_excel(io = r"../results.xlsx", sheet_name = "times as var");print("读取数据完毕")
latency_with_changed_times = pd.read_excel(io = r"results.xlsx", sheet_name = "times as var");print("读取数据完毕")

def extract_data(import_pd, first_col): 
  y1 = list(latency_with_changed_times.iloc[0:58, first_col])
  y2 = list(latency_with_changed_times.iloc[0:58, first_col+1])
  y3 = list(latency_with_changed_times.iloc[0:58, first_col+2])
  y4 = list(latency_with_changed_times.iloc[0:58, first_col+3])
  return y1, y2, y3, y4


for data_clon, line_marker in zip([13,9,5,1], ['^','s','o','*']):
  ''' 该循环不会直接出四张图，必须是关掉一个，下一个才会显示
    @param data_clon :  这里的顺序是excel数据的顺序，但是要注意excel里数据的顺序是相反的，7，15，20，32个节点的场景分别对应的data_clon是 13，9，5，1
    @param line_marker: 曲线上的标记，^给7 nodes，s给15，o给20，*给32  
    '''

  y1, y2, y3, y4 = extract_data(latency_with_changed_times, data_clon); print("提取数据完毕")  # 提取不同部分的数据

  fig_APtimesUE = plt.figure(figsize=(10, 7.5)) #设置图片尺寸
  ax = fig_APtimesUE.add_axes([0.2, 0.17, 0.68, 0.7], aspect='auto')
  ax.set_xlim(-1, 30) # 设置坐标轴范围
  ax.set_ylim(10, 95)
  # plt.xticks(np.arange(-1, 30, 2.5)) # 刻画x轴刻度
  plt.yticks(np.arange(10, 95, 5)) # 刻画y轴刻度，每5刻度画条线。
  ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)   #设置格子样式
  # def extract_data(import_pd, first_col):
  #   y1 = list(latency_with_changed_times.iloc[0:58, first_col])
  #   y2 = list(latency_with_changed_times.iloc[0:58, first_col+1])
  #   y3 = list(latency_with_changed_times.iloc[0:58, first_col+2])
  #   y4 = list(latency_with_changed_times.iloc[0:58, first_col+3])
  #   return y1, y2, y3, y4
  #
  # y1, y2, y3, y4 = extract_data(latency_with_changed_times, 13)
  x = np.arange(0, 29, 0.5)
  for Y, color, labels in zip(
    [y4, y3, y2, y1], 
    [ '#f4a582','#22B39D','#0571b0',  '#d7191c'],   #['#f4a582','#22B39D','#0571b0',  '#d7191c']分别是柚色，绿，蓝，红
    ["By target UE, ", "By first UE, ","By last ES, ",  "By TaCo (Our work)"]
    ):

    ax.plot(x, Y, 
            linestyle='-',        # 线的风格是实线
            marker=line_marker,           #线上的标志, ^给7 nodes，s给15，o给20，*给32
            markeredgewidth = 0.5,  #圆圈的边的粗细
            ms=8,                 # 线上点的大小
            color=color,          # 线的颜色
            linewidth=2,          # 线的粗细
            alpha=1,              # 线透明度
            label=labels,
            zorder=10,
            # mfc=(1, 0, 0, 0.8)          # 标志中心填的颜色
            )
  # fig.patch.set(linewidth=4, edgecolor='0.5') # 在图片的外圈加一个边框。

  font1 = {'family' : 'Times New Roman',
          'weight' : 'normal',
          'size'   : 14}
  ax.legend(loc='center left',prop=font1, ncol = 4, bbox_to_anchor = (0,1,0.5,0.13), columnspacing=0.5, handletextpad=0.5, labelspacing=0.5)
  plt.rcParams['font.size'] = 14
  plt.xlabel('The comptation ability times ($\eta$) that AP larger than UE.', size = 18)
  plt.ylabel('Inference latency (ms)', size = 18)
  # ax.text(25, 70, s = '7 nodes', zorder=100,
  #         ha='center', va='top', weight='bold', color='k',
  #         style='italic', fontfamily='Courier New',
  #         fontsize = 25
  #         ) # 给图片上加个文字
  plt.show()