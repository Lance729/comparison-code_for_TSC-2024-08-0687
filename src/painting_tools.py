import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter



def plot_energy_consumption(data):
    """
    绘制能耗柱状图，显示不同 Tau 和方案下的能耗情况。
    :param data: 输入的能耗数据字典
    :return: 返回绘制的图形（fig 对象），以便在 Jupyter Notebook 中显示
    """
    # 解析有效节点和tau值
    valid_nodes = ["node3_4G5G", "node15_4G5G", "node32_4G5G"]
    taus = [6, 15, 25]
    methods = ["UE_last", "UE_first",  "ES_last","taco"]
    
    # 创建并绘制每个节点的图
    fig_list = []
    for node in valid_nodes:
        # 检查节点是否存在于数据中
        energy_values = {}
        for tau in taus:
            energy_values[tau] = []
            for method in methods:
                # 修改：从 `energy_consumption` 下提取数据
                energy_values[tau].append(data.get("energy_consumption", {})
                                           .get(f"{node}_tau{tau}", {})
                                           .get(method, {})
                                           .get("energy_total", 0)/1000)

        # 创建新图表
        fig, ax = plt.subplots(figsize=(8, 3))  # 初始化 fig 和 ax
        x = np.arange(len(taus))  # 横坐标位置
        width = 0.15 # 柱宽

        # 为每个方案绘制柱状图
        for i, method in enumerate(methods):
            energy_values_for_method = [energy_values[tau][i] for tau in taus]
            bars = ax.bar(x + i * width, energy_values_for_method, width, label=method, edgecolor="black")

            # 在柱状图上方显示能耗值
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # 设置图表的标签和标题
        # ax.set_xlabel("Tau")
        ax.set_ylabel("Energy Consumption (Ws)")
        ax.set_title(f"Energy Consumption for {node}")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([f"tau={tau}" for tau in taus])
        ax.legend(title="Methods")

        # 调整布局
        plt.tight_layout()

        # 添加到图形列表
        fig_list.append(fig)

    # 返回图形列表
    return fig_list





def create_broken_bar_chart(data_dict):
    """
    绘制断轴柱状图，增强视觉效果。
    
    参数:
        data_dict (dict): 包含以下键值对的字典:
            - data (list): 数据列表
            - colors (list): 颜色列表
            - labels (list): 标签列表
            - xlabel (str): x轴标签
    
    返回:
        fig: matplotlib 图形对象
    """
    # 提取参数
    data = data_dict.get('data', [0.6, 10, 400])
    colors = data_dict.get('colors', ['#1A689C', '#FA7D0F', '#CC2628'])
    labels = data_dict.get('labels', ['A', 'B', 'C'])
    xlabel = data_dict.get('xlabel', 'Time cost for retraining 100 episodes')
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 3))  # 缩小y轴高度
    fig.subplots_adjust(wspace=0.06)  # 调整子图之间的距离
    fig.patch.set_facecolor('#F5F5F5')  # 增加背景颜色

    # 缩减柱宽度
    bar_width = 0.5

    # 绘制柱状图
    for i in range(len(data)):
        ax1.barh(i, data[i], color=colors[i], edgecolor='black', height=bar_width, linewidth=1.5)
        ax2.barh(i, data[i], color=colors[i], edgecolor='black', height=bar_width, linewidth=1.5)

        # 在柱顶端显示数据值
        ax1.text(data[i] + 0.2, i, f'{data[i]:.1f}', va='center', ha='left', fontsize=12)
        ax2.text(data[i] + 0.2, i, f'{data[i]:.1f}', va='center', ha='left', fontsize=12)

    # 设置两个子图的x轴范围
    ax1.set_xlim(0, 20)  # 缩小的比例范围
    ax2.set_xlim(350, 410)  # 放大比例范围

    # 隐藏子图之间的边框
    ax1.spines.right.set_visible(False)
    ax2.spines.left.set_visible(False)
    ax1.yaxis.set_visible(False)  # 隐藏y轴标签
    ax2.yaxis.set_visible(False)

    # 在断轴处绘制斜线
    d = 1.3  # 斜线的长度比例
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)

    # 增加框线宽度
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_edgecolor('black')

    # 添加图例
    fig.legend(handles=[plt.Line2D([0], [0], color=color, lw=4) for color in colors], labels=labels,
               loc='upper center', ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.05), fontsize=14, markerscale=1.5,
               handletextpad=0.5, handlelength=1)

    # 添加x轴标签
    ax1.set_xlabel(xlabel, fontsize=16, weight='bold')
    # 调整布局
    plt.tight_layout()
    return fig



def plot_latency_comparison(latencies_data_taco, latencies_data_dnn, latencies_data_bo):
    """
    绘制延迟对比图，显示不同方案下的延迟曲线。
    
    参数:
        latencies_data_taco (list): taco方案的延迟数据（长度为100）
        latencies_data_dnn (list): dnn方案的延迟数据（长度为100）
        latencies_data_bo (list): bo方案的延迟数据（长度为100）
    
    返回:
        fig: matplotlib 图形对象
    """
    # 数据准备
    x = np.arange(1, len(latencies_data_taco) + 1)  # 假设 x 轴是从 1 到 100 的序列
    data = [latencies_data_taco, latencies_data_dnn, latencies_data_bo]
    colors = ['#CC2628', '#FA7D0F', '#1A689C']
    methods = ['TaCo (Ours)', 'DNN-based DTOO [10]', 'BO-based PTC [9]']

    # 创建图形和子图
    fig, ax = plt.subplots(figsize=(10, 6))
    # fig.patch.set_facecolor('#F5F5F5')  # 增加背景颜色

    # 绘制曲线
    for latencies, color, method in zip(data, colors, methods):
        ax.plot(x, latencies, label=method, color=color, linewidth=4)

    # 设置图表的标签和标题
    ax.set_xlabel('Episodes', fontsize=16, weight='bold')  # x 轴标签
    ax.set_ylabel('Latency of Schemes (ms)', fontsize=16, weight='bold')  # y 轴标签
    ax.set_title('Performance Comparison', fontsize=16, weight='bold')  # 标题



    # 调整图例样式
    ax.legend(title='Methods', fontsize=12, title_fontsize=14)

    # 调整坐标刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=12)
    # 调整布局
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    return fig