import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

def plot_energy_consumption(data):
    """
    Plot energy consumption bar charts, showing energy consumption for different Tau and schemes.
    :param data: Input energy consumption data dictionary
    :return: Returns the plotted figure (fig object) for display in Jupyter Notebook
    """
    # Parse valid nodes and tau values
    valid_nodes = ["node3_4G5G", "node15_4G5G", "node32_4G5G"]
    taus = [6, 15, 25]
    methods = ["UE_last", "UE_first",  "ES_last","taco"]
    
    # Create and plot the chart for each node
    fig_list = []
    for node in valid_nodes:
        # Check if the node exists in the data
        energy_values = {}
        for tau in taus:
            energy_values[tau] = []
            for method in methods:
                # Modification: Extract data from `energy_consumption`
                energy_values[tau].append(data.get("energy_consumption", {})
                                           .get(f"{node}_tau{tau}", {})
                                           .get(method, {})
                                           .get("energy_total", 0)/1000)

        # Create a new chart
        fig, ax = plt.subplots(figsize=(8, 3))  # Initialize fig and ax
        x = np.arange(len(taus))  # X-axis positions
        width = 0.15  # Bar width

        # Plot bars for each scheme
        for i, method in enumerate(methods):
            energy_values_for_method = [energy_values[tau][i] for tau in taus]
            bars = ax.bar(x + i * width, energy_values_for_method, width, label=method, edgecolor="black")

            # Display energy values above the bars
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

        # Set chart labels and title
        # ax.set_xlabel("Tau")
        ax.set_ylabel("Energy Consumption (Ws)")
        ax.set_title(f"Energy Consumption for {node}")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([f"tau={tau}" for tau in taus])
        ax.legend(title="Methods")

        # Adjust layout
        plt.tight_layout()

        # Add to figure list
        fig_list.append(fig)

    # Return figure list
    return fig_list

def create_broken_bar_chart(data_dict):
    """
    Plot broken bar chart to enhance visual effect.
    
    Parameters:
        data_dict (dict): Dictionary containing the following key-value pairs:
            - data (list): Data list
            - colors (list): Color list
            - labels (list): Label list
            - xlabel (str): X-axis label
    
    Returns:
        fig: Matplotlib figure object
    """
    # Extract parameters
    data = data_dict.get('data', [0.6, 10, 400])
    colors = data_dict.get('colors', ['#1A689C', '#FA7D0F', '#CC2628'])
    labels = data_dict.get('labels', ['A', 'B', 'C'])
    xlabel = data_dict.get('xlabel', 'Time cost for retraining 100 episodes')
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 3))  # Reduce y-axis height
    fig.subplots_adjust(wspace=0.06)  # Adjust distance between subplots
    fig.patch.set_facecolor('#F5F5F5')  # Add background color

    # Reduce bar width
    bar_width = 0.5

    # Plot bars
    for i in range(len(data)):
        ax1.barh(i, data[i], color=colors[i], edgecolor='black', height=bar_width, linewidth=1.5)
        ax2.barh(i, data[i], color=colors[i], edgecolor='black', height=bar_width, linewidth=1.5)

        # Display data values at the top of the bars
        ax1.text(data[i] + 0.2, i, f'{data[i]:.1f}', va='center', ha='left', fontsize=12)
        ax2.text(data[i] + 0.2, i, f'{data[i]:.1f}', va='center', ha='left', fontsize=12)

    # Set x-axis range for both subplots
    ax1.set_xlim(0, 20)  # Reduced scale range
    ax2.set_xlim(350, 410)  # Enlarged scale range

    # Hide borders between subplots
    ax1.spines.right.set_visible(False)
    ax2.spines.left.set_visible(False)
    ax1.yaxis.set_visible(False)  # Hide y-axis labels
    ax2.yaxis.set_visible(False)

    # Draw diagonal lines at the broken axis
    d = 1.3  # Length ratio of diagonal lines
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)

    # Increase border width
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_edgecolor('black')

    # Add legend
    fig.legend(handles=[plt.Line2D([0], [0], color=color, lw=4) for color in colors], labels=labels,
               loc='upper center', ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.05), fontsize=14, markerscale=1.5,
               handletextpad=0.5, handlelength=1)

    # Add x-axis label
    ax1.set_xlabel(xlabel, fontsize=16, weight='bold')
    # Adjust layout
    plt.tight_layout()
    return fig

def plot_latency_comparison(latencies_data_taco, latencies_data_dnn, latencies_data_bo):
    """
    Plot latency comparison chart, showing latency curves for different schemes.
    
    Parameters:
        latencies_data_taco (list): Latency data for taco scheme (length 100)
        latencies_data_dnn (list): Latency data for dnn scheme (length 100)
        latencies_data_bo (list): Latency data for bo scheme (length 100)
    
    Returns:
        fig: Matplotlib figure object
    """
    # Data preparation
    x = np.arange(1, len(latencies_data_taco) + 1)  # Assume x-axis is a sequence from 1 to 100
    data = [latencies_data_taco, latencies_data_dnn, latencies_data_bo]
    colors = ['#CC2628', '#FA7D0F', '#1A689C']
    methods = ['TaCo (Ours)', 'DNN-based DTOO [10]', 'BO-based PTC [9]']

    # Create figure and subplot
    fig, ax = plt.subplots(figsize=(10, 6))
    # fig.patch.set_facecolor('#F5F5F5')  # Add background color

    # Plot curves
    for latencies, color, method in zip(data, colors, methods):
        ax.plot(x, latencies, label=method, color=color, linewidth=4)

    # Set chart labels and title
    ax.set_xlabel('Episodes', fontsize=16, weight='bold')  # X-axis label
    ax.set_ylabel('Latency of Schemes (ms)', fontsize=16, weight='bold')  # Y-axis label
    ax.set_title('Performance Comparison', fontsize=16, weight='bold')  # Title

    # Adjust legend style
    ax.legend(title='Methods', fontsize=12, title_fontsize=14)

    # Adjust tick label font size
    ax.tick_params(axis='both', which='major', labelsize=12)
    # Adjust layout
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    return fig