import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def add_box(ax, center, size, color, edgecolor, alpha=1.0, dimensions="", alternate=False):
    # Generate the vertices of a 3D rectangular box
    r = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0],
                  [-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1]]) * size + center
    faces = [[r[0], r[1], r[5], r[4]], [r[1], r[2], r[6], r[5]], [r[2], r[3], r[7], r[6]], [r[3], r[0], r[4], r[7]],
             [r[4], r[5], r[6], r[7]], [r[0], r[3], r[2], r[1]]]

    # Plot faces of the box
    for f in faces:
        ax.add_collection3d(Poly3DCollection([f], color=color, edgecolor=edgecolor, linewidth=0.5, alpha=alpha))
    
    # # Adjust the position of dimensions above or below the box based on the 'alternate' flag
    # vertical_position = center[1] + (0.2 if alternate else -0.4)
    # va_align = 'bottom' if alternate else 'top'
    # ax.text(center[0], vertical_position, center[2], dimensions, color='black', ha='center', va=va_align, fontsize=12)

def add_arrow(ax, start, end):
    # Add an arrow to show data flow between layers
    ax.quiver(start[0], start[1], start[2], end[0]-start[0], end[1]-start[1], end[2]-start[2], 
              color='black', arrow_length_ratio=0.3)

def draw_cnn_3d(ax):
    # Define layer positions, dimensions, and labels
    layers = [
        {'type': 'Input', 'dimensions': '13x26x1', 'height': 1.2, 'color': '#ffcccc'},
        {'type': 'Conv1 16, ReLu', 'dimensions': '13x26x16', 'height': 1.0, 'color': '#ccccff'},
        {'type': 'MaxPool', 'dimensions': '13x26x16', 'height': 1.0, 'color': '#ffcc99'},
        {'type': 'Conv2 32, ReLu', 'dimensions': '13x26x32', 'height':0.9, 'color': '#ccccff'},
        {'type': 'MaxPool', 'dimensions': '13x26x32', 'height': 0.9, 'color': '#ffcc99'},
        {'type': 'Conv3 64, ReLu', 'dimensions': '13x26x64', 'height': 0.8, 'color': '#ccccff'},
        {'type': 'MaxPool', 'dimensions': '6x13x64', 'height': 0.6, 'color': '#ffcc99'},
        {'type': 'Conv4 128, ReLu', 'dimensions': '6x13x128', 'height': 0.5, 'color': '#ccccff'},
        {'type': 'MaxPool', 'dimensions': '3x6x128', 'height': 0.3, 'color': '#ffcc99'},
        {'type': 'Conv5 256, ReLu', 'dimensions': '3x6x256', 'height': 0.3, 'color': '#ccccff'},
        {'type': 'MaxPool', 'dimensions': '1x3x256', 'height': 0.2, 'color': '#ffcc99'},
        {'type': 'Flatten', 'dimensions': '', 'height': 0.2, 'color': '#99ff99'},
        {'type': 'Dense 512', 'dimensions': '512', 'height': 0.2, 'color': '#99ff99'},
        {'type': 'Dropout 0.5', 'dimensions': '', 'height': 0.2, 'color': '#ffcccc'},
        {'type': 'Output', 'dimensions': 'Num Classes = 35', 'height': 0.2, 'color': '#99ff99'}
    ]
    
    x_positions = np.linspace(0, 30, len(layers))  # Adjusted for wider spacing
    y_positions = [0] * len(layers)
    z_positions = [0] * len(layers)
    layer_widths = [0.5] * len(layers)
    layer_depths = [0.1] * len(layers)
    
    # Draw layers with alternating label positions
    for i, layer in enumerate(layers):
        alternate = i % 2 == 0
        center = [x_positions[i], y_positions[i], z_positions[i]]
        size = [layer_widths[i], layer['height'], layer_depths[i]]
        add_box(ax, center, size, layer['color'], edgecolor='black', 
                  alternate=alternate)
        if i < len(layers) - 1:  # Draw arrows between layers
            next_center = [x_positions[i+1], y_positions[i+1], z_positions[i+1]]
            start = [center[0] + size[0]/2, center[1], center[2]]
            end = [next_center[0] - size[0]/2, next_center[1], next_center[2]]
            add_arrow(ax, start, end)
        
    # # Connect layers with lines
    # for i in range(len(x_positions)-1):
    #     ax.plot([x_positions[i] + layer_widths[i]/2, x_positions[i+1] - layer_widths[i+1]/2],
    #             [y_positions[i], y_positions[i+1]], [z_positions[i], z_positions[i+1]], 'k--')
        
    # Add legend
    for layer in layers:
        ax.plot([0], [0], [0], 's', markersize=10, label=f"{layer['type']}", color=layer['color'])

    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, 32)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 1.5)
ax.axis('off')

draw_cnn_3d(ax)
plt.show()