import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def add_box(ax, center, size, color, edgecolor, alpha=1.0, dimensions=""):
    # Generate the vertices of a 3D rectangular box
    r = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0],
                  [-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1]]) * size + center
    faces = [[r[0], r[1], r[5], r[4]], [r[1], r[2], r[6], r[5]], [r[2], r[3], r[7], r[6]], [r[3], r[0], r[4], r[7]],
             [r[4], r[5], r[6], r[7]], [r[0], r[3], r[2], r[1]]]

    # Plot faces of the box
    for f in faces:
        ax.add_collection3d(Poly3DCollection([f], color=color, edgecolor=edgecolor, linewidth=0.5, alpha=alpha))
    
    # Add dimensions below the box
    ax.text(center[0], center[1] - size[1]/2 - 0.1, center[2], f"{dimensions}", color='black', ha='center', va='top', fontsize=9)

def add_arrow(ax, start, end):
    # Add an arrow to show data flow between layers
    ax.quiver(start[0], start[1], start[2], end[0]-start[0], end[1]-start[1], end[2]-start[2], 
              color='black', arrow_length_ratio=0.3)

def draw_cnn_1d(ax):
    # Define layer positions, dimensions, and labels
    layers = [
        {'type': 'Input', 'dimensions': '13x26', 'height': 1.4, 'color': '#ccccff'},
        {'type': 'Conv1D 64', 'dimensions': '13x26x64', 'height': 1.2, 'color': '#f27980'},
        {'type': 'MaxPooling1D', 'dimensions': '6x13x64', 'height': 1.0, 'color': '#ffcc99'},
        {'type': 'Conv1D 128', 'dimensions': '6x13x128', 'height': 0.8, 'color': '#f27980'},
        {'type': 'MaxPooling1D', 'dimensions': '3x6x128', 'height': 0.6, 'color': '#ffcc99'},
        {'type': 'Bi-LSTM', 'dimensions': '3x6x256', 'height': 0.6, 'color': '#95c7ff'},
        {'type': 'Dropout 0.5', 'dimensions': 'Drop', 'height': 0.4, 'color': '#ffcccc'},
        {'type': 'Bi-LSTM', 'dimensions': '256', 'height': 0.4, 'color': '#95c7ff'},
        {'type': 'Dropout 0.5', 'dimensions': 'Drop', 'height': 0.2, 'color': '#ffcccc'},
        {'type': 'Dense 256', 'dimensions': '256', 'height': 0.2, 'color': '#99ff99'},
        {'type': 'Dropout 0.3', 'dimensions': 'Drop', 'height': 0.2, 'color': '#ffcccc'},
        {'type': 'Output', 'dimensions': 'Classes', 'height': 0.2, 'color': '#99ff99'}
    ]
    
    x_positions = np.linspace(0, 20, len(layers))  # Adjusted for wider spacing
    y_positions = [0] * len(layers)
    z_positions = [0] * len(layers)
    layer_widths = [0.5] * len(layers)
    layer_depths = [0.1] * len(layers)
    
    # Draw layers
    for i, layer in enumerate(layers):
        center = [x_positions[i], y_positions[i], z_positions[i]]
        size = [layer_widths[i], layer['height'], layer_depths[i]]
        add_box(ax, center, size, layer['color'], edgecolor='black', dimensions=layer['dimensions'])
        if i < len(layers) - 1:  # Draw arrows between layers
            next_center = [x_positions[i+1], y_positions[i+1], z_positions[i+1]]
            start = [center[0] + size[0]/2, center[1], center[2]]
            end = [next_center[0] - size[0]/2, next_center[1], next_center[2]]
            add_arrow(ax, start, end)

    # Add legend for layer types
    for i, layer in enumerate(layers):
        ax.text(x_positions[i], 1.5, 0, f"{layer['type']}", color=layer['color'], ha='center', va='center', fontsize=9)

fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, 21)
ax.set_ylim(-1, 2)  # Adjusted to accommodate the legend
ax.set_zlim(0, 1.5)
ax.axis('off')

draw_cnn_1d(ax)
plt.show()

