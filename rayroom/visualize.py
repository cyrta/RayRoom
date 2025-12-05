import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def plot_room(room, filename=None, show=True):
    """
    Plot the room in 3D.
    
    Args:
        room: The Room object.
        filename (str): If provided, save the figure to this path.
        show (bool): If True, show the plot window (blocking).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Walls
    for wall in room.walls:
        # Create a list of vertices for the polygon
        verts = [list(zip(wall.vertices[:,0], wall.vertices[:,1], wall.vertices[:,2]))]
        
        trans = np.mean(wall.material.transmission)
        alpha = max(0.1, 1.0 - trans) # If fully transparent, still show faint
        
        if trans > 0.5:
            alpha = 0.2
            
        color = 'gray'
        if "glass" in wall.material.name.lower():
            color = 'cyan'
            alpha = 0.15
        elif "wood" in wall.material.name.lower():
            color = 'peru'
        elif "brick" in wall.material.name.lower():
            color = 'firebrick'
        elif "concrete" in wall.material.name.lower():
            color = 'lightgray'
            
        poly = Poly3DCollection(verts, alpha=alpha, edgecolor='k', facecolor=color, linewidths=0.5)
        ax.add_collection3d(poly)
        
    # Plot Furniture
    for furn in room.furniture:
        face_polys = []
        for face in furn.faces:
            pts = furn.vertices[face]
            face_polys.append(list(zip(pts[:,0], pts[:,1], pts[:,2])))
            
        color = 'orange'
        if "human" in furn.name.lower() or "person" in furn.name.lower():
            color = 'blue'
            
        poly = Poly3DCollection(face_polys, alpha=0.6, edgecolor='k', facecolor=color, linewidths=0.2)
        ax.add_collection3d(poly)
        
    # Plot Sources
    for src in room.sources:
        ax.scatter(src.position[0], src.position[1], src.position[2], c='red', s=100, marker='^', label=f"Source: {src.name}", depthshade=False)
        
    # Plot Receivers
    for rx in room.receivers:
        ax.scatter(rx.position[0], rx.position[1], rx.position[2], c='green', s=100, marker='o', label=f"Rx: {rx.name}", depthshade=False)
    
    # Auto-scale axes
    all_verts = []
    for w in room.walls: all_verts.extend(w.vertices)
    for f in room.furniture: all_verts.extend(f.vertices)
    all_verts = np.array(all_verts)
    
    if len(all_verts) > 0:
        max_range = np.array([
            all_verts[:,0].max()-all_verts[:,0].min(), 
            all_verts[:,1].max()-all_verts[:,1].min(), 
            all_verts[:,2].max()-all_verts[:,2].min()
        ]).max() / 2.0
        
        mid_x = (all_verts[:,0].max()+all_verts[:,0].min()) * 0.5
        mid_y = (all_verts[:,1].max()+all_verts[:,1].min()) * 0.5
        mid_z = (all_verts[:,2].max()+all_verts[:,2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Create unique legend handles
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys())
    
    plt.title("Room Geometry")
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"Room image saved to {filename}")
        
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_room_2d(room, filename=None, show=True):
    """
    Plot the room in 2D (Top View).
    
    Args:
        room: The Room object.
        filename (str): If provided, save the figure to this path.
        show (bool): If True, show the plot window.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. Plot Floor/Walls Footprint
    # We can iterate walls. If a wall is vertical (normal has z=0), it appears as a line.
    # If a wall is floor (normal has z=1), it appears as a polygon area.
    
    # Draw floor polygon first if identifiable
    floor_walls = [w for w in room.walls if abs(w.normal[2]) > 0.9] # Floor or ceiling
    vertical_walls = [w for w in room.walls if abs(w.normal[2]) < 0.1]
    
    # If we have a floor, fill it
    for wall in floor_walls:
        if wall.vertices[0, 2] < 1.0: # Assume floor is low
            poly = plt.Polygon(wall.vertices[:, :2], fill=True, facecolor='#e6e6e6', edgecolor='none', alpha=0.5, label='Floor')
            ax.add_patch(poly)
            
    # Draw Wall outlines
    for wall in vertical_walls:
        # Project vertices to 2D
        pts = wall.vertices[:, :2]
        # It's a loop in 3D, in 2D it might be a line segment (seen from top)
        # or a rectangle if it has thickness (but here walls are planes)
        # A vertical plane wall projects to a line segment.
        # We can just plot the closed loop, which will look like a line (0 area).
        
        color = 'k'
        if "glass" in wall.material.name.lower(): color = 'cyan'
        
        ax.plot(pts[:,0], pts[:,1], color=color, linewidth=2)
        
    # Plot Furniture (Projected footprint)
    for furn in room.furniture:
        # Find convex hull of vertices in 2D? Or just plot all faces projected?
        # Simple bounding box or just points?
        # Let's project all faces that point up?
        # Or just all vertices projected.
        
        pts = furn.vertices[:, :2]
        # Draw a hull or just scatter?
        # Better: Draw faces.
        
        # Draw faces that are roughly horizontal?
        # Or just fill the footprint.
        # Let's just draw edges of faces.
        for face in furn.faces:
             f_pts = furn.vertices[face][:, :2]
             poly = plt.Polygon(f_pts, fill=True, facecolor='orange', alpha=0.5, edgecolor='darkorange')
             if "person" in furn.name.lower():
                 poly.set_facecolor('blue')
                 poly.set_edgecolor('darkblue')
             ax.add_patch(poly)
             
    # Plot Sources
    for src in room.sources:
        ax.scatter(src.position[0], src.position[1], c='red', s=150, marker='^', label=f"{src.name}", zorder=10)
        ax.annotate(src.name, (src.position[0], src.position[1]), xytext=(5, 5), textcoords='offset points')
        
        # Show orientation if directional
        if hasattr(src, 'orientation') and hasattr(src, 'directivity') and src.directivity != "omnidirectional":
             dx, dy = src.orientation[0], src.orientation[1]
             ax.arrow(src.position[0], src.position[1], dx*0.5, dy*0.5, head_width=0.1, head_length=0.1, fc='red', ec='red', zorder=10)

    # Plot Receivers
    for rx in room.receivers:
        ax.scatter(rx.position[0], rx.position[1], c='green', s=150, marker='o', label=f"{rx.name}", zorder=10)
        ax.annotate(rx.name, (rx.position[0], rx.position[1]), xytext=(5, 5), textcoords='offset points')
        
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Create unique legend handles
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
    plt.title("Room Top View")
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"Room 2D image saved to {filename}")
        
    if show:
        plt.show()
    else:
        plt.close(fig)
