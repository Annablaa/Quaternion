
''' This code visually shows how random 3D rotations (quaternions) 
correspond to 2D angle representations and how these would appear as motion 
on a screen.'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Here i used a common method to generate random unit quaternions 
# (which represent 3D rotations) uniformly.
# Quaternions are 4D: (𝑤,𝑥,𝑦,𝑧) and must be of unit length
def generate_quaternion():
    u1, u2, u3 = np.random.rand(3)
    w = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    x = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    y = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    z = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return w, x, y, z

# Convert quaternion to Euler angles (pitch, roll)
def quaternion_to_euler_xy(w, x, y, z, sensitivity=200):
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    # Map to screen X and Y
    X = sensitivity * roll
    Y = sensitivity * pitch
    return X, Y

# Visualize the X, Y movement
fig, ax = plt.subplots()
ax.set_xlim(-400, 400)
ax.set_ylim(-400, 400)
ax.set_title("Quaternion → Euler (Pitch & Roll) → Screen X, Y")
dot, = ax.plot([], [], 'ro', markersize=10)

def init():
    dot.set_data(0, 0)
    return dot,

def update(frame):
    w, x, y, z = generate_quaternion()
    x_screen, y_screen = quaternion_to_euler_xy(w, x, y, z)
    dot.set_data(x_screen, y_screen)
    return dot,

ani = animation.FuncAnimation(fig, update, init_func=init, frames=200, interval=100, blit=True)
plt.show()
