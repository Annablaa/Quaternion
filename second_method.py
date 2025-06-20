# Re-import needed packages after kernel reset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# === Step 1: Simulate a stream of random unit quaternions ===
def generate_quaternion():
    u1, u2, u3 = np.random.rand(3)
    w = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    x = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    y = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    z = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return w, x, y, z

# === Step 2: Convert quaternion to forward vector and project to (X, Y) ===
def quaternion_to_forward_vector_xy(w, x, y, z, sensitivity=200):
    # Compute forward vector from quaternion
    fx = 2 * (x * z + w * y)
    fy = 2 * (y * z - w * x)
    fz = 1 - 2 * (x * x + y * y)

    # Project forward vector onto XZ-plane (like screen X/Y)
    X = sensitivity * fx
    Y = sensitivity * fz
    return X, Y

# === Step 3: Visualize the X, Y movement ===
fig, ax = plt.subplots()
ax.set_xlim(-400, 400)
ax.set_ylim(-400, 400)
ax.set_title("Quaternion → Forward Vector → Screen X, Y")
dot, = ax.plot([], [], 'bo', markersize=10)

def init():
    dot.set_data(0, 0)
    return dot,

def update(frame):
    w, x, y, z = generate_quaternion()
    x_screen, y_screen = quaternion_to_forward_vector_xy(w, x, y, z)
    dot.set_data(x_screen, y_screen)
    return dot,

ani = animation.FuncAnimation(
    fig, update, init_func=init,
    frames=200, interval=100, blit=True
)
plt.show()
