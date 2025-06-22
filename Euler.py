import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import math


# ამ კოდში მოძრაობა აღიქმება, როგორც ტელეფონის კურსორის გადაადგილება.არ ვიცოდი ეს კვატერნიონები ზუსტად რის მოძრაობას გამოხატავდა.
class QuaternionToScreen:
    def __init__(self, screen_width=800, screen_height=600, max_tilt_degrees=45):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_center_x = screen_width / 2
        self.screen_center_y = screen_height / 2

        self.max_tilt_radians = math.radians(max_tilt_degrees)
        self.sensitivity_x = screen_width / (2 * self.max_tilt_radians)
        self.sensitivity_y = screen_height / (2 * self.max_tilt_radians)
        self.alpha = 0.2

        self.prev_x = self.screen_center_x
        self.prev_y = self.screen_center_y
        self.dead_zone = 1

    def quaternion_to_euler(self, w, x, y, z):
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def quaternion_to_screen_coords(self, w, x, y, z, use_smoothing=True):
        roll, pitch, yaw = self.quaternion_to_euler(w, x, y, z)

        roll_clamped = max(-self.max_tilt_radians, min(self.max_tilt_radians, roll))
        screen_x = self.screen_center_x + (roll_clamped * self.sensitivity_x)

        pitch_clamped = max(-self.max_tilt_radians, min(self.max_tilt_radians, pitch))
        screen_y = self.screen_center_y - (pitch_clamped * self.sensitivity_y)

        dx = screen_x - self.screen_center_x
        dy = screen_y - self.screen_center_y
        distance_from_center = math.sqrt(dx * dx + dy * dy)

        if distance_from_center < self.dead_zone:
            screen_x = self.screen_center_x
            screen_y = self.screen_center_y

        if use_smoothing:
            screen_x = self.alpha * screen_x + (1 - self.alpha) * self.prev_x
            screen_y = self.alpha * screen_y + (1 - self.alpha) * self.prev_y
            self.prev_x = screen_x
            self.prev_y = screen_y

        screen_x = max(0, min(self.screen_width, screen_x))
        screen_y = max(0, min(self.screen_height, screen_y))

        return screen_x, screen_y

    def get_tilt_info(self, w, x, y, z):
        roll, pitch, yaw = self.quaternion_to_euler(w, x, y, z)
        return {
            'roll_deg': math.degrees(roll),
            'pitch_deg': math.degrees(pitch),
            'yaw_deg': math.degrees(yaw),
            'roll_rad': roll,
            'pitch_rad': pitch,
            'yaw_rad': yaw
        }

def load_quaternions_from_file(filepath):
    quaternions = []
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split(',')
                w = float(parts[0].split(':')[1].strip())
                x = float(parts[1].split(':')[1].strip())
                y = float(parts[2].split(':')[1].strip())
                z = float(parts[3].split(':')[1].strip())
                quaternions.append((w, x, y, z))
            except (IndexError, ValueError):
                print(f"Skipping malformed line: {line}")
    return quaternions

def run_visualization():
    converter = QuaternionToScreen(screen_width=800, screen_height=600)
    quaternion_data = load_quaternions_from_file("quaternion.txt")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.set_xlim(0, converter.screen_width)
    ax1.set_ylim(0, converter.screen_height)
    ax1.set_aspect('equal')
    ax1.set_title('Screen Movement (2D)')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.grid(True, alpha=0.3)

    center_circle = Circle((converter.screen_center_x, converter.screen_center_y),
                           converter.dead_zone, fill=False, color='red', alpha=0.5)
    ax1.add_patch(center_circle)

    cursor, = ax1.plot([], [], 'bo', markersize=10, label='Cursor')
    trail, = ax1.plot([], [], 'b-', alpha=0.3, linewidth=1)

    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_title('Phone Orientation Info')
    ax2.axis('off')

    quat_text = ax2.text(0.05, 0.9, '', transform=ax2.transAxes, fontsize=10,
                         verticalalignment='top', family='monospace')
    euler_text = ax2.text(0.05, 0.6, '', transform=ax2.transAxes, fontsize=10,
                          verticalalignment='top', family='monospace')
    screen_text = ax2.text(0.05, 0.3, '', transform=ax2.transAxes, fontsize=10,
                           verticalalignment='top', family='monospace')

    trail_x = []
    trail_y = []
    max_trail_length = 100

    def animate(frame):
        if frame < len(quaternion_data):
            w, x, y, z = quaternion_data[frame]
        else:
            w, x, y, z = quaternion_data[-1]

        screen_x, screen_y = converter.quaternion_to_screen_coords(w, x, y, z)
        tilt_info = converter.get_tilt_info(w, x, y, z)

        cursor.set_data([screen_x], [screen_y])

        trail_x.append(screen_x)
        trail_y.append(screen_y)
        if len(trail_x) > max_trail_length:
            trail_x.pop(0)
            trail_y.pop(0)
        trail.set_data(trail_x, trail_y)

        quat_text.set_text(f'Quaternion:\nw: {w:.3f}\nx: {x:.3f}\ny: {y:.3f}\nz: {z:.3f}')
        euler_text.set_text(f'Euler Angles (degrees):\nRoll:  {tilt_info["roll_deg"]:6.1f}°\n'
                            f'Pitch: {tilt_info["pitch_deg"]:6.1f}°\n'
                            f'Yaw:   {tilt_info["yaw_deg"]:6.1f}°')
        screen_text.set_text(f'Screen Coordinates:\nX: {screen_x:6.1f}\nY: {screen_y:6.1f}')

        return cursor, trail, quat_text, euler_text, screen_text

    anim = animation.FuncAnimation(fig, animate, frames=len(quaternion_data),
                                   interval=100, blit=True)
    plt.tight_layout()
    plt.show()
    return anim


# 
def test_quaternion_conversion():
    converter = QuaternionToScreen()
    print("Testing Quaternion to Screen Conversion")
    print("=" * 50)

    test_cases = [
        [1, 0, 0, 0, "No rotation (identity)"],
        [0.966, 0.259, 0, 0, "Roll right 30°"],
        [0.966, -0.259, 0, 0, "Roll left 30°"],
        [0.966, 0, 0.259, 0, "Pitch forward 30°"],
        [0.966, 0, -0.259, 0, "Pitch backward 30°"],
        [0.933, 0.183, 0.183, 0.183, "Combined rotation"],
    ]

    for w, x, y, z, description in test_cases:
        screen_x, screen_y = converter.quaternion_to_screen_coords(w, x, y, z, use_smoothing=False)
        tilt_info = converter.get_tilt_info(w, x, y, z)
        print(f"\n{description}")
        print(f"Quaternion: w={w:.3f}, x={x:.3f}, y={y:.3f}, z={z:.3f}")
        print(f"Euler: Roll={tilt_info['roll_deg']:6.1f}°, Pitch={tilt_info['pitch_deg']:6.1f}°, Yaw={tilt_info['yaw_deg']:6.1f}°")
        print(f"Screen: X={screen_x:6.1f}, Y={screen_y:6.1f}")

if __name__ == "__main__":
    converter = QuaternionToScreen()

    # Run test conversions
    test_quaternion_conversion()

    print("\nStarting interactive visualization...")
    print("Close the plot window to exit.")
    try:
        anim = run_visualization()
    except KeyboardInterrupt:
        print("\nVisualization stopped.")


