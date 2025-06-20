import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import math

class QuaternionToScreen:
    def __init__(self, screen_width=800, screen_height=600, max_tilt_degrees=45):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_center_x = screen_width / 2
        self.screen_center_y = screen_height / 2
        
        # Maximum tilt angle in radians
        self.max_tilt_radians = math.radians(max_tilt_degrees)
        
        # Sensitivity scaling factors
        self.sensitivity_x = screen_width / (2 * self.max_tilt_radians)
        self.sensitivity_y = screen_height / (2 * self.max_tilt_radians)
        
        # Smoothing factor (0.0 = no smoothing, 1.0 = no filtering)
        self.alpha = 0.7
        
        # Previous smoothed values
        self.prev_x = self.screen_center_x
        self.prev_y = self.screen_center_y
        
        # Dead zone threshold (in pixels)
        self.dead_zone = 10

    def quaternion_to_euler(self, w, x, y, z):
        """
        Convert quaternion to Euler angles (roll, pitch, yaw)
        Returns angles in radians
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

    def quaternion_to_screen_coords(self, w, x, y, z, use_smoothing=True):
        """
        Convert quaternion to 2D screen coordinates
        Uses roll for X movement and pitch for Y movement
        """
        # Convert quaternion to Euler angles
        roll, pitch, yaw = self.quaternion_to_euler(w, x, y, z)
        
        # Map roll to X coordinate (left/right tilt)
        # Clamp to maximum tilt range
        roll_clamped = max(-self.max_tilt_radians, min(self.max_tilt_radians, roll))
        screen_x = self.screen_center_x + (roll_clamped * self.sensitivity_x)
        
        # Map pitch to Y coordinate (forward/backward tilt)
        # Note: Inverting pitch so forward tilt moves cursor up
        pitch_clamped = max(-self.max_tilt_radians, min(self.max_tilt_radians, pitch))
        screen_y = self.screen_center_y - (pitch_clamped * self.sensitivity_y)
        
        # Apply dead zone
        dx = screen_x - self.screen_center_x
        dy = screen_y - self.screen_center_y
        distance_from_center = math.sqrt(dx*dx + dy*dy)
        
        if distance_from_center < self.dead_zone:
            screen_x = self.screen_center_x
            screen_y = self.screen_center_y
        
        # Apply smoothing filter
        if use_smoothing:
            screen_x = self.alpha * screen_x + (1 - self.alpha) * self.prev_x
            screen_y = self.alpha * screen_y + (1 - self.alpha) * self.prev_y
            
            self.prev_x = screen_x
            self.prev_y = screen_y
        
        # Ensure coordinates stay within screen bounds
        screen_x = max(0, min(self.screen_width, screen_x))
        screen_y = max(0, min(self.screen_height, screen_y))
        
        return screen_x, screen_y

    def get_tilt_info(self, w, x, y, z):
        """
        Get human-readable tilt information
        """
        roll, pitch, yaw = self.quaternion_to_euler(w, x, y, z)
        
        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)
        
        return {
            'roll_deg': roll_deg,
            'pitch_deg': pitch_deg,
            'yaw_deg': yaw_deg,
            'roll_rad': roll,
            'pitch_rad': pitch,
            'yaw_rad': yaw
        }

# Demo function to simulate phone sensor data
def generate_sample_quaternions(t):
    """
    Generate sample quaternion data simulating phone movement
    """
    # Simulate gentle swaying motion
    roll_angle = 0.3 * math.sin(t * 0.5)  # Roll left/right
    pitch_angle = 0.2 * math.cos(t * 0.7)  # Pitch forward/back
    yaw_angle = 0.1 * math.sin(t * 0.3)   # Slight rotation
    
    # Convert Euler angles to quaternion
    cr = math.cos(roll_angle * 0.5)
    sr = math.sin(roll_angle * 0.5)
    cp = math.cos(pitch_angle * 0.5)
    sp = math.sin(pitch_angle * 0.5)
    cy = math.cos(yaw_angle * 0.5)
    sy = math.sin(yaw_angle * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return w, x, y, z

# Interactive visualization
def run_visualization():
    """
    Run an interactive visualization showing quaternion to screen mapping
    """
    # Initialize the converter
    converter = QuaternionToScreen(screen_width=800, screen_height=600)
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Screen visualization
    ax1.set_xlim(0, converter.screen_width)
    ax1.set_ylim(0, converter.screen_height)
    ax1.set_aspect('equal')
    ax1.set_title('Screen Movement (2D)')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.grid(True, alpha=0.3)
    
    # Add center point and dead zone
    center_circle = Circle((converter.screen_center_x, converter.screen_center_y), 
                          converter.dead_zone, fill=False, color='red', alpha=0.5)
    ax1.add_patch(center_circle)
    
    # Moving cursor
    cursor, = ax1.plot([], [], 'bo', markersize=10, label='Cursor')
    trail, = ax1.plot([], [], 'b-', alpha=0.3, linewidth=1)
    
    # Quaternion and angle display
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_title('Phone Orientation Info')
    ax2.axis('off')
    
    # Text displays
    quat_text = ax2.text(0.05, 0.9, '', transform=ax2.transAxes, fontsize=10, 
                        verticalalignment='top', family='monospace')
    euler_text = ax2.text(0.05, 0.6, '', transform=ax2.transAxes, fontsize=10, 
                         verticalalignment='top', family='monospace')
    screen_text = ax2.text(0.05, 0.3, '', transform=ax2.transAxes, fontsize=10, 
                          verticalalignment='top', family='monospace')
    
    # Data storage for trail
    trail_x = []
    trail_y = []
    max_trail_length = 100
    
    def animate(frame):
        t = frame * 0.1  # Time step
        
        # Generate sample quaternion data
        w, x, y, z = generate_sample_quaternions(t)
        
        # Convert to screen coordinates
        screen_x, screen_y = converter.quaternion_to_screen_coords(w, x, y, z)
        
        # Get tilt information
        tilt_info = converter.get_tilt_info(w, x, y, z)
        
        # Update cursor position
        cursor.set_data([screen_x], [screen_y])
        
        # Update trail
        trail_x.append(screen_x)
        trail_y.append(screen_y)
        if len(trail_x) > max_trail_length:
            trail_x.pop(0)
            trail_y.pop(0)
        trail.set_data(trail_x, trail_y)
        
        # Update text displays
        quat_text.set_text(f'Quaternion:\nw: {w:.3f}\nx: {x:.3f}\ny: {y:.3f}\nz: {z:.3f}')
        
        euler_text.set_text(f'Euler Angles (degrees):\nRoll:  {tilt_info["roll_deg"]:6.1f}°\n' +
                           f'Pitch: {tilt_info["pitch_deg"]:6.1f}°\n' +
                           f'Yaw:   {tilt_info["yaw_deg"]:6.1f}°')
        
        screen_text.set_text(f'Screen Coordinates:\nX: {screen_x:6.1f}\nY: {screen_y:6.1f}')
        
        return cursor, trail, quat_text, euler_text, screen_text
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=1000, interval=50, blit=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim

# Example usage and testing
def test_quaternion_conversion():
    """
    Test the quaternion to screen conversion with various orientations
    """
    converter = QuaternionToScreen()
    
    print("Testing Quaternion to Screen Conversion")
    print("=" * 50)
    
    # Test cases: [w, x, y, z, description]
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
    # Run tests
    test_quaternion_conversion()
    
    print("\nStarting interactive visualization...")
    print("Close the plot window to exit.")
    
    # Run visualization
    try:
        anim = run_visualization()
    except KeyboardInterrupt:
        print("\nVisualization stopped.")