import numpy as np
import math


# აქ ვცდილობ შევადარო ყველა მეთოდი,რომ ავარჩიო საუკეთესო. თუმცა,როგორც მანამდე აღვნიშნე,
# არსებობს გარკვეული შემთხვევები,როდესაც ყველა მათგანს აქვს გამოყენების პოტენციალი.

# ჩემთვის ეილერის მეთოდი მეტად ინტუიციური და გასაგებია, შედარების შედეგადაც აღმოჩნდა,რომ სხვა მეთოდებთან შედარებით
# უფრო სწრაფი და აკურატულია.
# მიუხედავად იმისა,რომ Rotation Matrix ნელია, შეიძლება ზუსტი 3D მოძრაობების გამოსასახად უკეთესი იყოს.
# თუ გვეცოდინება მთლიანი სურათი, მაშინ უფრო მარტივად გადავწყვეტთ რომელი მეთოდია შესაბამისი გამოსაყენებლად.

class ImprovedQuaternionConverter:
    def __init__(self, screen_width=100, screen_height=100, max_tilt_degrees=45):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_center_x = screen_width / 2
        self.screen_center_y = screen_height / 2
        self.max_tilt_radians = math.radians(max_tilt_degrees)
        
        # Different sensitivity for X and Y if needed
        self.sensitivity_x = screen_width / (2 * self.max_tilt_radians)
        self.sensitivity_y = screen_height / (2 * self.max_tilt_radians)
        
        # Smoothing parameters
        self.alpha = 0.7
        self.prev_x = self.screen_center_x
        self.prev_y = self.screen_center_y
        self.dead_zone = 10

    # Method 1: 
    def euler_method(self, w, x, y, z, use_smoothing=True):
        # Convert to Euler angles
        roll, pitch, yaw = self.quaternion_to_euler(w, x, y, z)
        
        # Map to screen coordinates
        roll_clamped = max(-self.max_tilt_radians, min(self.max_tilt_radians, roll))
        pitch_clamped = max(-self.max_tilt_radians, min(self.max_tilt_radians, pitch))
        
        screen_x = self.screen_center_x + (roll_clamped * self.sensitivity_x)
        screen_y = self.screen_center_y - (pitch_clamped * self.sensitivity_y)
        
        return self.apply_post_processing(screen_x, screen_y, use_smoothing)

    # Method 2: Direct quaternion component mapping
    def direct_method(self, w, x, y, z, use_smoothing=True):
        """
        Direct mapping from quaternion components - FAST but less intuitive
        """
        # Scale quaternion components directly
        screen_x = self.screen_center_x + (x * self.sensitivity_x * 2)
        screen_y = self.screen_center_y + (y * self.sensitivity_y * 2)
        
        return self.apply_post_processing(screen_x, screen_y, use_smoothing)

    # Method 3: Rotation matrix projection
    def rotation_matrix_method(self, w, x, y, z, use_smoothing=True):
        """
        Project reference vector through rotation matrix - PRECISE
        """
        # Convert quaternion to rotation matrix
        R = self.quaternion_to_rotation_matrix(w, x, y, z)
        
        # Project reference vector [0, 0, 1] (pointing up)
        reference_vector = np.array([0, 0, 1])
        rotated_vector = R @ reference_vector
        
        # Map X and Y components to screen
        screen_x = self.screen_center_x + (rotated_vector[0] * self.sensitivity_x)
        screen_y = self.screen_center_y - (rotated_vector[1] * self.sensitivity_y)
        
        return self.apply_post_processing(screen_x, screen_y, use_smoothing)

    # Method 4: Spherical coordinate projection
    def spherical_method(self, w, x, y, z, use_smoothing=True):
        # Calculate horizontal angle
        azimuth = math.atan2(2*(w*z + x*y), 1-2*(y*y + z*z))
        
        # Calculate vertical angle
        sin_elevation = 2*(w*y - z*x)
        if abs(sin_elevation) >= 1:
            elevation = math.copysign(math.pi/2, sin_elevation)
        else:
            elevation = math.asin(sin_elevation)
        
        # Clamp angles
        azimuth_clamped = max(-self.max_tilt_radians, min(self.max_tilt_radians, azimuth))
        elevation_clamped = max(-self.max_tilt_radians, min(self.max_tilt_radians, elevation))
        
        screen_x = self.screen_center_x + (azimuth_clamped * self.sensitivity_x)
        screen_y = self.screen_center_y - (elevation_clamped * self.sensitivity_y)
        
        return self.apply_post_processing(screen_x, screen_y, use_smoothing)

    # Method 5: Weighted hybrid approach
    def hybrid_method(self, w, x, y, z, use_smoothing=True):
        # Get results from different methods
        euler_x, euler_y = self.euler_method(w, x, y, z, False)
        sphere_x, sphere_y = self.spherical_method(w, x, y, z, False)
        
        # Weight the results (favor Euler for normal use)
        weight_euler = 0.8
        weight_sphere = 0.2
        
        screen_x = euler_x * weight_euler + sphere_x * weight_sphere
        screen_y = euler_y * weight_euler + sphere_y * weight_sphere
        
        return self.apply_post_processing(screen_x, screen_y, use_smoothing)

    # Helper methods
    def quaternion_to_euler(self, w, x, y, z):
        """Convert quaternion to Euler angles"""
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

    def quaternion_to_rotation_matrix(self, w, x, y, z):
        """Convert quaternion to 3x3 rotation matrix"""
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])

    def apply_post_processing(self, screen_x, screen_y, use_smoothing):
        """Apply dead zone, smoothing, and bounds checking"""
        # Apply dead zone
        dx = screen_x - self.screen_center_x
        dy = screen_y - self.screen_center_y
        distance_from_center = math.sqrt(dx*dx + dy*dy)
        
        if distance_from_center < self.dead_zone:
            screen_x = self.screen_center_x
            screen_y = self.screen_center_y
        
        # Apply smoothing
        if use_smoothing:
            screen_x = self.alpha * screen_x + (1 - self.alpha) * self.prev_x
            screen_y = self.alpha * screen_y + (1 - self.alpha) * self.prev_y
            self.prev_x = screen_x
            self.prev_y = screen_y
        
        # Ensure coordinates stay within bounds
        screen_x = max(0, min(self.screen_width, screen_x))
        screen_y = max(0, min(self.screen_height, screen_y))
        
        return screen_x, screen_y

    def compare_methods(self, w, x, y, z):
        """Compare all methods side by side"""
        methods = {
            'Euler': self.euler_method,
            'Direct': self.direct_method,
            'Rotation Matrix': self.rotation_matrix_method,
            'Spherical': self.spherical_method,
            'Hybrid': self.hybrid_method
        }
        
        results = {}
        for name, method in methods.items():
            x_coord, y_coord = method(w, x, y, z, use_smoothing=False)
            results[name] = (x_coord, y_coord)
        
        return results

# Performance comparison
def benchmark_methods():
    """Benchmark different conversion methods"""
    import time
    
    converter = ImprovedQuaternionConverter()
    
    # Test quaternions
    test_quaternions = [
        (1, 0, 0, 0),  # Identity
        (0.966, 0.259, 0, 0),  # Roll 30°
        (0.966, 0, 0.259, 0),  # Pitch 30°
        (0.933, 0.183, 0.183, 0.183),  # Combined
    ]
    
    methods = {
        'Euler': converter.euler_method,
        'Direct': converter.direct_method,
        'Rotation Matrix': converter.rotation_matrix_method,
        'Spherical': converter.spherical_method,
        'Hybrid': converter.hybrid_method
    }
    
    print("Performance Benchmark (1000 iterations each):")
    print("=" * 50)
    
    for method_name, method_func in methods.items():
        start_time = time.time()
        
        for _ in range(1000):
            for w, x, y, z in test_quaternions:
                method_func(w, x, y, z, use_smoothing=False)
        
        end_time = time.time()
        elapsed = (end_time - start_time) * 1000  # Convert to milliseconds
        
        print(f"{method_name:15}: {elapsed:6.2f} ms")

# Usage example
if __name__ == "__main__":
    converter = ImprovedQuaternionConverter()
    
    # Test with a sample quaternion
    w, x, y, z = 0.966, 0.259, 0.1, 0.05  # Sample rotation
    
    print("Comparison of Different Conversion Methods:")
    print("=" * 50)
    
    results = converter.compare_methods(w, x, y, z)
    for method_name, (screen_x, screen_y) in results.items():
        print(f"{method_name:20}: X={screen_x:6.1f}, Y={screen_y:6.1f}")
    
    print("\n")
    benchmark_methods()
