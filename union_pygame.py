import numpy as np
import math
from typing import Tuple, Optional
from collections import deque
import pygame
import threading
import time

class QuaternionTo2DMotion:
    
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080, 
                 sensitivity: float = 500.0, smoothing_window: int = 5):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.sensitivity = sensitivity
        
        # Previous quaternion for calculating relative motion
        self.prev_quaternion: Optional[np.ndarray] = None
        
        # Smoothing buffer for delta movements
        self.smoothing_window = smoothing_window
        self.delta_buffer = deque(maxlen=smoothing_window)
        
        # Current screen position (center initially)
        self.screen_x = screen_width // 2
        self.screen_y = screen_height // 2
        
        # Deadzone to prevent jitter from sensor noise
        self.deadzone = 0.001
    
    def _normalize_quaternion(self, q: np.ndarray) -> np.ndarray:
        """Normalize quaternion to unit length."""
        norm = np.linalg.norm(q)
        if norm == 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / norm
    
    def _quaternion_conjugate(self, q: np.ndarray) -> np.ndarray:
        """Calculate quaternion conjugate."""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
            w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
            w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
            w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
        ])
    
    def _quaternion_to_euler(self, q: np.ndarray) -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        w, x, y, z = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
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
    
    def _calculate_delta_quaternion(self, current_q: np.ndarray, previous_q: np.ndarray) -> np.ndarray:
        """Calculate the delta quaternion between two orientations."""
        # Delta = current * conjugate(previous)
        prev_conjugate = self._quaternion_conjugate(previous_q)
        delta_q = self._quaternion_multiply(current_q, prev_conjugate)
        return self._normalize_quaternion(delta_q)
    
    def update_position(self, quaternion: Tuple[float, float, float, float]) -> Tuple[int, int]:
        #Update screen position based on new quaternion data.
        # Convert to numpy array and normalize
        current_q = self._normalize_quaternion(np.array(quaternion))
        
        if self.prev_quaternion is None:
            # Initialize with first quaternion
            self.prev_quaternion = current_q
            return int(self.screen_x), int(self.screen_y)
        
        # Calculate delta quaternion
        delta_q = self._calculate_delta_quaternion(current_q, self.prev_quaternion)
        
        # Convert delta to Euler angles for easier interpretation
        delta_roll, delta_pitch, delta_yaw = self._quaternion_to_euler(delta_q)
        
        # Apply deadzone to reduce jitter
        if abs(delta_pitch) < self.deadzone:
            delta_pitch = 0
        if abs(delta_yaw) < self.deadzone:
            delta_yaw = 0
        
        # Map rotations to screen coordinates
        # Pitch (up/down rotation) -> Y movement (inverted for natural feel)
        # Yaw (left/right rotation) -> X movement
        delta_x = delta_yaw * self.sensitivity
        delta_y = -delta_pitch * self.sensitivity  
        
        # Add to smoothing buffer
        self.delta_buffer.append((delta_x, delta_y))
        
        # Apply smoothing by averaging recent deltas
        if len(self.delta_buffer) > 0:
            avg_delta_x = sum(d[0] for d in self.delta_buffer) / len(self.delta_buffer)
            avg_delta_y = sum(d[1] for d in self.delta_buffer) / len(self.delta_buffer)
        else:
            avg_delta_x = avg_delta_y = 0
        
        # Update screen position
        self.screen_x += avg_delta_x
        self.screen_y += avg_delta_y
        
        # Clamp to screen boundaries
        self.screen_x = max(0, min(self.screen_width - 1, self.screen_x))
        self.screen_y = max(0, min(self.screen_height - 1, self.screen_y))
        
        # Update previous quaternion
        self.prev_quaternion = current_q.copy()
        
        return int(self.screen_x), int(self.screen_y)
    
    def reset_position(self):
        """Reset to center of screen and clear history."""
        self.screen_x = self.screen_width // 2
        self.screen_y = self.screen_height // 2
        self.prev_quaternion = None
        self.delta_buffer.clear()
    
    def set_sensitivity(self, sensitivity: float):
        """Adjust movement sensitivity."""
        self.sensitivity = sensitivity
    
    def get_euler_angles(self, quaternion: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
        """Get Euler angles from quaternion for display purposes."""
        q = self._normalize_quaternion(np.array(quaternion))
        return self._quaternion_to_euler(q)



# Vizualisation part is written by AI
class MotionVisualizer:
    def __init__(self, motion_tracker: QuaternionTo2DMotion):
        self.motion_tracker = motion_tracker
        self.running = False
        
        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((motion_tracker.screen_width, motion_tracker.screen_height))
        pygame.display.set_caption("Quaternion 2D Motion Tracker")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 100, 255)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (200, 200, 200)
        self.GREEN = (0, 200, 0)
        
        # Visualization elements
        self.trail_points = deque(maxlen=100)  # Store recent positions for trail
        self.current_quaternion = (1.0, 0.0, 0.0, 0.0)
        self.demo_running = False
        self.demo_thread = None
        
        # Fonts
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
    
    def _draw_grid(self):
        """Draw a subtle grid on the background."""
        grid_size = 50
        for x in range(0, self.motion_tracker.screen_width, grid_size):
            pygame.draw.line(self.screen, self.LIGHT_GRAY, (x, 0), (x, self.motion_tracker.screen_height))
        for y in range(0, self.motion_tracker.screen_height, grid_size):
            pygame.draw.line(self.screen, self.LIGHT_GRAY, (0, y), (self.motion_tracker.screen_width, y))
    
    def _draw_center_crosshair(self):
        """Draw crosshair at center of screen."""
        center_x = self.motion_tracker.screen_width // 2
        center_y = self.motion_tracker.screen_height // 2
        
        pygame.draw.line(self.screen, self.GRAY, (center_x - 20, center_y), (center_x + 20, center_y), 2)
        pygame.draw.line(self.screen, self.GRAY, (center_x, center_y - 20), (center_x, center_y + 20), 2)
    
    def _draw_trail(self):
        """Draw trail of recent cursor positions."""
        if len(self.trail_points) > 1:
            for i in range(1, len(self.trail_points)):
                # Create fade effect
                alpha_factor = i / len(self.trail_points)
                color_intensity = int(255 * alpha_factor)
                trail_color = (0, int(100 * alpha_factor), color_intensity)
                
                start_pos = self.trail_points[i-1]
                end_pos = self.trail_points[i]
                pygame.draw.line(self.screen, trail_color, start_pos, end_pos, 2)
    
    def _draw_cursor(self, x: int, y: int):
        """Draw the current cursor position."""
        # Main cursor circle
        pygame.draw.circle(self.screen, self.RED, (x, y), 8)
        pygame.draw.circle(self.screen, self.WHITE, (x, y), 8, 2)
        
        # Cursor crosshair
        pygame.draw.line(self.screen, self.RED, (x - 15, y), (x + 15, y), 3)
        pygame.draw.line(self.screen, self.RED, (x, y - 15), (x, y + 15), 3)
    
    def _draw_info_panel(self, x: int, y: int):
        """Draw information panel with current status."""
        # Current position
        pos_text = self.font.render(f"Position: ({x}, {y})", True, self.BLACK)
        self.screen.blit(pos_text, (10, 10))
        
        # Sensitivity
        sens_text = self.font.render(f"Sensitivity: {self.motion_tracker.sensitivity:.0f}", True, self.BLACK)
        self.screen.blit(sens_text, (10, 35))
        
        # Current quaternion
        w, x_q, y_q, z_q = self.current_quaternion
        quat_text = self.small_font.render(f"Quaternion: W:{w:.3f} X:{x_q:.3f} Y:{y_q:.3f} Z:{z_q:.3f}", True, self.BLACK)
        self.screen.blit(quat_text, (10, 60))
        
        # Convert to Euler for display
        roll, pitch, yaw = self.motion_tracker.get_euler_angles(self.current_quaternion)
        euler_text = self.small_font.render(f"Euler: Roll:{math.degrees(roll):.1f}° Pitch:{math.degrees(pitch):.1f}° Yaw:{math.degrees(yaw):.1f}°", True, self.BLACK)
        self.screen.blit(euler_text, (10, 80))
        
        # Demo status
        demo_status = "ON" if self.demo_running else "OFF"
        demo_color = self.GREEN if self.demo_running else self.RED
        demo_text = self.font.render(f"Demo: {demo_status}", True, demo_color)
        self.screen.blit(demo_text, (10, 105))
    
    def _draw_controls(self):
        """Draw control instructions."""
        instructions = [
            "Controls:",
            "SPACE - Start/Stop Demo",
            "R - Reset Position", 
            "↑↓ - Adjust Sensitivity",
            "ESC - Quit"
        ]
        
        for i, instruction in enumerate(instructions):
            color = self.BLACK if i == 0 else self.GRAY
            text = self.small_font.render(instruction, True, color)
            self.screen.blit(text, (self.motion_tracker.screen_width - 200, 10 + i * 20))




    def _generate_demo_quaternion(self, t: float) -> Tuple[float, float, float, float]:
        """Generate smooth demo quaternion movement."""
        # Create smooth sinusoidal motion
        pitch = 0.3 * math.sin(t * 0.5)  # Slow up/down movement
        yaw = 0.4 * math.sin(t * 0.7)    # Slow left/right movement
        roll = 0.1 * math.sin(t * 1.2)   # Slight roll
        
        # Convert Euler to quaternion
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cy, sy = math.cos(yaw/2), math.sin(yaw/2)
        cr, sr = math.cos(roll/2), math.sin(roll/2)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return (w, x, y, z)
    
    def _demo_loop(self):
        """Run demo motion in separate thread."""
        start_time = time.time()
        while self.demo_running and self.running:
            current_time = time.time() - start_time
            demo_quat = self._generate_demo_quaternion(current_time)
            self.current_quaternion = demo_quat
            time.sleep(1/60)  # 60 FPS
    
    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    # Toggle demo
                    if not self.demo_running:
                        self.demo_running = True
                        self.demo_thread = threading.Thread(target=self._demo_loop)
                        self.demo_thread.daemon = True
                        self.demo_thread.start()
                    else:
                        self.demo_running = False
                elif event.key == pygame.K_r:
                    # Reset position
                    self.motion_tracker.reset_position()
                    self.trail_points.clear()
                elif event.key == pygame.K_UP:
                    # Increase sensitivity
                    new_sens = self.motion_tracker.sensitivity + 50
                    self.motion_tracker.set_sensitivity(new_sens)
                elif event.key == pygame.K_DOWN:
                    # Decrease sensitivity
                    new_sens = max(50, self.motion_tracker.sensitivity - 50)
                    self.motion_tracker.set_sensitivity(new_sens)
    
    def run(self):
        """Main visualization loop."""
        print("Quaternion Motion Visualization Started!")
        print("Press SPACE to start the demo")
        
        self.running = True
        
        while self.running:
            self._handle_events()
            
            # Update position with current quaternion (THE CORE METHOD)
            x, y = self.motion_tracker.update_position(self.current_quaternion)
            
            # Add to trail
            self.trail_points.append((x, y))
            
            # Clear screen
            self.screen.fill(self.WHITE)
            
            # Draw all elements
            self._draw_grid()
            self._draw_center_crosshair()
            self._draw_trail()
            self._draw_cursor(x, y)
            self._draw_info_panel(x, y)
            self._draw_controls()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
        
        # Cleanup
        self.demo_running = False
        if self.demo_thread and self.demo_thread.is_alive():
            self.demo_thread.join(timeout=1)
        pygame.quit()


def main():
    # Create motion tracker with reasonable defaults
    motion_tracker = QuaternionTo2DMotion(
        screen_width=1200, 
        screen_height=800, 
        sensitivity=300,
        smoothing_window=5
    )
    
    # Create and run visualizer
    visualizer = MotionVisualizer(motion_tracker)
    visualizer.run()
if __name__ == "__main__":
    main()