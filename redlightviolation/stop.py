import cv2
import json

"""
Interactive tool to mark stop line on traffic video
Click points to define the stop line, press 's' to save
"""

# Global variables
points = []
drawing = False

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for drawing stop line"""
    global points, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        drawing = True
        print(f"Point {len(points)}: ({x}, {y})")
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click to remove last point
        if points:
            removed = points.pop()
            print(f"Removed point: {removed}")

def mark_stop_line(video_path, output_file='stop_line_config.json'):
    """
    Interactive tool to mark stop line on video
    
    Instructions:
    - Left click to add points for the stop line
    - Right click to remove last point
    - Press 's' to save configuration
    - Press 'r' to reset all points
    - Press 'q' to quit without saving
    
    Args:
        video_path: Path to traffic video
        output_file: JSON file to save stop line coordinates
    """
    global points
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        return
    
    # Create a copy for drawing
    display_frame = frame.copy()
    
    # Create window and set mouse callback
    cv2.namedWindow('Mark Stop Line')
    cv2.setMouseCallback('Mark Stop Line', mouse_callback)
    
    print("\n" + "="*60)
    print("STOP LINE MARKING TOOL")
    print("="*60)
    print("Instructions:")
    print("  - LEFT CLICK: Add point to stop line")
    print("  - RIGHT CLICK: Remove last point")
    print("  - Press 'S': Save configuration")
    print("  - Press 'R': Reset all points")
    print("  - Press 'Q': Quit without saving")
    print("="*60 + "\n")
    
    while True:
        # Create fresh copy for drawing
        display_frame = frame.copy()
        
        # Draw existing points
        for i, point in enumerate(points):
            cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
            cv2.putText(display_frame, str(i+1), (point[0]+10, point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw line connecting points
        if len(points) > 1:
            for i in range(len(points)-1):
                cv2.line(display_frame, points[i], points[i+1], (0, 0, 255), 2)
        
        # Add instructions on frame
        cv2.putText(display_frame, f"Points: {len(points)} | S:Save R:Reset Q:Quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Mark Stop Line', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s') or key == ord('S'):
            # Save configuration
            if len(points) < 2:
                print("Error: Need at least 2 points to define a line!")
                continue
            
            config = {
                'stop_line': points,
                'video_path': video_path,
                'frame_width': frame.shape[1],
                'frame_height': frame.shape[0]
            }
            
            with open(output_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"\n✓ Stop line configuration saved to: {output_file}")
            print(f"✓ Total points: {len(points)}")
            break
        
        elif key == ord('r') or key == ord('R'):
            # Reset points
            points = []
            print("Points reset!")
        
        elif key == ord('q') or key == ord('Q'):
            print("Quit without saving")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return points


def load_stop_line_config(config_file='stop_line_config.json'):
    """Load stop line configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"✓ Loaded stop line config from: {config_file}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file {config_file} not found")
        return None


def visualize_stop_line(video_path, config_file='stop_line_config.json', duration=5):
    """
    Visualize the marked stop line on video
    
    Args:
        video_path: Path to video
        config_file: Path to stop line configuration
        duration: How many seconds to show (default 5 seconds)
    """
    config = load_stop_line_config(config_file)
    if not config:
        return
    
    stop_line = config['stop_line']
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames_to_show = fps * duration
    frame_count = 0
    
    print(f"\nVisualizing stop line for {duration} seconds...")
    
    while frame_count < frames_to_show:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw stop line
        for i in range(len(stop_line)-1):
            pt1 = tuple(stop_line[i])
            pt2 = tuple(stop_line[i+1])
            cv2.line(frame, pt1, pt2, (0, 0, 255), 3)
        
        # Add label
        cv2.putText(frame, "STOP LINE", (stop_line[0][0], stop_line[0][1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Stop Line Visualization', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("Visualization complete!")


# ============ USAGE ============

if __name__ == "__main__":
    video_path = 'traffic_video.mp4'  # Change to your video path
    
    # STEP 1: Mark the stop line interactively
    print("Starting stop line marking tool...")
    mark_stop_line(video_path, 'stop_line_config.json')
    
    # STEP 2: Visualize the marked line
    # visualize_stop_line(video_path, 'stop_line_config.json', duration=10)