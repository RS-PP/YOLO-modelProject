import cv2
import torch
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import os

"""
Complete Red Light Violation Detection System

Features:
- Vehicle detection using YOLOv5
- Stop line crossing detection
- Manual/automatic signal state tracking
- Violation image capture
- Challan generation
"""

class RedLightViolationDetector:
    def __init__(self, stop_line_config, signal_config=None):
        """
        Initialize the violation detector
        
        Args:
            stop_line_config: Path to stop line JSON config
            signal_config: Path to signal timing JSON config (optional)
        """
        # Load YOLOv5 model
        print("Loading YOLOv5 model...")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        # Vehicle classes from COCO dataset
        self.VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # Load stop line configuration
        with open(stop_line_config, 'r') as f:
            config = json.load(f)
            self.stop_line = config['stop_line']
        
        # Signal state (RED or GREEN)
        self.signal_state = "GREEN"
        
        # Load signal timing if provided
        self.signal_config = None
        if signal_config and os.path.exists(signal_config):
            with open(signal_config, 'r') as f:
                self.signal_config = json.load(f)
        
        # Violation tracking
        self.violations = []
        self.violation_count = 0
        
        # Create output directories
        Path("violations").mkdir(exist_ok=True)
        Path("violations/images").mkdir(exist_ok=True)
        
        print("✓ Violation detector initialized!")
    
    def point_below_line(self, point, line_points):
        """
        Check if a point is below the stop line
        Uses cross product to determine which side of line the point is on
        
        Args:
            point: (x, y) coordinates of vehicle bottom center
            line_points: List of points defining the stop line
        
        Returns:
            True if point crossed the line (is below/after it)
        """
        # Use first two points of line for simplicity
        if len(line_points) < 2:
            return False
        
        x1, y1 = line_points[0]
        x2, y2 = line_points[1]
        px, py = point
        
        # Cross product to determine side
        # Positive = below line, Negative = above line
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        
        return cross > 0
    
    def get_vehicle_bottom_center(self, bbox):
        """Get bottom center point of vehicle bounding box"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        bottom_y = y2  # Bottom of the box
        return (int(center_x), int(bottom_y))
    
    def draw_stop_line(self, frame):
        """Draw stop line on frame"""
        for i in range(len(self.stop_line) - 1):
            pt1 = tuple(self.stop_line[i])
            pt2 = tuple(self.stop_line[i + 1])
            
            # Color based on signal state
            color = (0, 0, 255) if self.signal_state == "RED" else (0, 255, 0)
            cv2.line(frame, pt1, pt2, color, 3)
        
        # Label
        label_color = (0, 0, 255) if self.signal_state == "RED" else (0, 255, 0)
        cv2.putText(frame, f"STOP LINE - {self.signal_state}", 
                   (self.stop_line[0][0], self.stop_line[0][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
    
    def update_signal_state_manual(self, frame_number, frame_time):
        """
        Update signal state based on manual configuration
        
        Args:
            frame_number: Current frame number
            frame_time: Current time in seconds
        """
        if not self.signal_config:
            return
        
        # Check if current time falls in RED signal periods
        for period in self.signal_config.get('red_periods', []):
            start = period['start']
            end = period['end']
            if start <= frame_time <= end:
                self.signal_state = "RED"
                return
        
        self.signal_state = "GREEN"
    
    def save_violation(self, frame, vehicle_info, frame_number, timestamp):
        """Save violation details and image"""
        self.violation_count += 1
        
        violation_data = {
            'violation_id': self.violation_count,
            'timestamp': timestamp,
            'frame_number': frame_number,
            'vehicle_type': vehicle_info['type'],
            'confidence': vehicle_info['confidence'],
            'bbox': vehicle_info['bbox']
        }
        
        # Save violation image
        img_filename = f"violations/images/violation_{self.violation_count}_{timestamp}.jpg"
        
        # Crop vehicle region
        x1, y1, x2, y2 = vehicle_info['bbox']
        vehicle_img = frame[y1:y2, x1:x2]
        cv2.imwrite(img_filename, vehicle_img)
        
        violation_data['image_path'] = img_filename
        
        self.violations.append(violation_data)
        
        print(f"⚠️  VIOLATION #{self.violation_count} detected at {timestamp}")
        
        return violation_data
    
    def generate_challan_report(self, output_file='violation_report.json'):
        """Generate final challan report"""
        report = {
            'total_violations': self.violation_count,
            'violations': self.violations,
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"\n{'='*60}")
        print(f"CHALLAN REPORT GENERATED")
        print(f"{'='*60}")
        print(f"Total Violations: {self.violation_count}")
        print(f"Report saved to: {output_file}")
        print(f"Violation images saved in: violations/images/")
        print(f"{'='*60}\n")
    
    def process_video(self, video_path, output_video='output_violations.mp4', 
                     show_display=True):
        """
        Process video and detect violations
        
        Args:
            video_path: Path to input video
            output_video: Path to save output video
            show_display: Whether to show live display
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        print(f"\nProcessing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        print(f"Total Frames: {total_frames}\n")
        
        frame_number = 0
        tracked_vehicles = {}  # Track vehicles across frames
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            frame_time = frame_number / fps
            
            # Update signal state
            self.update_signal_state_manual(frame_number, frame_time)
            
            # Draw stop line
            self.draw_stop_line(frame)
            
            # Detect vehicles
            results = self.model(frame)
            detections = results.xyxy[0].cpu().numpy()
            
            vehicle_count = 0
            
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                
                if int(cls) in self.VEHICLE_CLASSES:
                    vehicle_count += 1
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    
                    # Get vehicle bottom center point
                    vehicle_point = self.get_vehicle_bottom_center(bbox)
                    
                    # Check if vehicle crossed stop line during RED signal
                    crossed_line = self.point_below_line(vehicle_point, self.stop_line)
                    
                    # Violation detection
                    is_violation = self.signal_state == "RED" and crossed_line
                    
                    # Draw bounding box
                    color = (0, 0, 255) if is_violation else (0, 255, 0)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    
                    # Draw vehicle tracking point
                    cv2.circle(frame, vehicle_point, 5, color, -1)
                    
                    # Label
                    vehicle_type = self.model.names[int(cls)]
                    label = f"{vehicle_type} {conf:.2f}"
                    if is_violation:
                        label += " - VIOLATION!"
                    
                    cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Save violation
                    if is_violation:
                        vehicle_info = {
                            'type': vehicle_type,
                            'confidence': float(conf),
                            'bbox': bbox
                        }
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        self.save_violation(frame, vehicle_info, frame_number, timestamp)
            
            # Add info overlay
            info_y = 30
            cv2.putText(frame, f"Frame: {frame_number}/{total_frames}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Signal: {self.signal_state}", 
                       (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 0, 255) if self.signal_state == "RED" else (0, 255, 0), 2)
            
            cv2.putText(frame, f"Vehicles: {vehicle_count}", 
                       (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Violations: {self.violation_count}", 
                       (10, info_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Write frame
            out.write(frame)
            
            # Display
            if show_display:
                cv2.imshow('Red Light Violation Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress
            if frame_number % 30 == 0:
                print(f"Progress: {frame_number}/{total_frames} frames | Violations: {self.violation_count}")
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Generate report
        self.generate_challan_report()
        
        print(f"\n✓ Processing complete!")
        print(f"✓ Output video: {output_video}")


# ============ USAGE ============

if __name__ == "__main__":
    
    # Initialize detector with stop line config
    detector = RedLightViolationDetector(
        stop_line_config='stop_line_config.json',
        signal_config='signal_timing.json'  # Optional
    )
    
    # Process video
    detector.process_video(
        video_path='traffic_video.mp4',
        output_video='violation_detection_output.mp4',
        show_display=True
    )