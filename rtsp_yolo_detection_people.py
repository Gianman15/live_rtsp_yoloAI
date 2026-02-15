#!/usr/bin/env python3
"""
Real-time person detection using YOLOv8 on RTSP stream with ROI boundary support and saving paired snapshots (annotated and clean) 
when people are detected in order to help build a fine tuning dataset for better detection of people in IR footage.
This script is similar to rtsp_yolo_detection.py but with added functionality to save snapshots of detected events, which can be useful for creating a custom dataset for training or fine-tuning a model to improve performance on specific types of footage, 
such as IR videos where detection can be more challenging.
"""
import cv2
import argparse
import numpy as np
import os
import time
from datetime import datetime
from ultralytics import YOLO


def is_in_roi(box_coords, roi):
    """
    Check if the center of a bounding box is within the ROI.
    box_coords: [x1, y1, x2, y2]
    roi: [x1, y1, x2, y2] or None
    """
    if roi is None:
        return True
    
    # Calculate center of bounding box
    center_x = (box_coords[0] + box_coords[2]) / 2
    center_y = (box_coords[1] + box_coords[3]) / 2
    
    # Check if center is within ROI
    return (roi[0] <= center_x <= roi[2] and 
            roi[1] <= center_y <= roi[3])


def draw_roi(frame, roi, color=(0, 255, 0), thickness=2):
    """Draw ROI rectangle on frame"""
    if roi is not None:
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), color, thickness)
        cv2.putText(frame, "ROI", (roi[0] + 5, roi[1] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YOLOv8 Person Detection on RTSP Stream with ROI Support')
    parser.add_argument('--rtsp', type=str, required=True, help='RTSP stream URL')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                        help='YOLOv8 model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)')
    parser.add_argument('--conf', type=float, default=0.25, 
                        help='Confidence threshold for detection (0.0-1.0)')
    parser.add_argument('--device', type=str, default='0', 
                        help='Device to run inference on (0 for GPU, cpu for CPU)')
    parser.add_argument('--show', action='store_true', default=True,
                        help='Display the detection results')
    parser.add_argument('--save', action='store_true',
                        help='Save the detection results to video file')
    parser.add_argument('--roi', type=str, default=None,
                        help='Region of Interest as "x1,y1,x2,y2" in pixels or "x1p,y1p,x2p,y2p" as percentages (e.g., "0.2,0.2,0.8,0.8" for 20%%-80%% of frame)')
    parser.add_argument('--snapshots', action='store_true',
                        help='Save snapshots every second when people are detected')
    parser.add_argument('--snapshot-dir', type=str, default='data',
                        help='Directory to save snapshots (default: data)')
    args = parser.parse_args()

    # Load YOLOv8 model
    print(f"Loading YOLOv8 model: {args.model}")
    model = YOLO(args.model)
    
    # Open RTSP stream
    print(f"Connecting to RTSP stream: {args.rtsp}")
    cap = cv2.VideoCapture(args.rtsp)
    
    if not cap.isOpened():
        print("Error: Could not open RTSP stream")
        return
    
    # Get stream properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Stream properties - FPS: {fps}, Resolution: {width}x{height}")
    
    # Parse ROI if provided
    roi = None
    if args.roi:
        coords = [float(x) for x in args.roi.split(',')]
        if len(coords) != 4:
            print("Error: ROI must be 4 values (x1,y1,x2,y2)")
            return
        
        # If values are between 0 and 1, treat as percentages
        if all(0 <= c <= 1 for c in coords):
            roi = [
                int(coords[0] * width),
                int(coords[1] * height),
                int(coords[2] * width),
                int(coords[3] * height)
            ]
            print(f"ROI (percentage): {coords} -> pixels: {roi}")
        else:
            roi = [int(c) for c in coords]
            print(f"ROI (pixels): {roi}")
    else:
        print("No ROI specified - detecting people in entire frame")
    
    print("Detecting: PEOPLE ONLY (class 0)")
    
    # Setup snapshot directory if enabled
    snapshot_dir = None
    if args.snapshots:
        snapshot_dir = args.snapshot_dir
        os.makedirs(snapshot_dir, exist_ok=True)
        print(f"Snapshots enabled - saving to: {snapshot_dir}/")
    
    # Setup video writer if saving is enabled
    out = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_detection.mp4', fourcc, fps, (width, height))
        print("Saving output to: output_detection.mp4")
    
    frame_count = 0
    last_snapshot_time = 0
    snapshot_count = 0
    print("\nStarting real-time detection...")
    print("Press 'q' to quit\n")
    
    try:
        while True:
            # Read frame from stream
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame from stream")
                break
            
            frame_count += 1
            
            # Run YOLOv8 inference - detect only people (class 0)
            results = model.predict(
                source=frame,
                conf=args.conf,
                device=args.device,
                classes=[0],  # 0 = person in COCO dataset
                verbose=False,
                stream=False
            )
            
            # Get annotated frame with bounding boxes and labels
            annotated_frame = results[0].plot()
            
            # Draw ROI boundary on frame
            annotated_frame = draw_roi(annotated_frame, roi)
            
            # Filter detections by ROI if specified
            detections = results[0].boxes
            people_in_roi = 0
            
            if len(detections) > 0:
                if roi is not None:
                    # Count only people within ROI
                    for box in detections:
                        box_coords = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                        if is_in_roi(box_coords, roi):
                            people_in_roi += 1
                    print(f"Frame {frame_count}: {people_in_roi}/{len(detections)} people in ROI | Snapshots: {snapshot_count}", end='\r')
                else:
                    people_in_roi = len(detections)
                    print(f"Frame {frame_count}: Detected {people_in_roi} people | Snapshots: {snapshot_count}", end='\r')
            
            # Save snapshot pair if enabled and people detected (once per second)
            if snapshot_dir and people_in_roi > 0:
                current_time = time.time()
                if current_time - last_snapshot_time >= 1.0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # Save annotated image (with bounding boxes and predictions)
                    annotated_filename = os.path.join(snapshot_dir, f"{timestamp}_annotated.jpg")
                    cv2.imwrite(annotated_filename, annotated_frame)
                    # Save clean image (original frame without annotations)
                    clean_filename = os.path.join(snapshot_dir, f"{timestamp}_clean.jpg")
                    cv2.imwrite(clean_filename, frame)
                    snapshot_count += 1
                    last_snapshot_time = current_time
            
            # Save frame if enabled
            if args.save and out is not None:
                out.write(annotated_frame)
            
            # Show frame if enabled
            if args.show:
                cv2.imshow('YOLOv8 Person Detection - RTSP', annotated_frame)
                
                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopping detection...")
                    break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        print("Releasing resources...")
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()
