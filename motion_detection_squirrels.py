#!/usr/bin/env python3
"""
this exists because yolov8 was trained on COCO dataset which does not include squirrels, so it cannot detect them
i wanted to have a way to pre-screen footage for squirrel events using motion detection and contour filtering based on size and aspect ratio, which can help identify squirrel-like objects in the video footage without relying on YOLOv8's object detection capabilities
this script processes video footage, applies motion detection to find moving objects, and then filters those objects based on their size and aspect ratio to identify potential squirrel events. It also supports defining a region of interest (ROI) to focus on specific areas of the video frame, and it saves annotated images of detected events for further review.
Motion-based detection for processing video footage to find squirrel-like objects
Filters by motion and object size to pre-screen footage for squirrel events
"""
import cv2
import argparse
import os
import time
import numpy as np
from datetime import datetime


def is_in_roi(center_x, center_y, roi):
    """Check if point is within ROI"""
    if roi is None:
        return True
    return (roi[0] <= center_x <= roi[2] and 
            roi[1] <= center_y <= roi[3])


def draw_roi(frame, roi, color=(0, 255, 0), thickness=2):
    """Draw ROI rectangle on frame"""
    if roi is not None:
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), color, thickness)
        cv2.putText(frame, "ROI", (roi[0] + 5, roi[1] + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def filter_contours(contours, min_area, max_area, aspect_ratio_range=(0.3, 3.0)):
    """
    Filter contours based on size and aspect ratio
    Squirrels typically have width/height ratio between 0.3-3.0 depending on pose
    """
    filtered = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0
            if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                filtered.append(contour)
    return filtered


def main():
    parser = argparse.ArgumentParser(description='Motion Detection for Squirrel-like Objects in Video Footage')
    parser.add_argument('--input', type=str, required=True, help='Input video file path')
    parser.add_argument('--output-dir', type=str, default='squirrel_data', 
                        help='Directory to save detected events (default: squirrel_data)')
    parser.add_argument('--roi', type=str, default=None,
                        help='Region of Interest as "x1,y1,x2,y2" (pixels or percentages 0-1)')
    parser.add_argument('--min-area', type=int, default=500,
                        help='Minimum contour area in pixels (default: 500 for squirrels)')
    parser.add_argument('--max-area', type=int, default=15000,
                        help='Maximum contour area in pixels (default: 15000 for squirrels)')
    parser.add_argument('--sensitivity', type=int, default=25,
                        help='Motion sensitivity threshold (lower=more sensitive, default: 25)')
    parser.add_argument('--cooldown', type=float, default=2.0,
                        help='Seconds between captures of same motion event (default: 2.0)')
    parser.add_argument('--show', action='store_true',
                        help='Display detection results in real-time')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier (1.0=normal, 2.0=2x, etc.)')
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input video file not found: {args.input}")
        return

    # Open video file
    print(f"Loading video: {args.input}")
    cap = cv2.VideoCapture(args.input)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.1f} seconds")
    
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
            print(f"  ROI (percentage): {coords} -> pixels: {roi}")
        else:
            roi = [int(c) for c in coords]
            print(f"  ROI (pixels): {roi}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving detections to: {args.output_dir}/")
    print(f"Motion sensitivity: {args.sensitivity}")
    print(f"Object size range: {args.min_area}-{args.max_area} pixels")
    print(f"Cooldown between captures: {args.cooldown}s\n")
    
    # Initialize background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=500, 
        varThreshold=args.sensitivity, 
        detectShadows=False
    )
    
    # Variables for tracking
    frame_count = 0
    detection_count = 0
    last_capture_time = 0
    processed_start = time.time()
    
    # For display
    delay = int((1000 / fps) / args.speed) if fps > 0 else 33
    
    print("Processing video... Press 'q' to quit\n")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("\nEnd of video reached")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Apply ROI mask if specified
            mask_roi = None
            if roi:
                mask_roi = np.zeros(frame.shape[:2], dtype=np.uint8)
                mask_roi[roi[1]:roi[3], roi[0]:roi[2]] = 255
            
            # Apply background subtraction
            fg_mask = back_sub.apply(frame)
            
            # Apply ROI to motion mask
            if mask_roi is not None:
                fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=mask_roi)
            
            # Remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours for squirrel-like objects
            filtered_contours = filter_contours(
                contours, 
                args.min_area, 
                args.max_area,
                aspect_ratio_range=(0.3, 3.0)
            )
            
            # Create visualization frame
            display_frame = frame.copy()
            display_frame = draw_roi(display_frame, roi)
            
            # Check if we should capture
            motion_detected = False
            if len(filtered_contours) > 0:
                motion_detected = True
                
                # Draw contours and bounding boxes
                for contour in filtered_contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x, center_y = x + w//2, y + h//2
                    
                    # Check if in ROI
                    if is_in_roi(center_x, center_y, roi):
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                        area = cv2.contourArea(contour)
                        cv2.putText(display_frame, f"Area: {int(area)}", (x, y-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Capture if cooldown period has passed
                if current_time - last_capture_time >= args.cooldown:
                    video_timestamp = frame_count / fps if fps > 0 else frame_count
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Save annotated image (with bounding boxes)
                    annotated_filename = os.path.join(args.output_dir, 
                                                     f"{timestamp}_frame{frame_count}_annotated.jpg")
                    cv2.imwrite(annotated_filename, display_frame)
                    
                    # Save clean image (original frame)
                    clean_filename = os.path.join(args.output_dir, 
                                                 f"{timestamp}_frame{frame_count}_clean.jpg")
                    cv2.imwrite(clean_filename, frame)
                    
                    detection_count += 1
                    last_capture_time = current_time
                    print(f"Captured event #{detection_count} at frame {frame_count} ({video_timestamp:.1f}s)")
            
            # Update progress display
            progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
            elapsed = time.time() - processed_start
            fps_proc = frame_count / elapsed if elapsed > 0 else 0
            
            status = f"Frame: {frame_count}/{total_frames} ({progress:.1f}%) | " \
                    f"Speed: {fps_proc:.1f} fps | Detections: {detection_count}"
            
            if motion_detected:
                status += f" | Objects: {len(filtered_contours)}"
            
            print(status, end='\r')
            
            # Display if enabled
            if args.show:
                # Show both motion mask and annotated frame
                fg_mask_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
                combined = np.hstack([display_frame, fg_mask_colored])
                
                # Resize if too large
                display_height = 480
                if combined.shape[0] > display_height:
                    scale = display_height / combined.shape[0]
                    new_width = int(combined.shape[1] * scale)
                    combined = cv2.resize(combined, (new_width, display_height))
                
                cv2.imshow('Motion Detection | Original + Motion Mask', combined)
                
                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        elapsed_total = time.time() - processed_start
        print(f"\n\nProcessing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Total detections saved: {detection_count}")
        print(f"Processing time: {elapsed_total:.1f}s")
        print(f"Average processing speed: {frame_count/elapsed_total:.1f} fps")
        print(f"Output saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
