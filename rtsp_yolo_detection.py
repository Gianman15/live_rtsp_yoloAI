#!/usr/bin/env python3
"""
Real-time object detection using YOLOv8 on RTSP stream
"""
import cv2
import argparse
from ultralytics import YOLO


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YOLOv8 Real-time RTSP Stream Object Detection')
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
    
    # Setup video writer if saving is enabled
    out = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_detection.mp4', fourcc, fps, (width, height))
        print("Saving output to: output_detection.mp4")
    
    frame_count = 0
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
            
            # Run YOLOv8 inference
            results = model.predict(
                source=frame,
                conf=args.conf,
                device=args.device,
                verbose=False,
                stream=False
            )
            
            # Get annotated frame with bounding boxes and labels
            annotated_frame = results[0].plot()
            
            # Display detection information
            detections = results[0].boxes
            if len(detections) > 0:
                print(f"Frame {frame_count}: Detected {len(detections)} objects", end='\r')
            
            # Save frame if enabled
            if args.save and out is not None:
                out.write(annotated_frame)
            
            # Show frame if enabled
            if args.show:
                cv2.imshow('YOLOv8 RTSP Detection', annotated_frame)
                
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
