# YOLOv8 RTSP Real-Time Object Detection

This project uses Ultralytics YOLOv8 to perform real-time object detection on RTSP streams.

This is just a precursor to a larger custom project I have planned.
## Quick Start

### 1. Test with Webcam (Optional)
Before using RTSP, test if everything works with your webcam:

```bash
./yolovenv/bin/python test_webcam.py
```

Press 'q' to quit the webcam view.

### 2. Run RTSP Detection

**Basic usage:**
```bash
./yolovenv/bin/python rtsp_yolo_detection.py --rtsp "rtsp://your-camera-ip:port/stream"
```

**Full example with options:**
```bash
./yolovenv/bin/python rtsp_yolo_detection.py \
  --rtsp "rtsp://admin:password@192.168.1.100:554/stream1" \
  --model yolov8n.pt \
  --conf 0.25 \
  --device 0 \
  --show \
  --save
```

### Command Line Options

- `--rtsp` (required): RTSP stream URL
  - Format: `rtsp://[username:password@]host:port/path`
  - Example: `rtsp://admin:12345@192.168.1.100:554/stream1`

- `--model` (optional): YOLOv8 model variant
  - `yolov8n.pt` - Nano (fastest, least accurate) - **default**
  - `yolov8s.pt` - Small
  - `yolov8m.pt` - Medium
  - `yolov8l.pt` - Large
  - `yolov8x.pt` - Extra Large (slowest, most accurate)

- `--conf` (optional): Confidence threshold (default: 0.25)
  - Range: 0.0 to 1.0
  - Higher = fewer false positives, might miss objects

- `--device` (optional): Processing device
  - `0` - Use GPU (CUDA) - **default**
  - `cpu` - Use CPU only

- `--show` (optional): Display real-time video window (enabled by default)

- `--save` (optional): Save output video to `output_detection.mp4`

## Examples

**Basic RTSP stream:**
```bash
./yolovenv/bin/python rtsp_yolo_detection.py --rtsp "rtsp://192.168.1.100:554/stream"
```

**High accuracy with save:**
```bash
./yolovenv/bin/python rtsp_yolo_detection.py \
  --rtsp "rtsp://camera.local/live" \
  --model yolov8l.pt \
  --conf 0.4 \
  --save
```

**CPU only (no GPU):**
```bash
./yolovenv/bin/python rtsp_yolo_detection.py \
  --rtsp "rtsp://192.168.1.100:554/stream" \
  --device cpu
```

## Controls

- Press **'q'** to quit the detection window
- **Ctrl+C** in terminal to stop the script

## Common RTSP URL Formats

- **Generic:** `rtsp://ip:port/stream`
- **With auth:** `rtsp://username:password@ip:port/stream`
- **Hikvision:** `rtsp://admin:password@ip:554/Streaming/Channels/101`
- **Dahua:** `rtsp://admin:password@ip:554/cam/realmonitor?channel=1&subtype=0`
- **Axis:** `rtsp://ip:554/axis-media/media.amp`
- **Amcrest:** `rtsp://admin:password@ip:554/cam/realmonitor?channel=1&subtype=1`

## Detected Objects

YOLOv8 can detect 80 different object classes including:
- People, vehicles (car, truck, bus, motorcycle, bicycle)
- Animals (dog, cat, bird, horse, etc.)
- Sports equipment, furniture, electronics, and more

Full list: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml

## Troubleshooting

**Stream won't connect:**
- Verify RTSP URL is correct
- Check username/password if required
- Ensure camera is accessible on network
- Test URL with VLC media player first

**Low FPS / Laggy:**
- Use smaller model (yolov8n.pt instead of yolov8l.pt)
- Reduce stream resolution if possible
- Ensure GPU is being used (`--device 0`)

**Out of memory:**
- Use smaller model (yolov8n.pt or yolov8s.pt)
- Use CPU instead of GPU (`--device cpu`)

## Performance Tips

1. **For speed:** Use `yolov8n.pt` with lower confidence (0.2-0.3)
2. **For accuracy:** Use `yolov8l.pt` or `yolov8x.pt` with higher confidence (0.4-0.5)
3. **GPU recommended:** Much faster than CPU for real-time processing
4. **Lower stream resolution** on camera side for better FPS
