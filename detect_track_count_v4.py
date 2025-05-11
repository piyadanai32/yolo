import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import torch
import platform
import os

def detect_gpu():
    """Detect and select the best available GPU"""
    gpu_info = {
        'device': 'cpu',
        'name': 'CPU',
        'backend': None
    }
    
    try:
        # Check NVIDIA GPU
        if torch.cuda.is_available():
            gpu_info['device'] = 'cuda'
            gpu_info['name'] = torch.cuda.get_device_name(0)
            gpu_info['backend'] = 'cuda'
            print(f"NVIDIA GPU detected: {gpu_info['name']}")
            return gpu_info
            
        # Check AMD GPU on Windows
        elif platform.system() == 'Windows':
            try:
                import win32com.client
                wmi = win32com.client.GetObject("winmgmts:")
                gpu_list = wmi.InstancesOf("Win32_VideoController")
                for gpu in gpu_list:
                    if "AMD" in gpu.Name or "Radeon" in gpu.Name:
                        gpu_info['device'] = 'cpu'  # AMD uses CPU backend with ROCm
                        gpu_info['name'] = gpu.Name
                        gpu_info['backend'] = 'rocm'
                        print(f"AMD GPU detected: {gpu_info['name']}")
                        return gpu_info
            except:
                pass
                
        # Check Intel GPU
        if torch.backends.mps.is_available():
            gpu_info['device'] = 'mps'
            gpu_info['name'] = 'Intel Graphics'
            gpu_info['backend'] = 'mps'
            print(f"Intel GPU detected: {gpu_info['name']}")
            return gpu_info
            
    except Exception as e:
        print(f"Error detecting GPU: {e}")
    
    print("No dedicated GPU detected, using CPU")
    return gpu_info

# Detect available GPU
gpu_info = detect_gpu()

# Configure YOLO based on available GPU
model = YOLO('yolov8n.pt')

# Configure GPU settings
try:
    if gpu_info['backend'] == 'cuda':
        # NVIDIA GPU settings
        model.to('cuda')
        torch.backends.cudnn.benchmark = True
        print("Using NVIDIA GPU acceleration")
    elif gpu_info['backend'] == 'rocm':
        # AMD GPU settings
        os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
        os.environ['HIP_VISIBLE_DEVICES'] = '0'
        print("Using AMD GPU acceleration")
    elif gpu_info['backend'] == 'mps':
        # Intel GPU settings
        model.to('mps')
        model.model.half()
        print("Using Intel GPU acceleration")
    else:
        # CPU settings
        model.to('cpu')
        torch.set_num_threads(4)
        print("Using CPU acceleration")
except Exception as e:
    print(f"Error configuring GPU: {e}")
    print("Falling back to CPU")
    model.to('cpu')
    torch.set_num_threads(4)

# Enable OpenCV optimization
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

class_list = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
tracker = Tracker()

# Video capture with resolution based on GPU capability
cap = cv2.VideoCapture('2025-02-18 11-02-09.mkv')

# Set resolution based on GPU capability
if gpu_info['backend'] in ['cuda', 'rocm']:
    # Higher resolution for dedicated GPUs
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
else:
    # Lower resolution for integrated GPU/CPU
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize counters
down = {}
up = {}
counter_down = []
counter_up = []

def get_bottom_right_quadrant(frame):
    height, width = frame.shape[:2]
    start_x = width // 2
    start_y = height // 2
    return frame[start_y:height, start_x:width]

# Create window
cv2.namedWindow("Bottom Right Quadrant", cv2.WINDOW_NORMAL)

# FPS counter
frame_count = 0
start_time = cv2.getTickCount()

# GPU-specific batch size
batch_size = 4 if gpu_info['backend'] in ['cuda', 'rocm'] else 1

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Calculate FPS
    if frame_count % 30 == 0:
        current_time = cv2.getTickCount()
        elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
    
    # Extract bottom right quadrant
    quadrant = get_bottom_right_quadrant(frame)
    
    # Resize based on GPU capability
    if gpu_info['backend'] in ['cuda', 'rocm']:
        processed_frame = cv2.resize(quadrant, (600, 450), interpolation=cv2.INTER_AREA)
    else:
        processed_frame = cv2.resize(quadrant, (400, 300), interpolation=cv2.INTER_AREA)
    
    try:
        # GPU-specific inference
        if gpu_info['backend'] == 'cuda':
            with torch.amp.autocast("cuda"):
                results = model.predict(processed_frame, workers=4)
        else:
            results = model.predict(processed_frame, workers=2)
        
        result = results[0]
    except RuntimeError as e:
        print(f"Inference error: {e}")
        continue
        
    if len(result.boxes) == 0:
        cv2.imshow("Bottom Right Quadrant", processed_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue
        
    # Convert detections to DataFrame
    a = result.boxes.data
    if gpu_info['backend'] == 'cuda':
        px = pd.DataFrame(a.cpu().numpy()).astype("float")
    else:
        px = pd.DataFrame(a.numpy()).astype("float")
    
    list = []
    for index, row in px.iterrows():
        if int(row[5]) < len(class_list) and class_list[int(row[5])] == 'car':
            list.append([int(row[0]), int(row[1]), int(row[2]), int(row[3])])

    bbox_id = tracker.update(list)
    
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        red_line_y = 198 if gpu_info['backend'] not in ['cuda', 'rocm'] else 297
        blue_line_y = 268 if gpu_info['backend'] not in ['cuda', 'rocm'] else 402
        offset = 7

        # Counting logic
        if abs(cy - red_line_y) < offset:
            down[id] = cy   
            if id in up:
                counter_down.append(id)
                cv2.circle(processed_frame, (cx, cy), 4, (0, 0, 255), -1)
                
        if abs(cy - blue_line_y) < offset:
            up[id] = cy   
            if id in down:
                counter_up.append(id)
                cv2.circle(processed_frame, (cx, cy), 4, (0, 0, 255), -1)

    # Draw visualization
    cv2.line(processed_frame, (0, red_line_y), (processed_frame.shape[1], red_line_y), (0, 0, 255), 2)
    cv2.line(processed_frame, (0, blue_line_y), (processed_frame.shape[1], blue_line_y), (255, 0, 0), 2)
    
    # Add GPU info to display
    cv2.putText(processed_frame, f'GPU: {gpu_info["name"]}', (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(processed_frame, f'Down: {len(counter_down)}', (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(processed_frame, f'Up: {len(counter_up)}', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Bottom Right Quadrant", processed_frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()