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

# Only track cars, motorcycles, and buses as requested
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']
target_classes = ['car', 'motorcycle', 'bus']
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

# Initialize counters for each vehicle type
vehicle_counts = {
    'car': {'down': [], 'up': []},
    'motorcycle': {'down': [], 'up': []},
    'bus': {'down': [], 'up': []}
}

# Vehicle position tracking
vehicle_positions = {
    'down': {},  # Tracks vehicles going down
    'up': {}     # Tracks vehicles going up
}

# Store vehicle types by ID
vehicle_types = {}  # {id: class_name}

# เพิ่ม dictionary สำหรับเก็บตำแหน่ง cx ก่อนหน้า
vehicle_last_cx = {}

# เพิ่ม dictionary สำหรับเก็บ state ของแต่ละ id ว่าอยู่ฝั่งไหนของเส้น
vehicle_states = {}

# MODIFIED: Updated cropping function to get the bottom-right corner with custom dimensions
def get_custom_crop(frame):
    height, width = frame.shape[:2]
    
    # Calculate the starting point for the red area (half of width and height)
    red_start_x = width // 2
    red_start_y = height // 2
    
    # Calculate the dimensions of the orange area (smaller region within red area)
    # Adjust these percentages to match your orange box size
    orange_width_ratio = 0.7  # Percentage of red area width
    orange_height_ratio = 0.6  # Percentage of red area height
    
    # Calculate orange area dimensions
    orange_width = int((width - red_start_x) * orange_width_ratio)
    orange_height = int((height - red_start_y) * orange_height_ratio)
    
    # Calculate starting points for orange area (positioned in bottom right)
    orange_start_x = width - orange_width
    orange_start_y = height - orange_height
    
    # Return the cropped region (orange area)
    return frame[orange_start_y:height, orange_start_x:width]

# Create window
cv2.namedWindow("Vehicle Counter", cv2.WINDOW_NORMAL)

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
    
    # MODIFIED: Extract custom crop region (orange area)
    quadrant = get_custom_crop(frame)
    
    # Resize based on GPU capability
    if gpu_info['backend'] in ['cuda', 'rocm']:
        processed_frame = cv2.resize(quadrant, (600, 450), interpolation=cv2.INTER_AREA)
    else:
        processed_frame = cv2.resize(quadrant, (400, 300), interpolation=cv2.INTER_AREA)
    
    frame_height = processed_frame.shape[0]
    frame_width = processed_frame.shape[1]

    red_line_x = int(frame_width * 0.85)  
    blue_line_x = int(frame_width * 0.30)
    
    # คำนวณจุดเริ่มต้นและจุดสิ้นสุดสำหรับเส้น
    line_start_y_red = int(frame_height * 0.11)  
    line_end_y_red = int(frame_height * 0.54)    
    line_start_y_blue = int(frame_height * 0.40)  
    line_end_y_blue = int(frame_height * 40)
    
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
        # Display the counts even when no vehicles are detected
        y_offset = 40
        cv2.putText(processed_frame, f'GPU: {gpu_info["name"]}', (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        for vehicle_type in target_classes:
            cv2.putText(processed_frame, f'{vehicle_type.capitalize()} Down: {len(set(vehicle_counts[vehicle_type]["down"]))}', 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
            
            cv2.putText(processed_frame, f'{vehicle_type.capitalize()} Up: {len(set(vehicle_counts[vehicle_type]["up"]))}', 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            
        # วาดเส้นที่มีความยาวน้อยลง แม้ไม่มีวัตถุถูกตรวจจับ
        cv2.line(processed_frame, (red_line_x, line_start_y_red), (red_line_x, line_end_y_red), (0, 0, 255), 2)
        cv2.line(processed_frame, (blue_line_x, line_start_y_blue), (blue_line_x, line_end_y_blue), (255, 0, 0), 2)
        
        cv2.imshow("Vehicle Counter", processed_frame)
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
        class_id = int(row[5])
        if class_id < len(class_list):
            class_name = class_list[class_id]
            if class_name in target_classes:  # Only process target classes
                box = [int(row[0]), int(row[1]), int(row[2]), int(row[3])]
                list.append(box)
                if len(list) > 0 and index < len(px):  # Save class name to use later
                    # We'll store the class name associated with this box when we get its ID
                    temp_class = class_name
    
    bbox_id = tracker.update(list)
    
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        
        # Find class for this ID
        for index, row in px.iterrows():
            box = [int(row[0]), int(row[1]), int(row[2]), int(row[3])]
            box_cx = (box[0] + box[2]) // 2
            box_cy = (box[1] + box[3]) // 2
            
            # If the center points are close, this is likely the same object
            if abs(cx - box_cx) < 10 and abs(cy - box_cy) < 10:
                class_id = int(row[5])
                if class_id < len(class_list):
                    vehicle_type = class_list[class_id]
                    if vehicle_type in target_classes:
                        vehicle_types[id] = vehicle_type
                        break
        
        # If we don't have a type for this ID yet, use the last known type
        if id not in vehicle_types:
            continue
        
        vehicle_type = vehicle_types[id]
        
        offset = 7

        # Draw vehicle type label on the bounding box
        cv2.rectangle(processed_frame, (x3, y3), (x4, y4), (255, 0, 0), 2)
        cv2.putText(processed_frame, f'{vehicle_type}', (x3, y3-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # --- Begin: Robust counting logic with state ---
        # ตรวจสอบสถานะการข้ามเส้นแดง (ออก)
        prev_state_red = vehicle_states.get((id, 'red'), None)
        if cx < red_line_x:
            vehicle_states[(id, 'red')] = 'left'
        else:
            vehicle_states[(id, 'red')] = 'right'

        # ถ้าเคยอยู่ซ้ายแล้วข้ามไปขวา (ข้ามเส้นแดง)
        if prev_state_red == 'left' and vehicle_states[(id, 'red')] == 'right':
            if id not in vehicle_counts[vehicle_type]['down']:
                vehicle_counts[vehicle_type]['down'].append(id)
                cv2.circle(processed_frame, (cx, cy), 4, (0, 0, 255), -1)

        # ตรวจสอบสถานะการข้ามเส้นน้ำเงิน (เข้า)
        prev_state_blue = vehicle_states.get((id, 'blue'), None)
        if cx > blue_line_x:
            vehicle_states[(id, 'blue')] = 'right'
        else:
            vehicle_states[(id, 'blue')] = 'left'

        # ถ้าเคยอยู่ขวาแล้วข้ามไปซ้าย (ข้ามเส้นน้ำเงิน)
        if prev_state_blue == 'right' and vehicle_states[(id, 'blue')] == 'left':
            if id not in vehicle_counts[vehicle_type]['up']:
                vehicle_counts[vehicle_type]['up'].append(id)
                cv2.circle(processed_frame, (cx, cy), 4, (255, 0, 0), -1)
        # --- End: Robust counting logic with state ---

    # วาดเส้นที่มีความยาวน้อยลง
    cv2.line(processed_frame, (red_line_x, line_start_y_red), (red_line_x, line_end_y_red), (0, 0, 255), 2)
    cv2.line(processed_frame, (blue_line_x, line_start_y_blue), (blue_line_x, line_end_y_blue), (255, 0, 0), 2)
    
    # Add counts to display
    y_offset = 40
    cv2.putText(processed_frame, f'GPU: {gpu_info["name"]}', (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    for vehicle_type in target_classes:
        cv2.putText(processed_frame, f'{vehicle_type.capitalize()} Down: {len(set(vehicle_counts[vehicle_type]["down"]))}', 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 20
        
        cv2.putText(processed_frame, f'{vehicle_type.capitalize()} Up: {len(set(vehicle_counts[vehicle_type]["up"]))}', 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20

    cv2.imshow("Vehicle Counter", processed_frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Print final counts
print("\nFinal Vehicle Counts:")
for vehicle_type in target_classes:
    print(f"{vehicle_type.capitalize()}:")
    print(f"  Down: {len(set(vehicle_counts[vehicle_type]['down']))}")
    print(f"  Up: {len(set(vehicle_counts[vehicle_type]['up']))}")

cap.release()
cv2.destroyAllWindows()