import cv2
from ultralytics import YOLO
import os
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, Response
import base64

app = Flask(__name__)

# 모델 로드
model_path = "./runs/detect/train/weights/best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Error: Model file not found at {model_path}")
model = YOLO(model_path)

# 클래스 이름, 클립 저장 경로 설정
target_class = 'longsleevetop'  # 탐지할 대상 클래스 이름
output_clips_dir = './output'  # 클립 저장 경로
os.makedirs(output_clips_dir, exist_ok=True)

# 클립 시간 정보를 저장할 리스트
clip_times_list = []

# 비디오 처리 함수 정의
def process_video(file_path, start_time, clip_times):
    cap = cv2.VideoCapture(file_path)  # 비디오 파일 열기
    if not cap.isOpened():
        return {"status": "Error", "message": f"Could not open video {file_path}."}

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    combined_writer = None
    clip_start_frame = None
    clip_start_time = None
    clip_index = 0
    file_label = os.path.basename(file_path).split(".")[0]

    while cap.isOpened():
        ret, frame = cap.read()  # 프레임 읽기
        if not ret:
            break

        try:
            results = model.predict(frame, save=False, imgsz=640, conf=0.7, device="cuda")  # YOLO 모델로 프레임 예측
        except Exception as e:
            return {"status": "Error", "message": f"Error during YOLO prediction: {e}"}

        class_found = False
        for r in results:
            boxes = r.boxes.xyxy  # 바운딩 박스 좌표
            cls = r.boxes.cls  # 클래스 번호
            conf = r.boxes.conf  # 신뢰도
            cls_dict = r.names  # 클래스 이름 딕셔너리

            for box, cls_number, conf in zip(boxes, cls, conf):
                cls_number_int = int(cls_number.item())
                cls_name = cls_dict[cls_number_int]

                if cls_name == target_class:  # 대상 클래스가 발견되면
                    class_found = True
                    x1, y1, x2, y2 = box
                    x1_int = int(x1.item())
                    y1_int = int(y1.item())
                    x2_int = int(x2.item())
                    y2_int = int(y2.item())

                    frame = cv2.rectangle(frame, (x1_int, y1_int), (x2_int, y2_int), (0, 255, 0), 2)  # 프레임에 박스 그리기
                    conf_number = float(conf.item())
                    label = f"{cls_name} {conf_number:.2f}"
                    frame = cv2.putText(frame, label, (x1_int, y1_int - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 라벨 추가

        if class_found:
            if combined_writer is None:
                clip_filename = f'{file_label}_{clip_index}.mp4'
                combined_clip_path = os.path.join(output_clips_dir, clip_filename)
                combined_writer = cv2.VideoWriter(
                    combined_clip_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    frame_rate,
                    (frame_width, frame_height)
                )

            if clip_start_frame is None:
                clip_start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
                clip_start_time = start_time + timedelta(seconds=(clip_start_frame / frame_rate))
                clip_times.append({
                    "clip_file": combined_clip_path,
                    "start_time": clip_start_time,
                    "file_label": f"{file_label}_{clip_index}",
                    "video_label": file_label
                })

            combined_writer.write(frame)
        else:
            if combined_writer is not None:
                combined_writer.release()
                combined_writer = None
                clip_start_frame = None
                clip_index += 1

    if combined_writer is not None:
        combined_writer.release()

    cap.release()
    cv2.destroyAllWindows()

    return {"status": "Success", "message": "Video processed successfully."}

# 클립 시간 정보를 파일에 저장하는 함수
def save_clip_times_to_file():
    latest_clip_times = clip_times_list[-1] if clip_times_list else []
    latest_clip_times.sort(key=lambda x: x["start_time"])
    video_order_path = []
    for clip in latest_clip_times:
        if not video_order_path or video_order_path[-1] != clip["video_label"]:
            video_order_path.append(clip["video_label"])

    with open(os.path.join(output_clips_dir, 'clips_times.txt'), 'w') as f:
        for clip_info in latest_clip_times:
            f.write(f"클립 파일: {clip_info['clip_file']}, 시작 시간: {clip_info['start_time']}\n")
        
        f.write("\n비디오 파일 순서:\n")
        video_order_text = " ▶ ".join([clip_info['file_label'] for clip_info in latest_clip_times])
        f.write(video_order_text)
        
        f.write("\n\n이동 경로:\n")
        f.write(" ▶ ".join(video_order_path))

# 비디오 순서를 읽어오는 함수
def load_clip_times_from_file():
    if not os.path.exists(os.path.join(output_clips_dir, 'clips_times.txt')):
        return [], []
    
    clip_times = []
    video_order = []

    with open(os.path.join(output_clips_dir, 'clips_times.txt'), 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('클립 파일:'):
            parts = line.split(', ')
            clip_file = parts[0].split(': ')[1]
            start_time = parts[1].split(': ')[1]
            clip_times.append({"clip_file": clip_file, "start_time": start_time})
        elif '비디오 파일 순서' in line:
            video_order = line.strip().split(' ▶ ')

    return clip_times, video_order

@app.route('/upload_batch', methods=['POST'])
def upload_batch_videos():
    data = request.get_json()
    
    if not data or 'videos' not in data:
        return jsonify({"status": "Error", "message": "Missing video data."}), 400

    clip_times = []
    results = []
    for video_data in data['videos']:
        file_name = video_data.get('file_name')
        file_content_base64 = video_data.get('file_content')
        start_time_str = video_data.get('start_time')

        if not file_name or not file_content_base64 or not start_time_str:
            results.append({"file_name": file_name, "status": "Error", "message": "Missing data in video."})
            continue

        try:
            start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
        except ValueError as e:
            results.append({"file_name": file_name, "status": "Error", "message": f"Invalid start time format: {e}"})
            continue

        file_content = base64.b64decode(file_content_base64)
        file_path = os.path.join(output_clips_dir, file_name)
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        result = process_video(file_path, start_time, clip_times)
        results.append({"file_name": file_name, "status": result["status"], "message": result["message"]})

    clip_times_list.append(clip_times)
    save_clip_times_to_file()

    return jsonify(results)

@app.route('/get_clip_videos', methods=['GET'])
def get_clip_videos():
    clip_times, _ = load_clip_times_from_file()
    clips = []
    for clip in clip_times:
        with open(clip["clip_file"], "rb") as video_file:
            encoded_string = base64.b64encode(video_file.read()).decode('utf-8')
            clips.append({"file_name": os.path.basename(clip["clip_file"]), "file_content": encoded_string, "start_time": clip["start_time"]})
    return jsonify({"status": "Success", "clips": clips})

if __name__ == '__main__':
    # 클립 시간 정보를 파일에서 읽기
    clip_times, video_order = load_clip_times_from_file()

    app.run(debug=True)
