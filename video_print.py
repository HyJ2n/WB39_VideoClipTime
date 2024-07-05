# 비디오 한개에 저장

import cv2
from ultralytics import YOLO
import os
from datetime import datetime, timedelta

def get_video_start_time(index):
    start_times = [
        datetime(2024, 7, 4, 10, 2, 0),
        datetime(2024, 7, 4, 10, 1, 0),
        datetime(2024, 7, 4, 10, 0, 30)
    ]
    return start_times[index]

# 모델 로드
model_path = r"C:\Users\bit\VSCode\color_model.pt"
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit()
model = YOLO(model_path)

# 입력 비디오 파일 목록과 시작 시간 설정
video_path = [
    {"file": r"C:\Users\bit\VSCode\taeseung\input_videos\a.mp4", "start_time": get_video_start_time(0)},
    {"file": r"C:\Users\bit\VSCode\taeseung\input_videos\b.mp4", "start_time": get_video_start_time(1)},
    {"file": r"C:\Users\bit\VSCode\taeseung\input_videos\c.mp4", "start_time": get_video_start_time(2)}
]

# 클래스 이름
target_class = 'red'

# 클립 저장 경로 설정
output_clips_dir = r'C:\Users\bit\VSCode\taeseung\output_clips'
os.makedirs(output_clips_dir, exist_ok=True)

# 클립 시간 정보를 저장할 리스트
clip_times = []

for vp in video_path:
    # 비디오 파일이 존재하는지 확인
    if not os.path.exists(vp["file"]):
        print(f"Error: Video file not found at {vp['file']}")
        continue

    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(vp["file"])
    if not cap.isOpened():
        print(f"Error: Could not open video {vp['file']}.")
        continue

    # 비디오 프레임 속성 가져오기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    clip_start_frame = None
    first_clip_time_recorded = False

    # 결합된 비디오 파일 생성
    combined_clip_path = os.path.join(output_clips_dir, f'combined_output_{os.path.basename(vp["file"]).split(".")[0]}.mp4')
    combined_writer = None

    # 비디오 처리
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 모델을 사용하여 예측
        try:
            results = model.predict(frame, save=False, imgsz=640, conf=0.7, device="cuda")
        except Exception as e:
            print(f"Error during YOLO prediction: {e}")
            break

        # 예측 결과 처리
        class_found = False
        for r in results:
            boxes = r.boxes.xyxy
            cls = r.boxes.cls
            conf = r.boxes.conf  
            cls_dict = r.names

            for box, cls_number, conf in zip(boxes, cls, conf):
                cls_number_int = int(cls_number.item())
                cls_name = cls_dict[cls_number_int]

                if cls_name == target_class:
                    class_found = True
                    x1, y1, x2, y2 = box
                    x1_int = int(x1.item())
                    y1_int = int(y1.item())
                    x2_int = int(x2.item())
                    y2_int = int(y2.item())

                    frame = cv2.rectangle(frame, (x1_int, y1_int), (x2_int, y2_int), (0, 255, 0), 2)
                    conf_number = float(conf.item())
                    label = f"{cls_name} {conf_number:.2f}"
                    frame = cv2.putText(frame, label, (x1_int, y1_int - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if class_found:
            if combined_writer is None:
                combined_writer = cv2.VideoWriter(
                    combined_clip_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    frame_rate,
                    (frame_width, frame_height)
                )

            if not first_clip_time_recorded:
                clip_start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
                clip_start_time = vp["start_time"] + timedelta(seconds=(clip_start_frame / frame_rate))
                clip_times.append({
                    "clip_file": combined_clip_path,
                    "start_time": clip_start_time,
                    "file_label": os.path.basename(vp["file"]).split(".")[0]  # Save label for sorting
                })
                first_clip_time_recorded = True

            combined_writer.write(frame)
        else:
            clip_start_frame = None

    if combined_writer is not None:
        combined_writer.release()

    cap.release()
    cv2.destroyAllWindows()

# 클립 시간 정보를 실제 시간 기준으로 정렬
clip_times.sort(key=lambda x: x["start_time"])

# 클립 시간 정보를 텍스트 파일로 저장
with open(os.path.join(output_clips_dir, 'clips_times.txt'), 'w') as f:
    for clip_info in clip_times:
        f.write(f"클립 파일: {clip_info['clip_file']}, 시작 시간: {clip_info['start_time']}\n")
    
    # 비디오 파일 순서 추가
    f.write("\n비디오 파일 순서:\n")
    video_order = " ▶ ".join([clip_info['file_label'] for clip_info in clip_times])
    f.write(video_order)

# 정렬된 클립 시간 정보 출력
print("클립 시간 정보 (정렬된 순서):")
for clip_info in clip_times:
    print(f"클립 파일: {clip_info['clip_file']}, 시작 시간: {clip_info['start_time']}")

print("클립의 시간 정보가 clips_times.txt 파일로 저장되었습니다.")