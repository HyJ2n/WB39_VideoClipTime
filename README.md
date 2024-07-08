#=========================[Video_Print.py]=========================
여러 비디오들의 실제로 촬영된 시작 시간을 입력하면 객체가 탐지된 부분만 Clip을 따고 Clip들의 시간 순서대로 나열되는 코드 


target_class = 'red' : 빨간색 옷을 입은 특정 객체를 탐지하기 위해선 옷 색상 인식 모델(color_model.pt)필요 

클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\a_0.mp4, 시작 시간: 2024-07-04 10:00:00.099699
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\a_1.mp4, 시작 시간: 2024-07-04 10:00:00.332329
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\a_2.mp4, 시작 시간: 2024-07-04 10:00:08.740260
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\a_3.mp4, 시작 시간: 2024-07-04 10:00:09.139056
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\a_4.mp4, 시작 시간: 2024-07-04 10:00:14.256927
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\a_5.mp4, 시작 시간: 2024-07-04 10:00:24.592368
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\c_0.mp4, 시작 시간: 2024-07-04 10:00:30.099699
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\c_1.mp4, 시작 시간: 2024-07-04 10:00:30.332329
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\a_6.mp4, 시작 시간: 2024-07-04 10:00:32.269175
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\a_7.mp4, 시작 시간: 2024-07-04 10:00:33.930821
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\a_8.mp4, 시작 시간: 2024-07-04 10:00:34.196685
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\a_9.mp4, 시작 시간: 2024-07-04 10:00:36.589456
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\c_2.mp4, 시작 시간: 2024-07-04 10:00:38.740260
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\c_3.mp4, 시작 시간: 2024-07-04 10:00:39.139056
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\c_4.mp4, 시작 시간: 2024-07-04 10:00:44.256927
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\c_5.mp4, 시작 시간: 2024-07-04 10:00:54.592368
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\b_0.mp4, 시작 시간: 2024-07-04 10:01:00.099699
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\b_1.mp4, 시작 시간: 2024-07-04 10:01:00.332329
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\c_6.mp4, 시작 시간: 2024-07-04 10:01:02.269175
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\c_7.mp4, 시작 시간: 2024-07-04 10:01:03.930821
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\c_8.mp4, 시작 시간: 2024-07-04 10:01:04.196685
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\c_9.mp4, 시작 시간: 2024-07-04 10:01:06.589456
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\b_2.mp4, 시작 시간: 2024-07-04 10:01:08.740260
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\b_3.mp4, 시작 시간: 2024-07-04 10:01:09.139056
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\b_4.mp4, 시작 시간: 2024-07-04 10:01:14.256927
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\b_5.mp4, 시작 시간: 2024-07-04 10:01:24.592368
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\b_6.mp4, 시작 시간: 2024-07-04 10:01:32.269175
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\b_7.mp4, 시작 시간: 2024-07-04 10:01:33.930821
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\b_8.mp4, 시작 시간: 2024-07-04 10:01:34.196685
클립 파일: C:\Users\bit\VSCode\taeseung\output_clips\b_9.mp4, 시작 시간: 2024-07-04 10:01:36.589456

비디오 파일 순서:
a_0 ▶ a_1 ▶ a_2 ▶ a_3 ▶ a_4 ▶ a_5 ▶ c_0 ▶ c_1 ▶ a_6 ▶ a_7 ▶ a_8 ▶ a_9 ▶ c_2 ▶ c_3 
▶ c_4 ▶ c_5 ▶ b_0 ▶ b_1 ▶ c_6 ▶ c_7 ▶ c_8 ▶ c_9 ▶ b_2 ▶ b_3 ▶ b_4 ▶ b_5 ▶ b_6 ▶ b_7 ▶ b_8 ▶ b_9

비디오 파일 경로:
a ▶ c ▶ a ▶ c ▶ b ▶ c ▶ b

#=========================[Map.py]=========================
미완성 : 맵을 로딩한 후 CCTV 위치를 찍고 해당 가로 , 세로 , 단위를 입력 받아 선택한 CCTV 간격의 거리를 구하는 코드 
