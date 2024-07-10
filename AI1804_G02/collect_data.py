import cv2
import os

def extract_frames(video_paths, output_dir, num_frames=100):
    for idx, video_path in enumerate(video_paths, start=1):
        # Tạo thư mục con trong thư mục data
        video_folder = os.path.join(output_dir, f"{idx}")
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)

        # Đối tượng capture video từ file
        cap = cv2.VideoCapture(video_path)
        
        # Kiểm tra xem video có mở thành công hay không
        if not cap.isOpened():
            print(f"Không thể mở video: {video_path}")
            continue

        frame_count = 0
        while frame_count < num_frames:
            # Đọc từng khung hình
            ret, frame = cap.read()

            # Kiểm tra nếu không còn khung hình nào để đọc
            if not ret:
                break

            # Lưu khung hình thành file ảnh
            frame_filename = os.path.join(video_folder, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_filename, frame)

            frame_count += 1

        # Giải phóng tài nguyên
        cap.release()

# Đường dẫn đến các video đầu vào và thư mục đầu ra
video_paths = [
    r'D:\SourceCode\ASS_CPV301\video_ppt\bao.mp4',
    r'D:\SourceCode\ASS_CPV301\video_ppt\keo.mp4',
    r'D:\SourceCode\ASS_CPV301\video_ppt\bua.mp4'
]
output_dir = 'data'  # Thư mục để lưu các thư mục con của từng video

# Số lượng frame muốn lấy cho mỗi video
num_frames_per_video = 100

# Gọi hàm để tách video và lưu các frame vào các thư mục con
extract_frames(video_paths, output_dir, num_frames=num_frames_per_video)
