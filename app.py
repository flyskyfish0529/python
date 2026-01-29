import os
import cv2
import time
import shutil
import torch
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- 配置 ---
UPLOAD_FOLDER = 'uploads'
PENDING_FOLDER = 'pending_faces'  # 临时存放陌生人照片
KNOWN_FOLDER = 'known_faces'  # 正式人脸库
LINGER_THRESHOLD =4.0  # 陌生人停留多少秒触发报警
whitelist_logs = []       # 存储记录：[{'time': '12:00:01', 'name': 'Tom'}, ...]
whitelist_cooldowns = {}  # 存储冷却时间：{'Tom': 1712345678.9}
LOG_COOLDOWN = 60         # 冷却时间（秒）：同一个人 60秒内 不重复记录
os.environ['TORCH_HOME'] = './weights'

# 确保文件夹存在
for folder in [UPLOAD_FOLDER, PENDING_FOLDER, KNOWN_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- AI 模型初始化 ---
yolo_model = YOLO('yolov8n.pt')
resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(keep_all=False)

# --- 全局变量 ---
known_embeds = []
known_names = []
total_count = set()
video_source = 0

# 陌生人追踪逻辑变量
stranger_timers = {}  # { track_id: first_seen_timestamp }
processed_stranger_ids = set()  # 记录已经报警过的ID，防止重复截图

current_status = {
    "count": 0,
    "current_num": 0,
    "warning": False,
    "source_type": "webcam"
}


def load_face_db():
    global known_embeds, known_names
    # 清空旧数据，重新加载
    temp_embeds = []
    temp_names = []

    print("正在加载人脸库...")
    if not os.path.exists(KNOWN_FOLDER): os.makedirs(KNOWN_FOLDER)

    for file in os.listdir(KNOWN_FOLDER):
        if file.endswith(('.jpg', '.png')):
            try:
                img = Image.open(f'{KNOWN_FOLDER}/{file}')
                # 重新计算一次 embedding 确保准确
                img_cropped = mtcnn(img)
                if img_cropped is not None:
                    embed = resnet(img_cropped.unsqueeze(0)).detach().numpy()
                    temp_embeds.append(embed)
                    temp_names.append(os.path.splitext(file)[0])
            except Exception as e:
                print(f"Error loading {file}: {e}")

    known_embeds = temp_embeds
    known_names = temp_names
    print(f"已加载 {len(known_names)} 个人脸数据")


# 启动时加载一次
load_face_db()


def generate_frames():
    # 引入所有需要修改的全局变量
    global video_source, total_count, stranger_timers, processed_stranger_ids
    global whitelist_logs, whitelist_cooldowns, current_status

    cap = cv2.VideoCapture(video_source)

    while True:
        success, frame = cap.read()
        if not success:
            # 如果是视频文件，播放完后循环播放
            if isinstance(video_source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        # --- 1. YOLO 追踪 (只检测人 class=0) ---
        results = yolo_model.track(frame, persist=True, classes=0, verbose=False)
        current_person_count = 0

        # 记录当前帧出现的所有 ID，用于后续清理计时器
        current_frame_ids = set()

        if results[0].boxes:
            current_person_count = len(results[0].boxes)
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else -1

                if track_id != -1:
                    total_count.add(track_id)
                    current_frame_ids.add(track_id)

                # --- 初始化默认身份 ---
                identity = "Stranger"
                color = (0, 0, 255)  # 陌生人默认红色

                # 截取人脸区域
                face_roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]

                # 只有人脸区域足够大才进行处理，节省资源并提高准确率
                if face_roi.size > 0 and (y2 - y1) > 50:
                    try:
                        pil_img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                        face_tensor = mtcnn(pil_img)  # MTCNN 检测

                        is_known = False

                        # --- 2. 熟人识别逻辑 ---
                        if face_tensor is not None:
                            curr_embed = resnet(face_tensor.unsqueeze(0)).detach().numpy()

                            # 与库中人脸比对
                            for i, embed in enumerate(known_embeds):
                                if np.linalg.norm(curr_embed - embed) < 0.75:
                                    identity = known_names[i]
                                    color = (0, 255, 0)  # 熟人绿色
                                    is_known = True

                                    # [新增] 白名单日志记录 (带冷却时间)
                                    current_time = time.time()
                                    last_log_time = whitelist_cooldowns.get(identity, 0)

                                    if current_time - last_log_time > LOG_COOLDOWN:
                                        log_time_str = time.strftime("%H:%M:%S")
                                        whitelist_logs.insert(0, {
                                            "time": log_time_str,
                                            "name": identity,
                                            "id": track_id
                                        })
                                        whitelist_cooldowns[identity] = current_time
                                        # 保持日志列表不过长
                                        if len(whitelist_logs) > 50: whitelist_logs.pop()

                                    break  # 找到人后跳出比对循环

                        # --- 3. 陌生人停留检测逻辑 ---
                        if not is_known and track_id != -1:
                            # 第一次见到该 ID，开始计时
                            if track_id not in stranger_timers:
                                stranger_timers[track_id] = time.time()

                            duration = time.time() - stranger_timers[track_id]

                            # [修改版] 只要超时且未处理过，就强制截图 (不需要 MTCNN 必须通过)
                            if duration > LINGER_THRESHOLD and track_id not in processed_stranger_ids:
                                filename = f"pending_{track_id}_{int(time.time())}.jpg"
                                save_path = os.path.join(PENDING_FOLDER, filename)

                                if face_roi.size > 0:
                                    cv2.imwrite(save_path, face_roi)
                                    processed_stranger_ids.add(track_id)
                                    print(f"[System] 陌生人 ID:{track_id} 停留 {int(duration)}s，已截图保存。")

                            # 画面显示倒计时提示
                            if duration < LINGER_THRESHOLD:
                                cv2.putText(frame, f"Wait: {int(duration)}s", (x1, y1 - 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                        # 如果之前被当作陌生人计时，现在突然认出来了，删除计时器
                        elif is_known and track_id in stranger_timers:
                            del stranger_timers[track_id]

                    except Exception as e:
                        # 忽略人脸检测过程中的小错误（如侧脸检测不到）
                        pass

                # --- 4. 绘制框和文字 ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id} {identity}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- 5. 清理离开画面的 ID 计时器 ---
        for old_id in list(stranger_timers.keys()):
            if old_id not in current_frame_ids:
                del stranger_timers[old_id]

        # --- 6. 更新全局状态供前端查询 ---
        current_status["count"] = len(total_count)
        current_status["current_num"] = current_person_count
        current_status["warning"] = current_person_count >= 2

        # 编码推流
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# --- 路由 ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    return jsonify(current_status)


# 获取待确认的陌生人列表
# --- 在 app.py 中替换这两个函数 ---

@app.route('/get_pending_faces')
def get_pending_faces():
    faces = []
    # 使用绝对路径，更加稳健
    abs_pending_folder = os.path.abspath(PENDING_FOLDER)

    if os.path.exists(abs_pending_folder):
        # 按修改时间排序，最新的在最前面
        files = sorted(
            os.listdir(abs_pending_folder),
            key=lambda x: os.path.getmtime(os.path.join(abs_pending_folder, x)),
            reverse=True
        )
        for file in files:
            if file.endswith(('.jpg', '.png')):
                faces.append(file)

    # 打印日志到控制台，确认后端被前端呼叫了
    if len(faces) > 0:
        print(f"[Debug] 前端正在获取列表，发现 {len(faces)} 张图片: {faces}")

    return jsonify(faces)


@app.route('/pending_img/<filename>')
def pending_img(filename):
    from flask import send_from_directory
    # 同样使用绝对路径
    abs_pending_folder = os.path.abspath(PENDING_FOLDER)
    return send_from_directory(abs_pending_folder, filename)





# 确认陌生人入库
@app.route('/confirm_stranger', methods=['POST'])
def confirm_stranger():
    data = request.json
    filename = data.get('filename')
    name = data.get('name')

    if not filename or not name: return jsonify({'success': False})

    src = os.path.join(PENDING_FOLDER, filename)
    # 新文件名：Name.jpg
    dst = os.path.join(KNOWN_FOLDER, f"{name}.jpg")

    if os.path.exists(src):
        shutil.move(src, dst)  # 移动并重命名
        load_face_db()  # 重新加载模型内存库
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'File not found'})


# 删除/忽略陌生人
@app.route('/delete_stranger', methods=['POST'])
def delete_stranger():
    data = request.json
    filename = data.get('filename')
    path = os.path.join(PENDING_FOLDER, filename)
    if os.path.exists(path):
        os.remove(path)
        return jsonify({'success': True})
    return jsonify({'success': False})


# 其他原有路由...
@app.route('/upload_video', methods=['POST'])
def upload_video():
    global video_source, total_count, current_status, stranger_timers, processed_stranger_ids
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        video_source = filepath
        # 重置所有计数器
        total_count = set()
        stranger_timers = {}
        processed_stranger_ids = set()
        current_status["source_type"] = "video"
        return jsonify({"success": True, "filename": filename})


@app.route('/reset_camera', methods=['POST'])
def reset_camera():
    global video_source, total_count, stranger_timers, processed_stranger_ids
    video_source = 0
    total_count = set()
    stranger_timers = {}
    processed_stranger_ids = set()
    return jsonify({"success": True})
@app.route('/get_access_logs')
def get_access_logs():
    return jsonify(whitelist_logs)

@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    global whitelist_logs
    whitelist_logs = []
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)