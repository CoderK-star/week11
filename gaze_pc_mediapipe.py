import cv2
import numpy as np
import pyautogui
import time
from collections import deque
from pathlib import Path
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# =====================
# 初期設定
# =====================
pyautogui.FAILSAFE = False
SCREEN_W, SCREEN_H = pyautogui.size()

SMOOTHING = 0.82                  # 基本カーソル平滑化
SMOOTHING_FAST = 0.55             # 大きな移動時の平滑化
SMOOTHING_FAST_THRESHOLD = 120    # ピクセル差で高速化を判定
BLINK_EAR_TH = 0.18               # まばたき閾値
ACTION_COOLDOWN = 0.4             # クリック間隔[s]
GAZE_HISTORY_SIZE = 4             # 視線平滑化サンプル
DRIFT_THRESHOLD = 0.015           # ニュートラル再学習条件
NEUTRAL_LEARNING_RATE = 0.02
WINK_MIN_TIME = 0.04              # 片目閉じ保持秒数
BLINK_MIN_TIME = 0.12             # 両目閉じ保持秒数
WINK_EAR_MARGIN = 0.01           # ウィンク判定用の反対側開き余裕

smooth_x, smooth_y = SCREEN_W // 2, SCREEN_H // 2
last_action_time = 0.0
gaze_history = deque(maxlen=GAZE_HISTORY_SIZE)
last_gaze = None
left_wink_start = None
right_wink_start = None
blink_start = None

# 視線ニュートラル（自動）
neutral_x = None
neutral_y = None

# 視線スケール（感度）
SCALE_X = 1.8
SCALE_Y = 2.2

# =====================
# MediaPipe 初期化
# =====================
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
# Keep the downloaded model under the user's home directory to avoid
# OneDrive/Unicode path issues when the native code opens the file.
MODEL_DIR = Path.home() / ".mediapipe_models"
MODEL_PATH = MODEL_DIR / "face_landmarker.task"


def ensure_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return
    with urllib.request.urlopen(MODEL_URL) as response:
        MODEL_PATH.write_bytes(response.read())


ensure_model()

base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
face_mesh = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
)

# =====================
# ランドマーク番号
# =====================
# 左右虹彩
L_IRIS = [468, 469, 470, 471, 472]
R_IRIS = [473, 474, 475, 476, 477]

# まぶた（EAR用）
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# =====================
# 関数
# =====================
def avg_point(landmarks, idxs):
    xs = [landmarks[i].x for i in idxs]
    ys = [landmarks[i].y for i in idxs]
    return np.mean(xs), np.mean(ys)


def smooth_gaze_point(gx, gy):
    """Moving-averageフィルタで視線座標を滑らかにする"""
    gaze_history.append((gx, gy))
    xs = [p[0] for p in gaze_history]
    ys = [p[1] for p in gaze_history]
    return float(np.mean(xs)), float(np.mean(ys))

def eye_aspect_ratio(landmarks, eye):
    p = landmarks
    A = np.linalg.norm(np.array([p[eye[1]].x, p[eye[1]].y]) -
                       np.array([p[eye[5]].x, p[eye[5]].y]))
    B = np.linalg.norm(np.array([p[eye[2]].x, p[eye[2]].y]) -
                       np.array([p[eye[4]].x, p[eye[4]].y]))
    C = np.linalg.norm(np.array([p[eye[0]].x, p[eye[0]].y]) -
                       np.array([p[eye[3]].x, p[eye[3]].y]))
    return (A + B) / (2.0 * C)

# =====================
# カメラ開始
# =====================
cap = cv2.VideoCapture(0)

print("ESCキーで終了")
print("Cキー: 中央再キャリブレーション / 左右ウィンク: 左右クリック")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(time.perf_counter() * 1000)
    result = face_mesh.detect_for_video(mp_image, timestamp_ms)

    if result.face_landmarks:
        lm = result.face_landmarks[0]

        # ---- 虹彩中心（左右平均） ----
        lx, ly = avg_point(lm, L_IRIS)
        rx, ry = avg_point(lm, R_IRIS)

        gaze_x = (lx + rx) / 2
        gaze_y = (ly + ry) / 2

        gaze_x, gaze_y = smooth_gaze_point(gaze_x, gaze_y)
        last_gaze = (gaze_x, gaze_y)

        # ---- ニュートラル自動設定 ----
        if neutral_x is None:
            neutral_x = gaze_x
            neutral_y = gaze_y
        else:
            if (abs(gaze_x - neutral_x) < DRIFT_THRESHOLD and
                    abs(gaze_y - neutral_y) < DRIFT_THRESHOLD):
                neutral_x = ((1 - NEUTRAL_LEARNING_RATE) * neutral_x +
                             NEUTRAL_LEARNING_RATE * gaze_x)
                neutral_y = ((1 - NEUTRAL_LEARNING_RATE) * neutral_y +
                             NEUTRAL_LEARNING_RATE * gaze_y)

        # ---- 視線 → 画面座標 ----
        dx = (gaze_x - neutral_x) * SCALE_X
        dy = (gaze_y - neutral_y) * SCALE_Y

        tx = np.clip(0.5 + dx, 0, 1)
        ty = np.clip(0.5 + dy, 0, 1)

        px = int(tx * SCREEN_W)
        py = int(ty * SCREEN_H)

        cursor_delta = np.hypot(px - smooth_x, py - smooth_y)
        smoothing_factor = (
            SMOOTHING_FAST if cursor_delta > SMOOTHING_FAST_THRESHOLD else SMOOTHING
        )
        smooth_x = int(smooth_x * smoothing_factor + px * (1 - smoothing_factor))
        smooth_y = int(smooth_y * smoothing_factor + py * (1 - smoothing_factor))

        pyautogui.moveTo(smooth_x, smooth_y)

        # ---- まばたき検出 ----
        left_ear = eye_aspect_ratio(lm, LEFT_EYE)
        right_ear = eye_aspect_ratio(lm, RIGHT_EYE)

        now = time.time()
        left_closed = left_ear < BLINK_EAR_TH
        right_closed = right_ear < BLINK_EAR_TH
        left_open_enough = left_ear > (BLINK_EAR_TH + WINK_EAR_MARGIN)
        right_open_enough = right_ear > (BLINK_EAR_TH + WINK_EAR_MARGIN)

        if left_closed and right_open_enough:
            if left_wink_start is None:
                left_wink_start = now
            elif (now - left_wink_start >= WINK_MIN_TIME and
                    now - last_action_time > ACTION_COOLDOWN):
                pyautogui.click(button='left')
                last_action_time = now
                left_wink_start = None
        else:
            left_wink_start = None

        if right_closed and left_open_enough:
            if right_wink_start is None:
                right_wink_start = now
            elif (now - right_wink_start >= WINK_MIN_TIME and
                    now - last_action_time > ACTION_COOLDOWN):
                pyautogui.click(button='right')
                last_action_time = now
                right_wink_start = None
        else:
            right_wink_start = None

        if left_closed and right_closed:
            if blink_start is None:
                blink_start = now
            elif (now - blink_start >= BLINK_MIN_TIME and
                    now - last_action_time > ACTION_COOLDOWN):
                pyautogui.click()
                last_action_time = now
                blink_start = None
        else:
            blink_start = None

        # ---- デバッグ描画 ----
        cx = int(gaze_x * w)
        cy = int(gaze_y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

    cv2.imshow("MediaPipe Gaze Input (PC)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key in (ord('c'), ord('C')) and last_gaze is not None:
        neutral_x, neutral_y = last_gaze
        print("ニュートラルを再設定しました")

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
