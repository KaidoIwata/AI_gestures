import cv2
import mediapipe as mp
import time

# MediaPipeのセットアップ
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 指ごとの色指定 (BGR)
LANDMARK_COLORS = {
    "WRIST": (0, 0, 255),  # 手首
    "THUMB": (255, 0, 0),  # 親指
    "INDEX": (0, 255, 0),  # 人差し指
    "MIDDLE": (0, 255, 255),  # 中指
    "RING": (255, 0, 255),  # 薬指
    "PINKY": (255, 255, 0),  # 小指
}

# 指先のランドマークインデックス
FINGER_TIPS = [4, 8, 12, 16, 20]  # 親指, 人差し指, 中指, 薬指, 小指の先端
THUMB = [1, 2, 3, 4]
INDEX = [5, 6, 7, 8]
MIDDLE = [9, 10, 11, 12]
RING = [13, 14, 15, 16]
PINKY = [17, 18, 19, 20]
WRIST = [0]

# Webカメラの初期化
cap = cv2.VideoCapture(0)
last_time = time.time()  # 時間計測用

# Handsソリューションの初期化
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("カメラフレームが取得できませんでした。")
            continue
        
        # 画像をRGBに変換
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 手の検出
        results = hands.process(image)
        
        # 画像を描画用に再設定
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 検出結果から座標を取得
        if results.multi_hand_landmarks:
            hand_data = []  # 各手のデータを格納するリスト
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 各指先の座標を取得
                finger_tips_data = []
                for tip_index in FINGER_TIPS:
                    landmark = hand_landmarks.landmark[tip_index]
                    finger_tips_data.append({
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z
                    })
                hand_data.append({
                    "hand_index": hand_index,
                    "finger_tips": finger_tips_data
                })
                
                # 骨格の接続線を描画
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=2),  # 接続線の色
                )
                
                # 各ランドマークを色分けして描画
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    h, w, c = image.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)  # ピクセル座標に変換
                    
                    # 色の選択
                    if idx in WRIST:
                        color = LANDMARK_COLORS["WRIST"]
                    elif idx in THUMB:
                        color = LANDMARK_COLORS["THUMB"]
                    elif idx in INDEX:
                        color = LANDMARK_COLORS["INDEX"]
                    elif idx in MIDDLE:
                        color = LANDMARK_COLORS["MIDDLE"]
                    elif idx in RING:
                        color = LANDMARK_COLORS["RING"]
                    elif idx in PINKY:
                        color = LANDMARK_COLORS["PINKY"]
                    else:
                        color = (255, 255, 255)  # デフォルトは白
                    
                    # ランドマークの描画
                    cv2.circle(image, (cx, cy), 6, color, -1)
            
            # 5秒ごとにターミナルに出力
            current_time = time.time()
            if current_time - last_time >= 5:
                print("=== 指先座標データ ===")
                for hand in hand_data:
                    print(f"手のインデックス: {hand['hand_index']}")
                    for finger_index, finger_tip in enumerate(hand["finger_tips"]):
                        print(f"  指{finger_index + 1} (x, y, z): ({finger_tip['x']:.3f}, {finger_tip['y']:.3f}, {finger_tip['z']:.3f})")
                last_time = current_time
        
        # カメラ画像の表示
        cv2.imshow('MediaPipe Hands - Colored Landmarks', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
