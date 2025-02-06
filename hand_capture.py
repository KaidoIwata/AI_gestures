import cv2
import mediapipe as mp
import time
import json
import pyttsx3
import numpy as np

# ======== 音声エンジンの初期化 ========
engine = pyttsx3.init()

mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

collected_data = []
collecting = False

last_print_time = 0
PRINT_INTERVAL = 4.0

def is_vertical_line(hand_data, tip_indices,
                     x_tolerance=0.05,
                     y_min_range=0.15):
    """
    指定した指先(tip_indices)が「縦一列」に見えるかどうかを判定する関数。
    - x_tolerance: 横幅がこの値以下なら「狭い」とみなす
    - y_min_range: 縦幅がこの値以上なら「高い」とみなす
    
    返り値: True / False
    """
    # tip_indices に対応する x, y 座標を抽出
    x_coords = [hand_data[i]["x"] for i in tip_indices]
    y_coords = [hand_data[i]["y"] for i in tip_indices]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # 横幅が小さい (width < x_tolerance) ＋ 縦幅が大きい (height > y_min_range)
    if width < x_tolerance and height > y_min_range:
        return True
    else:
        return False
      
def is_others_folded(hand_data, except_indices=[8,12], fold_threshold=0.9):
    """
    except_indices 以外の指先が「ある程度下に折れている (y が大きい)」とみなす例。
    fold_threshold はざっくり「画面の下側にいる」基準。
    - たとえば y=0 が上側、 y=1 が下側（Mediapipeの標準正規化）
    """
    for i in range(21):
        if i in except_indices:
            continue
        # 例: 指先の y 座標が 0.9 より大きい(=下側にある)なら曲げているとみなす
        # （値はカメラ距離によって要調整）
        if hand_data[i]["y"] < fold_threshold:
            # fold_threshold より上にいたら「まだ伸びている」かも
            return False
    return True

def judge_gesture(hand_data):
    """
    ジェスチャー(ありがとう、こんにちは)を判定して文字列を返す。
    """
    # 4本指 → 「ありがとう」
    four_fingers = [8, 12, 16, 20]  # 人差し指・中指・薬指・小指の先端
    if is_vertical_line(hand_data, four_fingers, x_tolerance=0.05, y_min_range=0.06):
        return "ありがとう"
    
    # 2本指 → 「こんにちは」
    two_fingers = [8, 12]  # 人差し指・中指の先端
    if is_vertical_line(hand_data, two_fingers, x_tolerance=0.06, y_min_range=0.15):
        # 2本の場合は y_min_range を少し下げるなど、調整してもよい
        return "こんにちは"
    
    return None

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("カメラフレームが取得できませんでした。")
            continue
        
        # RGBに変換
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 手の検出
        results = hands.process(image)
        
        # BGRに戻す
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.append({
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z
                    })
                
                # データ収集モードがONなら収集
                if collecting:
                    collected_data.append({
                        "hand_index": hand_index,
                        "landmarks": hand_data,
                        "timestamp": time.time()
                    })
                
                # ---- ジェスチャーを判定 ----
                gesture_name = judge_gesture(hand_data)
                if gesture_name is not None:
                    # 2秒おきに出力＆音声化
                    current_time = time.time()
                    if current_time - last_print_time > PRINT_INTERVAL:
                        last_print_time = current_time
                        print("判定:", gesture_name)
                        engine.say(gesture_name)
                        engine.runAndWait()
        
        # 画面に映像を表示
        cv2.imshow('Hand Landmark Collector', cv2.flip(image, 1))
        
        key = cv2.waitKey(5)
        if key & 0xFF == ord('c'):  # 'c'キーで収集モードON/OFF
            collecting = not collecting
            print("データ収集モード: ", "ON" if collecting else "OFF")
        elif key & 0xFF == ord('s'):  # 's'キーでデータ保存
            with open('hand_data.json', 'w') as f:
                json.dump(collected_data, f, indent=4)
            print("データを保存しました: hand_data.json")
        elif key & 0xFF == 27:  # ESCキーで終了
            break

cap.release()
cv2.destroyAllWindows()
