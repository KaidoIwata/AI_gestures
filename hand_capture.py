import cv2
import mediapipe as mp
import time
import json  # データ保存用

# MediaPipeのセットアップ
mp_hands = mp.solutions.hands

# Webカメラの初期化
cap = cv2.VideoCapture(0)

# 点群データ保存用
collected_data = []
collecting = False  # データ収集モードのフラグ

# 指先のランドマークインデックス
LANDMARKS_COUNT = 21  # 手のランドマーク数

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
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 各ランドマークの座標を取得
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.append({
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z
                    })
                
                # データ収集モードが有効なら保存
                if collecting:
                    collected_data.append({
                        "hand_index": hand_index,
                        "landmarks": hand_data,
                        "timestamp": time.time()  # タイムスタンプも記録
                    })
        
        # カメラ画像の表示
        cv2.imshow('Hand Landmark Collector', cv2.flip(image, 1))
        
        # キーボード入力でモード切替
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
