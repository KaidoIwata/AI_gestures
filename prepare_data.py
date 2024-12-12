import json
import numpy as np
from sklearn.model_selection import train_test_split

# JSONデータを読み込む
def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    X = []  # 入力データ (ランドマーク座標)
    y = []  # ラベル ("ありがとう" -> 1, その他 -> 0)

    for item in data:
        landmarks = item['landmarks']
        # x, y, zの順で平坦化された座標配列に変換
        landmarks_flat = [coord for lm in landmarks for coord in (lm['x'], lm['y'], lm['z'])]
        X.append(landmarks_flat)

        # TODO: ラベルを設定 (仮: "ありがとう" -> 1, その他 -> 0)
        # 例: itemに"gesture"フィールドがある場合
        if 'gesture' in item and item['gesture'] == "ありがとう":
            y.append(1)
        else:
            y.append(0)

    return np.array(X), np.array(y)

# データの分割
def prepare_dataset(json_path, test_size=0.2):
    X, y = load_data(json_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# 実行例
json_path = 'hand_data.json'  # JSONデータファイルのパス
X_train, X_test, y_train, y_test = prepare_dataset(json_path)
print(f"訓練データ数: {len(X_train)}, テストデータ数: {len(X_test)}")
