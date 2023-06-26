import torch
from PIL import Image
import numpy as np

# YOLOv5モデルのロード
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 動画ファイルの読み込み
import cv2
vid = cv2.VideoCapture('input_video.mp4')

# 出力ビデオファイルの設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # コーデックを指定
fps = vid.get(cv2.CAP_PROP_FPS)  # 入力動画と同じフレームレートを使用
frame_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # フレームのサイズを取得
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, frame_size)  # VideoWriterオブジェクトを作成

while True:
    # フレームの読み込み
    ret, frame = vid.read()

    # フレームが存在しない場合、動画の終了
    if not ret:
        break

    # PILイメージに変換
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 推論の実行
    results = model(frame)

    # 検出結果の描画
    result_frame = np.array(results.render()[0])
    
    # 描画結果をBGR形式に変換し、出力動画に書き込み
    out.write(cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))

# リソースの解放
vid.release()
out.release()
cv2.destroyAllWindows()
