import cv2  # 用于图片处理的库
import numpy as np  # 用于数值计算的库
import mediapipe as mp  # 用于手部关键点检测的库
import pandas as pd  # 用于数据处理和存储的库
import os  # 用于文件和文件夹操作的库

# 初始化 MediaPipe Hands 模型
mpHands = mp.solutions.hands
# 创建一个手部检测对象，设置为处理静态图片模式，最多检测一只手，检测置信度阈值为 0.7
hands = mpHands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils  # 用于绘制手部关键点和连接线等可视化操作

# 设置主文件夹路径，该文件夹包含多个子文件夹，每个子文件夹代表一个手势类别
main_folder = "change_dataest"

# 初始化列表，用于存储提取到的手部关键点坐标数据和对应的手势标签
landmark_data = []
gesture_labels = []

# 遍历主文件夹中的每个子文件夹（每个子文件夹代表一个手势类别）
for label in os.listdir(main_folder):
    label_folder = os.path.join(main_folder, label)  # 构建当前类别文件夹的完整路径
    if not os.path.isdir(label_folder):
        continue  # 如果不是文件夹，则跳过

    # 获取当前类别文件夹中的所有图片文件路径
    image_paths = []
    for filename in os.listdir(label_folder):
        # 判断文件是否为图片文件（支持常见的图片格式）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_paths.append(os.path.join(label_folder, filename))  # 添加图片文件的完整路径到列表中

    # 处理当前类别文件夹中的每张图片
    for image_path in image_paths:
        image = cv2.imread(image_path)  # 读取图片文件
        if image is None:
            print(f"无法读取图片: {image_path}")  # 如果图片读取失败，打印提示信息并跳过
            continue

        # 将图片从 BGR 格式转换为 RGB 格式，因为 MediaPipe 的模型要求输入图片为 RGB 格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 使用 MediaPipe Hands 模型处理图片，获取手部关键点检测结果
        results = hands.process(image_rgb)

        # 如果检测到手部关键点
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 提取每个关键点的 x、y、z 坐标，并将其转换为 NumPy 数组
                landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
                landmarks = landmarks.flatten()  # 将数组展平为一维数组，方便后续存储和处理
                landmark_data.append(landmarks)  # 将提取到的关键点坐标数据添加到列表中

                # 使用文件夹名称作为手势标签，并将其添加到标签列表中
                gesture_labels.append(label)

# 将提取到的关键点数据和标签整合为一个 DataFrame
data = pd.DataFrame(landmark_data)
data['label'] = gesture_labels  # 添加标签列

# 将数据保存到 CSV 文件中，方便后续用于机器学习模型的训练等任务
data.to_csv('change_gesture_data3.csv', index=False)

print("数据已保存到 change_gesture_data3.csv")  # 打印数据保存完成的提示信息