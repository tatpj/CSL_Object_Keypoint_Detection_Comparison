import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import csv

# 初始化MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# 加载之前训练好的手势识别模型
model = load_model('gesture_model_cnn.keras')

# 手势识别函数
def recognize_gesture(landmarks):
    normalized_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
    normalized_landmarks = normalized_landmarks.flatten()
    input_data = np.array([normalized_landmarks])
    prediction = model.predict(input_data)
    return np.argmax(prediction)

# 加载图片并预处理
def load_and_process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# 图片手势识别
def predict_gesture_from_image(image):
    results = hands.process(image)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            return recognize_gesture(hand_landmarks)
    return None

# 定义手势类别与文本的映射关系
gesture_map = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "You",
    11: "Me",
    12: "Good",
    13: "Bad",
    14: "Very",
    15: "de",
    16: "Call",
    17: "Think",
    18: "Notice"
}

# 主验证函数
def validate_model_on_dataset(dataset_folder, output_csv, output_img_folder):
    correct_count = 0  # 正确预测的计数
    incorrect_count = 0  # 错误预测的计数
    unknown_count = 0  # 未识别的计数
    total_count = 0  # 总样本计数

    # 统计每种真实标签的预测结果数量
    true_labels_counts = defaultdict(int)
    predicted_labels_counts = defaultdict(int)

    # 创建CSV文件并写入表头
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['image_path', 'true_label', 'predicted_label', 'is_correct']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 遍历测试文件夹中的每个子文件夹
        for label in os.listdir(dataset_folder):
            label_folder = os.path.join(dataset_folder, label)
            if not os.path.isdir(label_folder):
                continue

            # 遍历子文件夹中的每个图片
            for image_name in os.listdir(label_folder):
                total_count += 1  # 每处理一张图片，总样本数加1
                image_path = os.path.join(label_folder, image_name)
                if not os.path.isfile(image_path):
                    continue

                # 创建输出图片文件夹
                output_label_folder = os.path.join(output_img_folder, label)
                if not os.path.exists(output_label_folder):
                    os.makedirs(output_label_folder)

                # 读取图片
                frame = cv2.imread(image_path)
                if frame is None:
                    print(f"无法读取图片: {image_path}")
                    continue

                # 进行手势识别
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                predicted_label = None

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        gesture_id = recognize_gesture(hand_landmarks)
                        predicted_label = gesture_map.get(gesture_id, "unknown")
                if predicted_label is None:
                    predicted_label = "unknown"
                    is_correct = False
                    unknown_count += 1
                else:
                    is_correct = (predicted_label == label)

                # 累加正确预测、错误预测的计数
                if is_correct:
                    correct_count += 1
                else:
                    incorrect_count += 1

                # 累计真实标签和预测标签的计数
                true_labels_counts[label] += 1
                predicted_labels_counts[predicted_label] += 1

                # 写入CSV文件
                writer.writerow({
                    'image_path': image_path,
                    'true_label': label,
                    'predicted_label': predicted_label,
                    'is_correct': is_correct
                })

                # 打印结果
                print(f"图片: {image_path}, 真实标签: {label}, 预测标签: {predicted_label}, 正确: {is_correct}")

        # 计算准确率
        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"模型准确率: {accuracy:.2%}")

        # 打印各类统计信息
        print(f"总样本数: {total_count}")
        print(f"正确预测数: {correct_count}")
        print(f"错误预测数: {incorrect_count}")
        print(f"未识别数: {unknown_count}")

        print("\n按真实标签统计:")
        for label, count in true_labels_counts.items():
            print(f"真实标签 '{label}' 的样本数: {count}")

        print("\n按预测标签统计:")
        for label, count in predicted_labels_counts.items():
            print(f"预测标签 '{label}' 的样本数: {count}")

    return accuracy, total_count, correct_count, incorrect_count, unknown_count

if __name__ == '__main__':
    # 测试文件夹路径
    dataset_folder = r"F:\pythonProject\pytorch project\zuanti\csl_3\test2"  # 替换为你的测试数据集文件夹路径

    # 输出CSV文件路径
    output_csv = 'validation_results.csv'

    # 输出图片文件夹路径
    output_img_folder = r'F:\pythonProject\pytorch project\zuanti\csl_3\output_images'  # 替换为你的输出文件夹路径

    # 评估模型并获取相关统计信息
    accuracy, total_count, correct_count, incorrect_count, unknown_count = validate_model_on_dataset(
        dataset_folder, output_csv, output_img_folder
    )

    print(f"评估结果已保存到: {output_csv}")
    print(f"带有预测结果的图片已保存到: {output_img_folder}")
    print(f"模型准确率: {accuracy:.2%}")
    print(f"总样本数: {total_count}")
    print(f"正确预测数: {correct_count}")
    print(f"错误预测数: {incorrect_count}")
    print(f"未识别数: {unknown_count}")