import torch
import cv2
import os
import csv
from collections import defaultdict
from ultralytics import YOLO

# 定义自定义标签信息
CUSTOM_LABELS = [
    {"name": "0", "color": "#7d050a"},
    {"name": "1", "color": "#d12345"},
    {"name": "2", "color": "#350dea"},
    {"name": "3", "color": "#479ffe"},
    {"name": "4", "color": "#4a649f"},
    {"name": "5", "color": "#478144"},
    {"name": "6", "color": "#57236b"},
    {"name": "7", "color": "#1cdda5"},
    {"name": "8", "color": "#e2bc6e"},
    {"name": "9", "color": "#f067db"},
    {"name": "You", "color": "#a966b0"},
    {"name": "Me", "color": "#08d3e6"},
    {"name": "Good", "color": "#a7d696"},
    {"name": "Bad", "color": "#70faa5"},
    {"name": "Very", "color": "#bf3a91"},
    {"name": "de", "color": "#9ef06c"},
    {"name": "Call", "color": "#ca2f79"},
    {"name": "Think", "color": "#a497c2"},
    {"name": "Notice", "color": "#4a0401"},
]

# DETECTOR_PATH = "runs/detect/train8/weights/best.pt"
DETECTOR_PATH = "runs/detect/train9/weights/best.pt"

# 将标签信息转换为易于访问的字典
LABELS_DICT = {item["name"]: item["color"] for item in CUSTOM_LABELS}


# 将十六进制颜色转换为BGR格式
def hex_to_bgr(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i + 2], 16) for i in (4, 2, 0))  # 返回BGR格式


# 创建标签到颜色的映射
LABEL_COLORS = {label: hex_to_bgr(color) for label, color in LABELS_DICT.items()}


class baseDetector(object):
    def __init__(self):
        self.img_size = 640
        self.conf = 0.25
        self.iou = 0.70

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")


class yoloDetector(baseDetector):
    def __init__(self):
        super(yoloDetector, self).__init__()
        self.init_model()

    def init_model(self):
        self.weights = DETECTOR_PATH  # 使用YOLOv8的预训练模型
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(self.weights)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def detect(self, im):
        results = self.model(im, imgsz=self.img_size, conf=self.conf, iou=self.iou, device=self.device)
        detected_boxes = results[0].boxes
        pred_boxes = []
        for box in detected_boxes:
            class_id = box.cls.int().cpu().item()
            lbl = self.names[class_id]
            xyxy = box.xyxy.cpu()
            x1, y1, x2, y2 = xyxy[0].numpy()
            confidence = box.conf.cpu().item()
            pred_boxes.append((x1, y1, x2, y2, lbl, confidence))
        return pred_boxes

    def draw_bboxes(self, im, pred_boxes):
        for box in pred_boxes:
            x1, y1, x2, y2, lbl, _ = box
            # 使用自定义标签颜色
            if lbl in LABEL_COLORS:
                color = LABEL_COLORS[lbl]
            else:
                color = (0, 0, 0)  # 默认颜色

            thickness = 1
            # 绘制边界框
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

            # 添加文本标签
            text = f'{lbl}'

            # 计算颜色亮度以确定文字颜色
            luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
            text_color = (255, 255, 255) if luminance < 128 else (0, 0, 0)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

            # 半透明文字背景框位置
            padding = 5
            text_x = int(x1)
            text_y = int(y1) - 5
            box_start = (text_x, text_y - text_size[1] - 2 * padding)
            box_end = (text_x + text_size[0] + 2 * padding, text_y)

            # 绘制半透明背景
            overlay = im.copy()
            cv2.rectangle(overlay, box_start, box_end, color, -1)
            alpha = 0.6  # 背景透明度
            cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0, im)

            # 绘制文字
            text_pos = (text_x + padding, text_y - padding)
            cv2.putText(im, text, text_pos,
                        font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)
        return im


def evaluate_model(detector, test_folder, output_csv, output_img_folder):
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
        for label in os.listdir(test_folder):
            label_folder = os.path.join(test_folder, label)
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

                output_image_path = os.path.join(output_label_folder, image_name)

                # 读取图片
                frame = cv2.imread(image_path)
                if frame is None:
                    print(f"无法读取图片: {image_path}")
                    continue

                # 进行目标检测
                pred_boxes = detector.detect(frame)

                # 绘制预测框
                frame_with_boxes = detector.draw_bboxes(frame.copy(), pred_boxes)

                # 保存带有预测框的图片
                cv2.imwrite(output_image_path, frame_with_boxes)

                # 获取预测结果
                # 这里假设每个图片中只有一个主要目标，取置信度最高的预测结果
                predicted_label = None
                max_confidence = -1
                for box in pred_boxes:
                    _, _, _, _, lbl, confidence = box
                    if confidence > max_confidence:
                        max_confidence = confidence
                        predicted_label = lbl

                # 如果没有检测到任何目标，记录为错误
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
    # 初始化YOLO检测器
    myDetector = yoloDetector()

    # 测试文件夹路径
    # test_folder = r'F:\pythonProject\pytorch project\YOLOv8\test\test'  # 替换为您的测试文件夹路径
    test_folder = r'F:\pythonProject\pytorch project\zuanti\csl_3\test2'  # 替换为您的测试文件夹路径

    # 输出CSV文件路径
    output_csv = 'evaluation_results_100.csv'

    # 输出图片文件夹路径
    output_img_folder = r'F:\pythonProject\pytorch project\YOLOv8\test\output_images'  # 替换为您的输出文件夹路径
    # 评估模型并获取相关统计信息
    accuracy, total_count, correct_count, incorrect_count, unknown_count = evaluate_model(
        myDetector, test_folder, output_csv, output_img_folder
    )

    print(f"评估结果已保存到: {output_csv}")
    print(f"带有预测框的图片已保存到: {output_img_folder}")
    print(f"模型准确率: {accuracy:.2%}")
    print(f"总样本数: {total_count}")
    print(f"正确预测数: {correct_count}")
    print(f"错误预测数: {incorrect_count}")
    print(f"未识别数: {unknown_count}")