# Example of training a model with improved CNN architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 用于在非交互式环境下保存图像
import matplotlib.pyplot as plt

# 加载数据集，替换为您的CSV文件路径
data = pd.read_csv('gesture_data.csv')
# 提取特征（所有列除了'label'列）
X = data.drop('label', axis=1)
# 提取标签（'label'列）
y = data['label']

# 将数据划分为训练集和测试集
# test_size=0.2 表示测试集占20%，random_state=42 确保结果可重复
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据重塑为CNN模型所需的三维格式（样本数，时间步，特征数）
X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
# 打印训练集和测试集的形状信息，检查是否正确
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# 定义改进后的CNN模型结构
model = Sequential()

# 添加第一个一维卷积层，64个滤波器，每个窗口大小为5，ReLU激活函数
model.add(Conv1D(64, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)))
# 添加最大池化层，窗口大小为2
model.add(MaxPooling1D(pool_size=2))
# 添加Dropout层，随机丢弃20%的连接，防止过拟合
model.add(Dropout(0.2))

# 添加第二个一维卷积层，32个滤波器，窗口大小为3，ReLU激活函数
model.add(Conv1D(32, kernel_size=3, activation='relu'))
# 添加最大池化层，窗口大小为2
model.add(MaxPooling1D(pool_size=2))
# 添加Dropout层，保持模型泛化能力
model.add(Dropout(0.2))

# 将数据展平为一维向量，为全连接层做准备
model.add(Flatten())

# 添加全连接层，64个神经元，ReLU激活函数
model.add(Dense(64, activation='relu'))
# 添加另一个全连接层，32个神经元，继续特征提取
model.add(Dense(32, activation='relu'))
# 输出层，19个神经元（对应19类手势），softmax激活函数用于多分类
model.add(Dense(19, activation='softmax'))

# 编译模型
# 使用Adam优化器，稀疏分类交叉熵损失函数（适用于整数标签）
# metrics=['accuracy'] 指定监控准确率指标
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
# epochs=500 表示训练500轮，validation_data 指定验证集
history = model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))

# 保存训练好的模型，方便后续加载使用
model.save('gesture_model_cnn.keras')

# 绘制训练过程中的准确率曲线
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
# 保存准确率图像
plt.savefig('accuracy_plot_cnn.png')
plt.close()

# 绘制训练过程中的损失曲线
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
# 保存损失图像
plt.savefig('loss_plot_cnn.png')
plt.close()