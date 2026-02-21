#视频处理，图像识别
import cv2
import numpy as np
import torchvision
from torchvision import transforms
import time
import torch
from torch import nn

#------------------------搭建模型并识别图片---------------------------
def organizate(img):
     to_pil = transforms.ToPILImage()
     img = to_pil(img) #numpy->PIL
     transform = torchvision.transforms.Compose([torchvision.transforms.Resize((64, 48)),torchvision.transforms.ToTensor()])
     image = transform(img)

     class Discern(nn.Module):  # 还原完整模型，模型结构
         def __init__(self):
             super().__init__()
             self.model = nn.Sequential(
                 nn.Conv2d(3, 32, 3, padding=1),
                 nn.BatchNorm2d(32),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(2),

                 nn.Conv2d(32, 64, 3, padding=1),
                 nn.BatchNorm2d(64),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(2),

                 nn.Conv2d(64, 128, 3, padding=1),
                 nn.BatchNorm2d(128),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(2),

                 nn.Flatten(),
                 nn.Linear(128 * 8 * 6, 128),
                 nn.ReLU(inplace=True),
                 nn.Dropout(0.5),
                 nn.Linear(128, 4)
             )
         def forward(self, x):
             x = self.model(x)
             return x

     fs = Discern()  # 实例化
     model = torch.load("model_comp_color.pth", "cpu")  # 加载预训练权重，模型权重
     fs.load_state_dict(model)  # 将权重填充到神经网络中，使结构完整

     image = torch.reshape(image, (1, 3, 64, 48))
     fs.eval()  # 切换为测试模式
     with torch.no_grad():  # 关闭梯度计算，节省内存和计算资源
         output = fs(image)
     return output.argmax(dim=1)

#----------------------计算包围所有轮廓的外接矩形------------------
def get_bounding_box_for_all_contours(contours):
    if not contours:
        return 0, 0, 0, 0

    # 收集所有轮廓的所有点
    all_points = []
    for cnt in contours:
        points = cnt.reshape(-1, 2)  # 形状变为 (n, 2)，每行是(x,y)
        all_points.extend(points)

    # 转换为numpy数组方便计算
    all_points = np.array(all_points)

    # 计算极值
    min_x = np.min(all_points[:, 0])
    min_y = np.min(all_points[:, 1])
    max_x = np.max(all_points[:, 0])
    max_y = np.max(all_points[:, 1])

    return min_x, min_y, max_x, max_y


#-----------------读取视频，框选目标，传入识别------------------
def video_contour_detection():
    bag=["coca","juice","lays","spring"]
    cap = cv2.VideoCapture(0)  # 调用电脑默认摄像头（编号0）

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("错误：无法打开视频或摄像头！")
        return

    # 设置轮廓面积阈值,小于100则过滤
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频读取完毕或无法获取帧画面")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #转灰度图
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  #高斯模糊去噪
        edges = cv2.Canny(blurred, 30, 100)   #边缘检测

        contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #获取轮廓
        x,y,w,h = get_bounding_box_for_all_contours(contours)  #调用函数，取得角点


        cv2.rectangle(frame, (x,y),(w,h), (0, 255, 0), 2)
        frame_get=frame[y:h,x:w]
        # 7. 显示处理后的画面
        cv2.imshow("Video Contour Detection (Filtered)", frame)
        cv2.imshow("Canny Edges", edges)

        # 2. 创建480×640的空白背景
        bg_x=640
        bg_y=480
        background = np.zeros((bg_y, bg_x, 3), dtype=np.uint8) #高 宽 通道数. x*y=640*480
        background[:] = (1,1,1)
        frame_h, frame_w = frame_get.shape[:2]  # 截取图片的长宽 h->高,w->宽
        x_offset = (bg_x - frame_w) // 2  # 背景边长-图片边长/2=居中放置的坐标
        y_offset = (bg_y - frame_h) // 2
        background[y_offset: y_offset + frame_h, x_offset: x_offset + frame_w] = frame_get
        key = cv2.waitKey(1) & 0xFF #防卡死
        end = time.time() - start
        if end > 5:   #5秒后保存图片，传入识别函数
            out=organizate(background)
            print("object:",bag[out])   #此输出为结果
            break


    # 8. 释放资源，关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()

# 调用函数
if __name__ == "__main__":
    video_contour_detection()
