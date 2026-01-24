import cv2
import numpy as np

class TiltedYellowSquareRecognizer:
    def __init__(self):
        # 初始化摄像头（设置缓冲区大小，降低延迟）
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        # 摄像头配置：降低分辨率以提高帧率，减少延迟
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 缓冲区大小设为1，减少延迟
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # 强制设置帧率
        
        # HSV颜色范围配置（优化阈值，提高识别灵敏度）
        self.yellow_lower = np.array([18, 90, 90])
        self.yellow_upper = np.array([32, 255, 255])
        
        # 内部物体颜色识别的HSV范围（优化饱和度阈值，适应不同光照）
        self.object_color_ranges = {
            'Red': ([0, 100, 70], [10, 255, 255]),
            'Red2': ([170, 100, 70], [180, 255, 255]),
            'Green': ([35, 100, 70], [77, 255, 255]),
            'Blue': ([100, 100, 70], [130, 255, 255]),
            'Orange': ([11, 100, 70], [20, 255, 255]),
            'Purple': ([131, 100, 70], [169, 255, 255]),
            'Cyan': ([78, 100, 70], [99, 255, 255])
        }
        
        # 颜色绘制用的BGR值（物体轮廓颜色+文字颜色）
        self.color_bgr = {
            'Red': (0, 0, 255),
            'Green': (0, 255, 0),
            'Blue': (255, 0, 0),
            'Orange': (0, 165, 255),
            'Purple': (128, 0, 128),
            'Cyan': (255, 255, 0),
            'Yellow': (0, 255, 255),  # 文字颜色
            'Black': (0, 0, 0)
        }
        
        # 形状名称英文映射
        self.shape_mapping = {
            "三角形": "Triangle",
            "正方形": "Square",
            "矩形": "Rectangle",
            "圆形": "Circle",
            "未知": "Unknown"
        }

    def find_yellow_square(self, frame):
        """快速查找任意倾斜角度的黄色正方形，返回轮廓、掩码和区域信息"""
        # 简化预处理：降低模糊核大小，提高速度
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # 提取黄色区域掩码
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        
        # 简化形态学操作：减少迭代次数
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((5, 5), np.uint8)
        yellow_mask = cv2.erode(yellow_mask, kernel_small, iterations=1)
        yellow_mask = cv2.dilate(yellow_mask, kernel_small, iterations=1)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel_large)
        
        # 查找轮廓（只保留外部轮廓，减少计算）
        contours, _ = cv2.findContours(yellow_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        yellow_square_contour = None
        square_mask = None  # 黄色正方形的掩码（用于后续内部物体提取）
        square_center = None
        square_size = None
        
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < 800:  # 降低最小面积阈值，支持更小的正方形
                continue
            
            # 轮廓近似（降低近似精度，提高速度）
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.05 * peri, True)  # 0.05比0.04更快
            if len(approx) != 4:
                continue
            
            # 旋转矩形判断
            rect = cv2.minAreaRect(contour)
            (center_x, center_y), (w, h), angle = rect
            aspect_ratio = min(w, h) / max(w, h)
            rect_area = w * h
            area_ratio = contour_area / rect_area
            
            # 放宽判断条件，提高识别速度
            if aspect_ratio >= 0.85 and area_ratio >= 0.8:
                yellow_square_contour = approx
                square_center = (int(center_x), int(center_y))
                square_size = int(max(w, h))
                
                # 创建正方形的掩码（用于后续提取内部物体）
                square_mask = np.zeros_like(yellow_mask)
                cv2.drawContours(square_mask, [approx], -1, 255, -1)
                
                break
        
        return yellow_square_contour, square_mask, square_center, square_size

    def detect_inner_objects(self, frame, square_mask):
        """实时检测黄色正方形内部的物体，返回轮廓、颜色和形状"""
        if square_mask is None:
            return [], [], []
        
        # 直接提取正方形内部区域，无需旋转校正（提高速度）
        inner_mask = cv2.bitwise_and(square_mask, square_mask)
        # 过滤黄色背景，只保留非黄色物体
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        non_yellow_mask = cv2.bitwise_not(cv2.inRange(hsv, self.yellow_lower, self.yellow_upper))
        combined_mask = cv2.bitwise_and(inner_mask, non_yellow_mask)
        
        # 快速形态学优化
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        
        # 查找内部物体轮廓
        inner_contours, _ = cv2.findContours(combined_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        colors = []
        shapes = []
        
        for contour in inner_contours:
            if cv2.contourArea(contour) < 150:  # 降低小轮廓阈值，支持更小的物体
                continue
            
            # 快速识别形状和颜色
            shape = self.recognize_shape(contour)
            color = self.recognize_color(frame, contour, square_mask)
            
            if shape != "Unknown" and color != "Unknown":
                valid_contours.append(contour)
                colors.append(color)
                shapes.append(shape)
        
        return valid_contours, colors, shapes

    def recognize_shape(self, contour):
        """快速识别形状"""
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)  # 降低精度，提高速度
        area = cv2.contourArea(contour)
        
        if len(approx) == 3:
            return self.shape_mapping["三角形"]
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.8 <= aspect_ratio <= 1.2:
                return self.shape_mapping["正方形"]
            else:
                return self.shape_mapping["矩形"]
        elif len(approx) > 6:  # 降低边数阈值，更快识别圆形
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * radius * radius
            if 0.7 * circle_area <= area <= 1.3 * circle_area:
                return self.shape_mapping["圆形"]
        
        return self.shape_mapping["未知"]

    def recognize_color(self, frame, contour, square_mask):
        """快速识别颜色：直接在原始帧中计算，无需ROI裁剪"""
        # 创建物体的掩码（结合正方形掩码，确保只计算内部区域）
        obj_mask = np.zeros_like(square_mask)
        cv2.drawContours(obj_mask, [contour], -1, 255, -1)
        obj_mask = cv2.bitwise_and(obj_mask, square_mask)
        
        # 计算HSV均值（只计算物体区域）
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mean_hsv = cv2.mean(hsv, mask=obj_mask)[:3]
        h, s, v = mean_hsv
        
        if s < 50:  # 降低饱和度阈值，适应浅色物体
            return "Unknown"
        
        # 快速颜色匹配
        if (0 <= h <= 10) or (170 <= h <= 180):
            return "Red"
        elif 11 <= h <= 20:
            return "Orange"
        elif 21 <= h <= 32:
            return "Yellow"  # 理论上内部不会有黄色，防止误识别
        elif 33 <= h <= 77:
            return "Green"
        elif 78 <= h <= 99:
            return "Cyan"
        elif 100 <= h <= 130:
            return "Blue"
        elif 131 <= h <= 169:
            return "Purple"
        
        return "Unknown"

    def draw_annotations(self, frame, yellow_contour, inner_contours, inner_colors, inner_shapes, square_center, square_size):
        """实时绘制所有标注，确保流畅无延迟"""
        # 1. 绘制黄色正方形轮廓（绿色框）
        if yellow_contour is not None:
            cv2.drawContours(frame, [yellow_contour.astype(np.int32)], -1, (0, 255, 0), 2)
        
        # 2. 绘制内部物体轮廓和中心点（实时标注）
        for i, contour in enumerate(inner_contours):
            color = inner_colors[i]
            shape = inner_shapes[i]
            
            # 绘制物体轮廓（对应颜色）
            contour_color = self.color_bgr.get(color, self.color_bgr['Black'])
            cv2.drawContours(frame, [contour], -1, contour_color, 2)
            
            # 绘制中心点（增强直观性）
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cX, cY), 3, self.color_bgr['Black'], -1)
        
        # 3. 绘制识别结果文字（黄色英文，正方形上方）
        if square_center is not None and square_size is not None and len(inner_colors) > 0:
            center_x, center_y = square_center
            text_y = max(20, center_y - square_size//2 - 10)
            
            # 构建文字（多个物体用逗号分隔，更简洁）
            text_parts = [f"{color} {shape}" for color, shape in zip(inner_colors, inner_shapes)]
            annotation_text = ", ".join(text_parts)
            
            # 自适应文字大小
            font_scale = min(0.6, square_size / 250)
            thickness = 2
            
            # 水平居中
            text_size = cv2.getTextSize(annotation_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = center_x - text_size[0] // 2
            
            # 快速绘制文字（减少边框厚度，提高速度）
            cv2.putText(frame, annotation_text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
            cv2.putText(frame, annotation_text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.color_bgr['Yellow'], thickness)
        
        # 4. 绘制提示信息（简化文字，提高速度）
        cv2.putText(frame, "Press 'q' to quit | Real-time detection", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

    def run(self):
        """运行实时识别程序"""
        print("Real-time detection started. Press 'q' to quit.")
        print("Supports tilted yellow squares and inner objects.")
        
        while True:
            # 读取摄像头帧（启用非阻塞读取，降低延迟）
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame. Exiting...")
                break
            
            # 1. 查找黄色正方形（每帧快速检测）
            yellow_contour, square_mask, square_center, square_size = self.find_yellow_square(frame)
            
            # 2. 实时检测内部物体（每帧重新识别）
            inner_contours, inner_colors, inner_shapes = self.detect_inner_objects(frame, square_mask)
            
            # 3. 实时绘制标注
            result_frame = self.draw_annotations(frame, yellow_contour, inner_contours, inner_colors, inner_shapes, square_center, square_size)
            
            # 显示结果（使用CV_WINDOW_NORMAL，避免窗口大小问题）
            cv2.imshow("Real-time Tilted Yellow Square Recognition", result_frame)
            
            # 快速退出判断（减少等待时间）
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        recognizer = TiltedYellowSquareRecognizer()
        recognizer.run()
    except Exception as e:
        print(f"Program Error: {e}")
        cv2.destroyAllWindows()
