import customtkinter
import os
from PIL import Image, ImageTk
import cv2 as cv
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("计算机视觉大作业")
        self.geometry("700x450")

        #布局设置
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        #加载图片与主题
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images")
        self.logo_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "CustomTkinter_logo_single.png")), size = (26, 26))
        self.large_test_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "large_test_image.png")), size = (500, 150))
        self.image_icon_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "image_icon_light.png")), size = (20, 20))
        self.home_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "home_dark.png")),
                                                 dark_image=Image.open(os.path.join(image_path, "home_light.png")), size = (20, 20))
        self.chat_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "chat_dark.png")),
                                                 dark_image=Image.open(os.path.join(image_path, "chat_light.png")), size = (20, 20))
        self.add_user_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "add_user_dark.png")),
                                                     dark_image=Image.open(os.path.join(image_path, "add_user_light.png")), size=(20, 20))

        #创建导航框架
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(8, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="Computer Vision", image=self.logo_image,
                                                             compound="left", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        # home按钮设置
        self.home_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                   border_spacing=10, text="Home",
                                                   fg_color="transparent", text_color=("gray10", "gray90"),
                                                   hover_color=("gray70", "gray30"),
                                                   image=self.home_image, anchor="w", command=self.home_button_event)
        self.home_button.grid(row=1, column=0, sticky="ew")
        # 作业1按钮
        self.frame_2_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                      border_spacing=10, text="作业一",
                                                      fg_color="transparent", text_color=("gray10", "gray90"),
                                                      hover_color=("gray70", "gray30"),
                                                      image=self.add_user_image, anchor="w",
                                                      command=self.frame_2_button_event)
        self.frame_2_button.grid(row=2, column=0, sticky="ew")
        # 作业2按钮
        self.frame_3_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                      border_spacing=10, text="作业二",
                                                      fg_color="transparent", text_color=("gray10", "gray90"),
                                                      hover_color=("gray70", "gray30"),
                                                      image=self.add_user_image, anchor="w",
                                                      command=self.frame_3_button_event)
        self.frame_3_button.grid(row=3, column=0, sticky="ew")
        # 作业3按钮
        self.frame_4_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                      border_spacing=10, text="作业三",
                                                      fg_color="transparent", text_color=("gray10", "gray90"),
                                                      hover_color=("gray70", "gray30"),
                                                      image=self.add_user_image, anchor="w",
                                                      command=self.frame_4_button_event)
        self.frame_4_button.grid(row=4, column=0, sticky="ew")
        # 作业4按钮
        self.frame_5_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                      border_spacing=10, text="作业四",
                                                      fg_color="transparent", text_color=("gray10", "gray90"),
                                                      hover_color=("gray70", "gray30"),
                                                      image=self.add_user_image, anchor="w",
                                                      command=self.frame_5_button_event)
        self.frame_5_button.grid(row=5, column=0, sticky="ew")
        # 作业5按钮
        self.frame_6_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                      border_spacing=10, text="作业五",
                                                      fg_color="transparent", text_color=("gray10", "gray90"),
                                                      hover_color=("gray70", "gray30"),
                                                      image=self.add_user_image, anchor="w",
                                                      command=self.frame_6_button_event)
        self.frame_6_button.grid(row=6, column=0, sticky="ew")
        # 作业6按钮
        self.frame_7_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                      border_spacing=10, text="作业六",
                                                      fg_color="transparent", text_color=("gray10", "gray90"),
                                                      hover_color=("gray70", "gray30"),
                                                      image=self.add_user_image, anchor="w",
                                                      command=self.frame_7_button_event)
        self.frame_7_button.grid(row=7, column=0, sticky="ew")


        self.appearance_mode_menu = customtkinter.CTkOptionMenu(self.navigation_frame, values=["Light", "Dark", "System"],
                                                                command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=8, column=0, padx=20, pady=20, sticky="s")

        #创建主页面显示
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure(0, weight=1)
        self.home_frame_large_image_label = customtkinter.CTkLabel(self.home_frame, text="", image=self.large_test_image)
        self.home_frame_large_image_label.grid(row=0, column=0, padx=20, pady=10)

        large_font = customtkinter.CTkFont(family="微软雅黑", size=20, weight="bold")
        small_font = customtkinter.CTkFont(family="微软雅黑", size=15, weight="bold", underline=True)

        # 成员列表
        self.home_frame_text_1 = customtkinter.CTkLabel(self.home_frame, text="小组成员介绍", font=large_font)
        self.home_frame_text_1.grid(row=1, column=0, padx=20, pady=10)
        self.home_frame_text_2 = customtkinter.CTkLabel(self.home_frame, text="吴玉堂", font=small_font)
        self.home_frame_text_2.grid(row=2, column=0, padx=20, pady=10)
        self.home_frame_text_3 = customtkinter.CTkLabel(self.home_frame, text="周华威", font=small_font)
        self.home_frame_text_3.grid(row=3, column=0, padx=20, pady=10)
        self.home_frame_text_4 = customtkinter.CTkLabel(self.home_frame, text="王娜", font=small_font)
        self.home_frame_text_4.grid(row=4, column=0, padx=20, pady=10)
        self.home_frame_text_5 = customtkinter.CTkLabel(self.home_frame, text="聂欣宇", font=small_font)
        self.home_frame_text_5.grid(row=5, column=0, padx=20, pady=10)

        # 创建第二页面
        self.second_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.second_frame.grid_columnconfigure(0, weight=1)
        self.second_frame_text_1 = customtkinter.CTkLabel(self.second_frame, text="图像拼接", font=large_font)
        self.second_frame_text_1.grid(row=1, column=0, padx=20, pady=10)

        self.image1_path = ""
        self.image2_path = ""

        self.label1 = customtkinter.CTkLabel(self.second_frame, text="输入图片一")
        self.label1.grid(row=2, column=0, padx=20, pady=20)
        self.button1 = customtkinter.CTkButton(self.second_frame, text="选择图片一", command=self.select_image1)
        self.button1.grid(row=2, column=1, padx=20, pady=20)

        self.label2 = customtkinter.CTkLabel(self.second_frame, text="输入图片二")
        self.label2.grid(row=3, column=0, padx=20, pady=20)
        self.button2 = customtkinter.CTkButton(self.second_frame, text="选择图片二", command=self.select_image2)
        self.button2.grid(row=3, column=1, padx=20, pady=20)

        self.concat_button = customtkinter.CTkButton(self.second_frame, text="图片拼接", command=self.concat_images)
        self.concat_button.grid(row=4, column=0, columnspan=2, padx=20, pady=20)

        self.result_label = customtkinter.CTkLabel(self.second_frame, text="")
        self.result_label.grid(row=5, column=0, columnspan=2, padx=20, pady=20)

        self.image_label = customtkinter.CTkLabel(self.second_frame, text="")
        self.image_label.grid(row=6, column=0, columnspan=2, padx=20, pady=20)

        # 创建第三页面
        self.third_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.third_frame.grid_columnconfigure(0, weight=1)
        self.third_frame_text_1 = customtkinter.CTkLabel(self.third_frame, text="摄像机标定", font=large_font)
        self.third_frame_text_1.grid(row=1, column=0, padx=20, pady=10)


        # 创建第四页面
        self.fourth_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")

        # 创建第五页面
        self.fifth_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.fifth_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.fifth_frame.grid_columnconfigure(0, weight=1)
        self.fifth_frame.grid_columnconfigure(3, weight=1)
        self.fifth_frame.grid_rowconfigure(1, weight=1)
        self.fifth_frame.grid_rowconfigure(4, weight=1)
        # 页面5标题
        self.fifth_frame_text_1 = customtkinter.CTkLabel(self.fifth_frame, text="给定图片的基础，单应矩阵估计", font=large_font)
        self.fifth_frame_text_1.grid(row=1, column=0, columnspan=4, padx=20, pady=10)
        # 加载图片并调整大小
        left_image_5 = Image.open("I1.bmp")
        right_image_5 = Image.open("I2.bmp")
        max_size = (200, 150)
        left_image_5.thumbnail(max_size, Image.Resampling.LANCZOS)
        right_image_5.thumbnail(max_size, Image.Resampling.LANCZOS)
        left_photo = ImageTk.PhotoImage(left_image_5)
        right_photo = ImageTk.PhotoImage(right_image_5)
        # 图片显示
        self.image_label_left = customtkinter.CTkLabel(self.fifth_frame, image=left_photo, text="")
        self.image_label_left.image = left_photo
        self.image_label_left.grid(row=2, column=2, padx=10, pady=10)
        self.image_label_right = customtkinter.CTkLabel(self.fifth_frame, image=right_photo, text="")
        self.image_label_right.image = right_photo
        self.image_label_right.grid(row=2, column=3, padx=10, pady=10)
        # 设置特征关系抽取按钮
        self.feature_extraction_button = customtkinter.CTkButton(self.fifth_frame, text="特征关系抽取", command=self.extract_features)
        self.feature_extraction_button.grid(row=3, column=0, columnspan=4, padx=20, pady=5)
        # 结果展示图片
        self.result_image_label_5 = customtkinter.CTkLabel(self.fifth_frame, text="")
        self.result_image_label_5.grid(row=4, column=0, columnspan=4, padx=20, pady=5)
        # 矩阵抽取按钮
        self.matrix_extraction_button = customtkinter.CTkButton(self.fifth_frame, text="矩阵抽取", command=self.matrix_features)
        self.matrix_extraction_button.grid(row=5, column=0, columnspan=4, padx=20, pady=5)
        # 矩阵展示
        self.h = None
        self.f = None
        self.matrix_label_1 = customtkinter.CTkLabel(self.fifth_frame, text="")
        self.matrix_label_1.grid(row=6, column=0, columnspan=2, padx=10, pady=10)
        self.matrix_label_2 = customtkinter.CTkLabel(self.fifth_frame, text="")
        self.matrix_label_2.grid(row=6, column=2, columnspan=2, padx=10, pady=10)

        # 创建第六页面
        self.sixth_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.sixth_frame.grid_columnconfigure(0, weight=1)
        self.sixth_frame_text_1 = customtkinter.CTkLabel(self.sixth_frame, text="平行视图矫正", font=large_font)
        self.sixth_frame_text_1.grid(row=1, column=0, padx=20, pady=10)

        self.image3_path = ""
        self.image4_path = ""

        self.label3 = customtkinter.CTkLabel(self.sixth_frame, text="请输入平行视图左：")
        self.label3.grid(row=2, column=0, padx=20, pady=20)
        self.button3 = customtkinter.CTkButton(self.sixth_frame, text="选择图片一", command=self.select_image3)
        self.button3.grid(row=2, column=1, padx=20, pady=20)

        self.label4 = customtkinter.CTkLabel(self.sixth_frame, text="输入平行视图有右：")
        self.label4.grid(row=3, column=0, padx=20, pady=20)
        self.button4 = customtkinter.CTkButton(self.sixth_frame, text="选择图片二", command=self.select_image4)
        self.button4.grid(row=3, column=1, padx=20, pady=20)

        self.pingxing_button = customtkinter.CTkButton(self.sixth_frame, text="平行视图矫正",
                                                       command=self.find_fundamental_matrix)
        self.pingxing_button.grid(row=4, column=0, columnspan=2, padx=20, pady=20)

        self.result_label_px = customtkinter.CTkLabel(self.sixth_frame, text="")
        self.result_label_px.grid(row=5, column=0, columnspan=2, padx=20, pady=20)

        self.image_label_1 = customtkinter.CTkLabel(self.sixth_frame, text="")
        self.image_label_1.grid(row=6, column=0, padx=(20, 5), pady=20)
        self.image_label_2 = customtkinter.CTkLabel(self.sixth_frame, text="")
        self.image_label_2.grid(row=6, column=1, padx=(5, 20), pady=20)
        # 创建第七页面
        self.seventh_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")

        # 选择默认框架
        self.select_frame_by_name("home")

    def select_frame_by_name(self, name):

        # 设置按钮格式与按钮选择
        self.home_button.configure(fg_color=("gray75", "gray25") if name == "home" else "transparent")
        self.frame_2_button.configure(fg_color=("gray75", "gray25") if name == "frame_2" else "transparent")
        self.frame_3_button.configure(fg_color=("gray75", "gray25") if name == "frame_3" else "transparent")
        self.frame_4_button.configure(fg_color=("gray75", "gray25") if name == "frame_4" else "transparent")
        self.frame_5_button.configure(fg_color=("gray75", "gray25") if name == "frame_5" else "transparent")
        self.frame_6_button.configure(fg_color=("gray75", "gray25") if name == "frame_6" else "transparent")
        self.frame_7_button.configure(fg_color=("gray75", "gray25") if name == "frame_7" else "transparent")

        # 展示所选的框架
        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
        if name == "frame_2":
            self.second_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.second_frame.grid_forget()
        if name == "frame_3":
            self.third_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.third_frame.grid_forget()
        if name == "frame_4":
            self.fourth_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.fourth_frame.grid_forget()
        if name == "frame_5":
            self.fifth_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.fifth_frame.grid_forget()
        if name == "frame_6":
            self.sixth_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.sixth_frame.grid_forget()
        if name == "frame_7":
            self.seventh_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.seventh_frame.grid_forget()


    def home_button_event(self):
        self.select_frame_by_name("home")

    # frame_2
    def frame_2_button_event(self):
        self.select_frame_by_name("frame_2")

    # frame_3
    def frame_3_button_event(self):
        self.select_frame_by_name("frame_3")

    # frame_4
    def frame_4_button_event(self):
        self.select_frame_by_name("frame_4")

    # frame_5
    def frame_5_button_event(self):
        self.select_frame_by_name("frame_5")

    # frame_6
    def frame_6_button_event(self):
        self.select_frame_by_name("frame_6")

    # frame_7
    def frame_7_button_event(self):
        self.select_frame_by_name("frame_7")

    def change_appearance_mode_event(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    """
    下面为与作业相关的函数,后期会进行简化
    """

    # 选择图像1
    def select_image1(self):
        self.image1_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        print(f"Selected Image 1 Path: {self.image1_path}")  # 打印路径
        if self.image1_path:
            self.label1.configure(text=f"输入图片一: {self.image1_path.split('/')[-1]}")
    # 选择图像2
    def select_image2(self):
        self.image2_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        print(f"Selected Image 2 Path: {self.image2_path}")  # 打印路径
        if self.image2_path:
            self.label2.configure(text=f"输入图片二: {self.image2_path.split('/')[-1]}")

   # 选择图像3
    def select_image3(self):
        self.image3_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        print(f"Selected Image 1 Path: {self.image3_path}")  # 打印路径
        if self.image3_path:
            self.label3.configure(text=f"输入图片一: {self.image3_path.split('/')[-1]}")

    # 选择图像4
    def select_image4(self):
        self.image4_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        print(f"Selected Image 2 Path: {self.image4_path}")  # 打印路径
        if self.image4_path:
            self.label4.configure(text=f"输入图片二: {self.image4_path.split('/')[-1]}")

    #图像拼接相关函数
    def sift_keypoints_detect(image):
        # 处理图像一般很少用到彩色信息，通常直接将图像转换为灰度图
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # 获取图像特征sift-SIFT特征点,实例化对象sift
        #sift = cv.xfeatures2d.SIFT_create()
        sift = cv.SIFT_create()

        # keypoints:特征点向量,向量内的每一个元素是一个KeyPoint对象，包含了特征点的各种属性信息(角度、关键特征点坐标等)
        # features:表示输出的sift特征向量，通常是128维的
        keypoints, features = sift.detectAndCompute(image, None)

        # cv.drawKeyPoints():在图像的关键特征点部位绘制一个小圆圈
        # 如果传递标志flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,它将绘制一个大小为keypoint的圆圈并显示它的方向
        # 这种方法同时显示图像的坐标，大小和方向，是最能显示特征的一种绘制方式
        keypoints_image = cv.drawKeypoints(
            gray_image, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        # 返回带关键特征点的图像、关键特征点和sift的特征向量
        return keypoints_image, keypoints, features

    # 使用KNN检测来自左右图像的SIFT特征，随后进行匹配
    def get_feature_point_ensemble(features_right, features_left):
        # 创建BFMatcher对象解决匹配
        bf = cv.BFMatcher()
        # knnMatch()函数：返回每个特征点的最佳匹配k个匹配点
        # features_right为模板图，features_left为匹配图
        matches = bf.knnMatch(features_right, features_left, k=2)
        # 利用sorted()函数对matches对象进行升序(默认)操作
        matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)
        # x:x[]字母可以随意修改，排序方式按照中括号[]里面的维度进行排序，[0]按照第一维排序，[2]按照第三维排序

        # 建立列表good用于存储匹配的点集
        good = []
        for m, n in matches:
            # ratio的值越大，匹配的线条越密集，但错误匹配点也会增多
            ratio = 0.6
            if m.distance < ratio * n.distance:
                good.append(m)

        # 返回匹配的关键特征点集
        return good

    # 计算视角变换矩阵H，用H对右图进行变换并返回全景拼接图像
    def Panorama_stitching(image_right, image_left):
        _, keypoints_right, features_right = App.sift_keypoints_detect(image_right)
        _, keypoints_left, features_left = App.sift_keypoints_detect(image_left)
        goodMatch = App.get_feature_point_ensemble(features_right, features_left)

        # 当筛选项的匹配对大于4对(因为homography单应性矩阵的计算需要至少四个点)时,计算视角变换矩阵
        if len(goodMatch) > 4:
            # 获取匹配对的点坐标
            ptsR = np.float32(
                [keypoints_right[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            ptsL = np.float32(
                [keypoints_left[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)

            # ransacReprojThreshold：将点对视为内点的最大允许重投影错误阈值(仅用于RANSAC和RHO方法时),若srcPoints和dstPoints是以像素为单位的，该参数通常设置在1到10的范围内
            ransacReprojThreshold = 4

            # cv.findHomography():计算多个二维点对之间的最优单映射变换矩阵 H(3行x3列),使用最小均方误差或者RANSAC方法
            # 函数作用:利用基于RANSAC的鲁棒算法选择最优的四组配对点，再计算转换矩阵H(3*3)并返回,以便于反向投影错误率达到最小
            Homography, status = cv.findHomography(
                ptsR, ptsL, cv.RANSAC, ransacReprojThreshold)

            # cv.warpPerspective()：透视变换函数，用于解决cv2.warpAffine()不能处理视场和图像不平行的问题
            # 作用：就是对图像进行透视变换，可保持直线不变形，但是平行线可能不再平行
            Panorama = cv.warpPerspective(
                image_right, Homography, (image_right.shape[1] + image_left.shape[1], image_right.shape[0]))

            # cv.imshow("扭曲变换后的右图", Panorama)
            cv.waitKey(0)
            cv.destroyAllWindows()
            # 将左图加入到变换后的右图像的左端即获得最终图像
            Panorama[0:image_left.shape[0], 0:image_left.shape[1]] = image_left

            # 返回全景拼接的图像
            return Panorama

    # 图像拼接
    def concat_images(self):
        if not self.image1_path or not self.image2_path:
            self.result_label.configure(text="请选择两张图片")
            return

        # 使用OpenCV加载图片
        ima = cv.imread(self.image1_path)
        imb = cv.imread(self.image2_path)

        # 检查图片是否加载成功
        if ima is None:
            self.result_label.configure(text="无法加载输入图片一")
            return
        if imb is None:
            self.result_label.configure(text="无法加载输入图片二")
            return
        imb = cv.resize(imb, None, fx=0.4, fy=0.24)
        ima = cv.resize(ima, (imb.shape[1], imb.shape[0]))

        # 获取检测到关键特征点后的图像的相关参数
        keypoints_image_right, keypoints_right, features_right = App.sift_keypoints_detect(imb)
        keypoints_image_left, keypoints_left, features_left = App.sift_keypoints_detect(ima)

        # 利用np.hstack()函数同时将原图和绘有关键特征点的图像沿着竖直方向(水平顺序)堆叠起来
        #cv.imshow("左图关键特征点检测", np.hstack((ima, keypoints_image_left)))
        # 一般在imshow后设置 waitKey(0) , 代表按任意键继续
        cv.waitKey(0)
        # 删除先前建立的窗口
        cv.destroyAllWindows()
        #cv.imshow("右图关键特征点检测", np.hstack((imb, keypoints_image_right)))
        cv.waitKey(0)
        cv.destroyAllWindows()
        goodMatch = App.get_feature_point_ensemble(features_right, features_left)

        # cv.drawMatches():在提取两幅图像特征之后，画出匹配点对连线
        # matchColor – 匹配的颜色（特征点和连线),若matchColor==Scalar::all(-1),颜色随机
        all_goodmatch_image = cv.drawMatches(
            imb, keypoints_right, ima, keypoints_left, goodMatch, None, None, None, None, flags=2)
        #cv.imshow("所有匹配的SIFT关键特征点连线", all_goodmatch_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # 把图片拼接成全景图并保存
        new_img = App.Panorama_stitching(imb, ima)
        new_img_path = "pinjie.png"
        cv.imwrite(new_img_path, new_img)

        # 使用PIL显示拼接后的图片
        new_img_pil = Image.open(new_img_path)
        new_img_tk = ImageTk.PhotoImage(new_img_pil)
        self.image_label.configure(image=new_img_tk)
        self.image_label.image = new_img_tk
        self.result_label.configure(text="拼接成功: pinjie.png")

    #平行视图矫正相关函数
    def rectify_stereo_cameras(imgL, imgR, ptsL, ptsR, F):
        # 估计本质矩阵E
        h, w, _ = imgL.shape
        K = np.eye(3)  # 假设相机内参矩阵为单位矩阵，实际应用中应从相机标定得到
        E, _ = cv.findEssentialMat(ptsL, ptsR, K)

        # 从本质矩阵恢复旋转和平移
        _, R, T, _ = cv.recoverPose(E, ptsL, ptsR, K)

        # 将旋转矩阵转换为旋转向量
        r, _ = cv.Rodrigues(R)

        # 分别计算左右摄像机的旋转向量
        rL = 0.5 * r
        rR = -0.5 * r

        # 计算左右摄像机的旋转矩阵
        RL, _ = cv.Rodrigues(rL)
        RR, _ = cv.Rodrigues(rR)

        # 应用透视变换
        H1 = np.dot(K, np.dot(RL, np.linalg.inv(K)))
        H2 = np.dot(K, np.dot(RR, np.linalg.inv(K)))
        imgL_rectified = cv.warpPerspective(imgL, H1, (w, h))
        imgR_rectified = cv.warpPerspective(imgR, H2, (w, h))

        return imgL_rectified, imgR_rectified

    # 平行视图矫正
    def find_fundamental_matrix(self):
        if not self.image3_path or not self.image4_path:
            self.result_label.configure(text="请选择两张图片")
            return

        imgL = cv.imread(self.image3_path)
        imgR = cv.imread(self.image4_path)

        # 检查照片是否已经加载
        if imgL is None:
            self.result_label.configure(text="无法加载输入图片一")
            return
        if imgR is None:
            self.result_label.configure(text="无法加载输入图片二")
            return

        sift = cv.SIFT_create()
        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
        keypointsL, descriptorsL = sift.detectAndCompute(grayL, None)
        keypointsR, descriptorsR = sift.detectAndCompute(grayR, None)

        flann = cv.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
        matches = flann.knnMatch(descriptorsL, descriptorsR, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) >= 8:
            ptsL = np.float32([keypointsL[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            ptsR = np.float32([keypointsR[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            imgL_rectified, imgR_rectified = App.rectify_stereo_cameras(imgL, imgR, ptsL, ptsR, None)

            # 把图片拼接成全景图并保存
            pinxin_img_path1 = "rectified_left.jpg"
            pinxin_img_path2 = "rectified_right.jpg"
            cv.imwrite(pinxin_img_path1, imgL_rectified)
            cv.imwrite(pinxin_img_path2, imgR_rectified)

            # Load and resize the images using PIL
            px_img_pil_1 = Image.open(pinxin_img_path1)
            px_img_pil_2 = Image.open(pinxin_img_path2)

            # Define the size you want to resize the images to
            max_size = (200, 150)  # For example, resize to a maximum of 300x300 pixels

            px_img_pil_1.thumbnail(max_size, Image.Resampling.LANCZOS)
            px_img_pil_2.thumbnail(max_size, Image.Resampling.LANCZOS)
            px_img_tk_1 = ImageTk.PhotoImage(px_img_pil_1)
            px_img_tk_2 = ImageTk.PhotoImage(px_img_pil_2)

            self.image_label_1.configure(image=px_img_tk_1)
            self.image_label_1.image = px_img_tk_1
            self.image_label_2.configure(image=px_img_tk_2)
            self.image_label_2.image = px_img_tk_2

            self.result_label_px.configure(text="平行视图: rectified_left.jpg，rectified_right.jpg 矫正成功")

            print("矫正后的图像已保存。")
        else:
            print("未找到足够的匹配点。")

    # 特征图抽取
    def extract_features(self):
        img1 = cv.imread('I1.bmp', cv.IMREAD_GRAYSCALE)
        img2 = cv.imread('I2.bmp', cv.IMREAD_GRAYSCALE)

        # 初始化SIFT检测器
        sift = cv.SIFT_create()

        # 检测关键点和描述符
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # 使用FLANN匹配器进行关键点匹配
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # 应用比率测试以过滤好的匹配
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # 绘制匹配结果
        img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None)
        match_img_path = "img_matches.jpg"
        cv.imwrite(match_img_path, img_matches)
        match_img_pil = Image.open(match_img_path)
        max_size_match = (200, 150)
        match_img_pil.thumbnail(max_size_match, Image.Resampling.LANCZOS)
        match_img_tk = ImageTk.PhotoImage(match_img_pil)

        self.result_image_label_5.configure(image=match_img_tk)
        self.result_image_label_5.image = match_img_tk
        # plt.imshow(img_matches)
        # plt.title('SIFT Matches')
        # plt.show()

    # 矩阵抽取
    def matrix_features(self):
        img1 = cv.imread('I1.bmp', cv.IMREAD_GRAYSCALE)
        img2 = cv.imread('I2.bmp', cv.IMREAD_GRAYSCALE)

        # 初始化SIFT检测器
        sift = cv.SIFT_create()

        # 检测关键点和描述符
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # 使用FLANN匹配器进行关键点匹配
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # 应用比率测试以过滤好的匹配
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # 绘制匹配结果
        img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None)
        match_img_path = "img_matches.jpg"
        cv.imwrite(match_img_path, img_matches)
        match_img_pil = Image.open(match_img_path)
        max_size_match = (200, 150)
        match_img_pil.thumbnail(max_size_match, Image.Resampling.LANCZOS)
        match_img_tk = ImageTk.PhotoImage(match_img_pil)

        # self.result_image_label_5.configure(image=match_img_tk)
        # self.result_image_label_5.image = match_img_tk
        # plt.imshow(img_matches)
        # plt.title('SIFT Matches')
        # plt.show()

        # 提取好的匹配点坐标
        points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
        points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

        for i, match in enumerate(good_matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        # 检查是否有足够的匹配点进行四点法计算
        if len(good_matches) >= 4:
            # 取得分最高的四个匹配点
            best_matches = sorted(good_matches, key=lambda x: x.distance)[:4]
            selected_points1 = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
            selected_points2 = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

            # 计算单应矩阵
            self.h, status = cv.findHomography(selected_points1, selected_points2, cv.RANSAC, 5.0)
            print("使用自动选定的四个点计算的单应矩阵:\n", self.h)
        else:
            print("匹配点不足四对，无法使用四点法计算单应矩阵。")

        # 计算基础矩阵
        self.f, mask = cv.findFundamentalMat(points1, points2, cv.FM_8POINT)
        print("基础矩阵:\n", self.f)
        h_str = np.array2string(self.h, precision=2, separator=', ')
        f_str = np.array2string(self.f, precision=2, separator=', ')
        self.matrix_label_1.configure(text=f"Homography Matrix (h):\n{h_str}")
        self.matrix_label_2.configure(text=f"Fundamental Matrix (f):\n{f_str}")


if __name__ == "__main__":
    app = App()
    app.mainloop()

