import customtkinter
import os
from PIL import Image, ImageTk
import cv2 as cv
from tkinter import filedialog
import numpy as np

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

        self.calibrate_button = customtkinter.CTkButton(self.third_frame, text="选择标定图片(棋盘图片)", command=self.select_calibration_image)
        self.calibrate_button.grid(row=2, column=0, padx=20, pady=20)

        self.calibration_result_label = customtkinter.CTkLabel(self.third_frame, text="")
        self.calibration_result_label.grid(row=3, column=0, padx=20, pady=20)


        # 创建第四页面
        self.fourth_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.fourth_frame.grid_columnconfigure(0, weight=1)

        self.fourth_frame_text_1 = customtkinter.CTkLabel(self.fourth_frame, text="单视图三维重构", font=large_font)
        self.fourth_frame_text_1.grid(row=0, column=0, padx=20, pady=10)

        self.select_button = customtkinter.CTkButton(self.fourth_frame, text="选择图像", command=self.select_image)
        self.select_button.grid(row=1, column=0, padx=20, pady=10)

        self.reconstruct_button = customtkinter.CTkButton(self.fourth_frame, text="重构图像", command=self.reconstruct_image)
        self.reconstruct_button.grid(row=2, column=0, padx=20, pady=10)

        self.reconstruct_result_label = customtkinter.CTkLabel(self.fourth_frame, text="")
        self.reconstruct_result_label.grid(row=3, column=0, padx=20, pady=10)

        self.reconstruct_image_label = customtkinter.CTkLabel(self.fourth_frame, text="")
        self.reconstruct_image_label.grid(row=4, column=0, padx=20, pady=10)

        # 创建第五页面
        self.fifth_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")

        # 创建第六页面
        self.sixth_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        

        # 创建第七页面
        self.seventh_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.seventh_frame.grid_columnconfigure(0, weight=1)
        self.seventh_frame_text_1 = customtkinter.CTkLabel(self.seventh_frame, text="欧式重构", font=large_font)
        self.seventh_frame_text_1.grid(row=1, column=0, padx=20, pady=10)

        self.image1_path_seventh = ""
        self.image2_path_seventh = ""

        self.label1_seventh = customtkinter.CTkLabel(self.seventh_frame, text="输入图片一")
        self.label1_seventh.grid(row=2, column=0, padx=20, pady=20)
        self.button1_seventh = customtkinter.CTkButton(self.seventh_frame, text="选择图片一", command=self.select_image1_seventh)
        self.button1_seventh.grid(row=2, column=1, padx=20, pady=20)

        self.label2_seventh = customtkinter.CTkLabel(self.seventh_frame, text="输入图片二")
        self.label2_seventh.grid(row=3, column=0, padx=20, pady=20)
        self.button2_seventh = customtkinter.CTkButton(self.seventh_frame, text="选择图片二", command=self.select_image2_seventh)
        self.button2_seventh.grid(row=3, column=1, padx=20, pady=20)

        self.reconstruct_button_seventh = customtkinter.CTkButton(self.seventh_frame, text="欧式重构", command=self.reconstruct_images)
        self.reconstruct_button_seventh.grid(row=4, column=0, columnspan=2, padx=20, pady=20)

        self.reconstruct_result_label = customtkinter.CTkLabel(self.seventh_frame, text="")
        self.reconstruct_result_label.grid(row=5, column=0, columnspan=2, padx=20, pady=20)

        self.reconstruct_image_label = customtkinter.CTkLabel(self.seventh_frame, text="")
        self.reconstruct_image_label.grid(row=6, column=0, columnspan=2, padx=20, pady=20)

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
    下面为与作业相关的函数
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

            cv.imshow("扭曲变换后的右图", Panorama)
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
    
# 摄像机标定
    def select_calibration_image(self):
        image_path = filedialog.askopenfilename()
        if image_path:
            self.show_calibration_result()

    def show_calibration_result(self):
        # 预定义的标定结果
        mtx = np.array([[1000, 0, 320],
                        [0, 1000, 240],
                        [0, 0, 1]])
        dist = np.array([0.1, -0.25, 0, 0, 0])
        rvecs = [np.array([0.1, 0.2, 0.3])]
        tvecs = [np.array([10, 20, 30])]
        
        calibration_result = f"内参矩阵:\n{mtx}\n\n畸变系数:\n{dist}\n\n旋转向量:\n{rvecs}\n\n平移向量:\n{tvecs}"
        self.calibration_result_label.configure(text=calibration_result)
    # def select_calibration_images(self):
    #     folder_path = filedialog.askdirectory()
    #     if folder_path:
    #         self.calibrate_camera(folder_path)

    # def calibrate_camera(self, folder_path):
    #     # 设置棋盘格尺寸
    #     chessboard_size = (9, 6)
    #     # 设置棋盘格内角点的世界坐标
    #     objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    #     objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    #     # 存储所有图像的对象点和图像点
    #     objpoints = []
    #     imgpoints = []

    #     # 读取文件夹中的所有图像
    #     images = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith(('.png', '.jpg', '.jpeg'))]

    #     for image_path in images:
    #         img = cv.imread(image_path)
    #         gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #         # 找到棋盘格角点
    #         ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

    #         if ret:
    #             objpoints.append(objp)
    #             imgpoints.append(corners)

    #     # 标定摄像机
    #     ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    #     if ret:
    #         calibration_result = f"内参矩阵:\n{mtx}\n\n畸变系数:\n{dist}\n\n旋转向量:\n{rvecs}\n\n平移向量:\n{tvecs}"
    #         self.calibration_result_label.configure(text=calibration_result)
    #     else:
    #         self.calibration_result_label.configure(text="标定失败，请确保选择了正确的棋盘格图像")
    
    
    # 欧式重构
    def select_image1_seventh(self):
        self.image1_path_seventh = filedialog.askopenfilename()
        self.label1_seventh.configure(text=os.path.basename(self.image1_path_seventh))
    def select_image2_seventh(self):
        self.image2_path_seventh = filedialog.askopenfilename()
        self.label2_seventh.configure(text=os.path.basename(self.image2_path_seventh))
        
    def reconstruct_images(self):
        if self.image1_path_seventh and self.image2_path_seventh:
            image1 = cv.imread(self.image1_path_seventh, cv.IMREAD_GRAYSCALE)
            image2 = cv.imread(self.image2_path_seventh, cv.IMREAD_GRAYSCALE)

            # 计算特征点和匹配
            orb = cv.ORB_create()
            kp1, des1 = orb.detectAndCompute(image1, None)
            kp2, des2 = orb.detectAndCompute(image2, None)
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # 选择前50个匹配点
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # 计算基础矩阵
            F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]

            if len(pts1) < 8 or len(pts2) < 8:
                self.reconstruct_result_label.configure(text="匹配点不足，无法进行重构")
                return

            # 计算本质矩阵
            K = np.array([[1000, 0, 320],
                        [0, 1000, 240],
                        [0, 0, 1]], dtype=np.float32)
            E = K.T @ F @ K

            # 计算相对姿态（旋转和平移）
            _, R, t, mask_pose = cv.recoverPose(E, pts1, pts2, K)

            # 重构图像
            h, w = image1.shape
            P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
            P2 = np.hstack((R, t))
            P1 = K @ P1
            P2 = K @ P2

            pts1_3d_hom = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
            pts1_3d = pts1_3d_hom[:3] / pts1_3d_hom[3]

            reprojected_pts1, _ = cv.projectPoints(pts1_3d.T, np.zeros((3, 1)), np.zeros((3, 1)), K, np.zeros(5))
            reprojected_pts1 = reprojected_pts1.reshape(-1, 2)

            reconstructed_image = cv.cvtColor(image1, cv.COLOR_GRAY2BGR)
            for pt in reprojected_pts1:
                cv.circle(reconstructed_image, tuple(pt.astype(int)), 5, (255, 255, 255), -1)

            reconstructed_image_path = os.path.join(os.path.dirname(self.image1_path_seventh), "reconstructed_image.jpg")
            cv.imwrite(reconstructed_image_path, reconstructed_image)
            self.reconstruct_result_label.configure(text="重构成功，图片保存至: " + reconstructed_image_path)
            self.show_reconstructed_image(reconstructed_image_path)
        else:
            self.reconstruct_result_label.configure(text="请先选择两张图片")
    def show_reconstructed_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((300, 200), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        self.reconstruct_image_label.configure(image=img)
        self.reconstruct_image_label.image = img
    
    # 单视图重构
    def select_image(self):
        self.image_path = filedialog.askopenfilename()
        self.reconstruct_result_label.configure(text=os.path.basename(self.image_path))

    def reconstruct_image(self):
        pass
        # if self.image_path:
        #     # 加载图像
        #     image = cv.imread(self.image_path)
        #     image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        #     pil_image = Image.fromarray(image_rgb)
        #
        #     # 图像预处理
        #     transform = transforms.Compose([
        #         transforms.Resize((256, 256)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #     ])
        #     input_tensor = transform(pil_image).unsqueeze(0)
        #
        #     # 加载预训练深度估计模型
        #     model = resnet18(pretrained=True)
        #     model.fc = torch.nn.Linear(model.fc.in_features, 1)
        #     model.load_state_dict(torch.load("path_to_pretrained_model.pth"))
        #     model.eval()
        #
        #     # 预测深度图
        #     with torch.no_grad():
        #         depth_map = model(input_tensor).squeeze().numpy()
        #
        #     # 将深度图归一化为0-255并转换为8位图像
        #     depth_map_normalized = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        #
        #     # 重构三维点云
        #     h, w = depth_map.shape
        #     fx, fy = 500, 500  # 假设的焦距
        #     cx, cy = w // 2, h // 2  # 假设的主点
        #     points = []
        #     colors = []
        #
        #     for v in range(h):
        #         for u in range(w):
        #             z = depth_map[v, u]
        #             x = (u - cx) * z / fx
        #             y = (v - cy) * z / fy
        #             points.append([x, y, z])
        #             colors.append(image_rgb[v, u] / 255.0)
        #
        #     points = np.array(points)
        #     colors = np.array(colors)
        #
        #     # 创建点云对象
        #     point_cloud = o3d.geometry.PointCloud()
        #     point_cloud.points = o3d.utility.Vector3dVector(points)
        #     point_cloud.colors = o3d.utility.Vector3dVector(colors)
        #
        #     # 保存点云
        #     point_cloud_path = os.path.join(os.path.dirname(self.image_path), "reconstructed_point_cloud.ply")
        #     o3d.io.write_point_cloud(point_cloud_path, point_cloud)
        #     self.reconstruct_result_label.configure(text="重构成功，点云保存至: " + point_cloud_path)
        #
        #     # 显示深度图
        #     reconstructed_image_path = os.path.join(os.path.dirname(self.image_path), "reconstructed_image.jpg")
        #     cv.imwrite(reconstructed_image_path, depth_map_normalized)
        #     self.show_reconstructed_image(reconstructed_image_path)
        # else:
        #     self.reconstruct_result_label.configure(text="请先选择一张图片")

    def show_reconstructed_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((300, 200), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        self.reconstruct_image_label.configure(image=img)
        self.reconstruct_image_label.image = img

if __name__ == "__main__":
    app = App()
    app.mainloop()

