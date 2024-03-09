#  YOLOV3

![image-20240112150829170](https://github.com/jiangsu415/YOLOV3/assets/130949548/6c61a68c-4a1f-4e39-8a4d-e0154f738cb3)

进行一个多尺度的不同大小的候选框，分别用来检测不同大小的物体

![image-20240112151315328](https://github.com/jiangsu415/YOLOV3/assets/130949548/718cd151-da43-458a-b4df-d97afcd5723a)

Box1,Box2,Box3分别指的是三种不同规格的先验框，一共是九种先验框
![image-20240112151443973](https://github.com/jiangsu415/YOLOV3/assets/130949548/e5ffa4dc-c087-4e3f-8df3-16a9d944282b)

13 * 13的一个结果做一个上采样变为26 * 26的，与原来的26 * 26做一个融合
![image-20240112152723248](https://github.com/jiangsu415/YOLOV3/assets/130949548/983ec0d0-6319-48bd-8cf9-73bab0626825)

ResNet至少不比原来差，同等映射X，如果加上这一层后效果为0，那么这一层的参数置为0，直接输入上一层的结果

![image-20240112153424928](https://github.com/jiangsu415/YOLOV3/assets/130949548/eb1ae988-6488-4c4e-b3fd-37cfda2f5226)

![image-20240112154123126](https://github.com/jiangsu415/YOLOV3/assets/130949548/061077c6-6998-4ad7-895a-343c5ca8b4c9)

![image-20240112154821310](https://github.com/jiangsu415/YOLOV3/assets/130949548/68cc3dd5-38dd-44e9-b234-37f500c0047f)

![image-20240112155108917](https://github.com/jiangsu415/YOLOV3/assets/130949548/7765757e-ec84-4719-9640-906ce0e358f5)

![image-20240112155527842](https://github.com/jiangsu415/YOLOV3/assets/130949548/13c38053-0398-4e78-808a-95bdb21edcb4)

![img](https://img-blog.csdnimg.cn/8d9057b4d9a94e898a1565f816fe35c9.png#pic_center)
![数据流程图](https://github.com/jiangsu415/YOLOV3/assets/130949548/86519cc8-2580-4af5-b243-84b6508a446f)

![模型](https://github.com/jiangsu415/YOLOV3/assets/130949548/732bf1b2-946c-4236-b418-fca8ea0aa7ec)
![损失函数](https://github.com/jiangsu415/YOLOV3/assets/130949548/be07dfdf-eff3-4460-b3f6-6ac9b248d60f)


![数据流程图](E:\张紫扬01\研01\咕泡学习\目标检测\YOLO\YOLOV3\数据流程图.png)

![未命名文件](E:\张紫扬01\研01\咕泡学习\目标检测\YOLO\YOLOV3\模型.png)
