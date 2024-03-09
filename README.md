#  YOLOV3

![image-20240112150829170](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20240112150829170.png)

进行一个多尺度的不同大小的候选框，分别用来检测不同大小的物体

![image-20240112151315328](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20240112151315328.png)

Box1,Box2,Box3分别指的是三种不同规格的先验框，一共是九种先验框

![image-20240112151443973](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20240112151443973.png)

13 * 13的一个结果做一个上采样变为26 * 26的，与原来的26 * 26做一个融合

![image-20240112152723248](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20240112152723248.png)

ResNet至少不比原来差，同等映射X，如果加上这一层后效果为0，那么这一层的参数置为0，直接输入上一层的结果

![image-20240112153424928](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20240112153424928.png)



![image-20240112154123126](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20240112154123126.png)

![image-20240112154821310](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20240112154821310.png)

![image-20240112155108917](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20240112155108917.png)

![image-20240112155527842](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20240112155527842.png)

![img](https://img-blog.csdnimg.cn/8d9057b4d9a94e898a1565f816fe35c9.png#pic_center)



![数据流程图](E:\张紫扬01\研01\咕泡学习\目标检测\YOLO\YOLOV3\数据流程图.png)

![未命名文件](E:\张紫扬01\研01\咕泡学习\目标检测\YOLO\YOLOV3\模型.png)

![image-20240131154749762](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20240131154749762.png)
