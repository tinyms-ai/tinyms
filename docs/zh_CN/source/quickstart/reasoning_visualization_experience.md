# TinyMS推理可视化体验

结合OpenCV图像视觉库，TinyMS v0.3.1聚焦可视化特性。通过简单直观的图片可视化，帮助用户更快地了解模型推理的效果。

针对不想编写代码的用户，TinyMS提供了WEB UI可视化界面，用户只需在浏览器页面上传待推理的图片即可轻松体验，当前提供了`LeNet5`、`CycleGan`和`SSD300`模型的支持。

### WEB UI推理可视化

用户需要先部署可视化服务器，详情请看[Nginx版本的TinyMS](https://tinyms.readthedocs.io/zh_CN/latest/quickstart/install.html)安装。服务器部署成功后，访问浏览器呈现的首页和推理效果页（以`CycleGan`模型为例）如下：

![首页](../_static/tinyms_web_index.jpg)

![推理页](../_static/tinyms_web_reasoning.jpg)

针对想运行代码的用户，TinyMS提供了模型推理可视化模块，仅需`5`步骤代码即可快速体验，当前仅提供`SSD300`对象检测模型的支持。

### 模型推理可视化模块应用

如果您需要第一时间体验模型推理可视化模块应用，可下载[TinyMS官方仓项目](https://github.com/tinyms-ai/tinyms)代码，执行如下操作：

* 静态图像检测

1. 环境准备

   * 有可视化桌面的操作系统（比如`Window_x64`或者`Ubuntu18.04`）

2. 验证模块应用

   ```script
   # 下载tinyms项目
   git clone https://github.com/tinyms-ai/tinyms.git
   cd tinyms/tests/st/app/object_detection/
   # 运行静态图像检测
   python opencv_image_app.py
   ```

   待检测图片和执行推理后的图片展示如下：
   
   ![待检测图片](../_static/tinyms_visulization_origin.jpg)
   
   ![推理效果图](../_static/tinyms_visulization_reasoning.jpg)

* 摄像头采集的动态视频图像检测

1. 环境准备：

   * 带有可视化桌面的操作系统（`Window_x64`或者`Ubuntu18.04`）

   * 确保操作系统可以正常访问摄像头

     > 注：
     >
     > 对于在主机（比如`Window_x64`和`Ubuntu 18.04`）下的操作系统，一般来说都可以正常访问摄像头；而对于在虚拟机下的操作系统，请确保**开启了相关的虚拟机服务以及连接了摄像头驱动**。我们以在window下的虚拟机`VMware Workstation`为例：
     >
     > 1. 首先我们在terminal（您的测试系统的terminal，比如挂在VM的`Ubuntu 18.04`系统下的terminal）下输入命令`ls /dev/v*`查看是否有`/dev/video0`驱动程序，若有，则说明是正常的，请忽略以下操作，若没有，则我们执行以下操作。
     > 2. 开启相关的虚拟机服务，在windows主机中开启服务`VMware USB Arbitration Servcie`，即键盘`Win+R`输入`services.msc`找到相应服务并开启。开启之后需将虚拟机重启。
     > 3. 连接摄像头驱动，在`VMware Workstation`的菜单栏上点击`虚拟机(M)`=>`可移动设备`=>`摄像头名称`=>`主机连接`，并且在`虚拟机(M)`=>`设置(S)`=>`USB`中，选择`USB3.0`。
     > 4. 你可以通过`cheese`测试摄像头是不是可以正常访问。

2. 验证模块应用

   对于不同操作系统的选型以及不同测试方式的选择，我们提供以下具体环境的验证，请您在对应环境进行验证，注意**以下环境均已满足环境准备的俩个条件，这很重要**。

   - 环境一：Windows

     - 若您采用主机测试

       **环境需求：**

       - 操作系统：`Window_64`

       - 环境依赖： `Git` + `Python 3. 7.5` + `TinyMS 0.3.1` + `Microsoft Visual C++ 14.0 or greater`

       - 命令行工具：`Git Bash`

         > 注：有关`vc++ 14.0` 的环境依赖详情请参考[Pypi安装TinyMS](https://tinyms.readthedocs.io/zh_CN/latest/quickstart/install.html)下的注释

       满足环境需求后执行以下命令：

       ```script
       # 1.在容器内下载tinyms项目
       git clone https://github.com/tinyms-ai/tinyms.git
       cd tinyms/tests/st/app/object_detection/
       # 2.运行摄像头采集的动态视频图像检测
       python opencv_camera_app.py
       ```

     - 若您采用虚拟机测试

       我们以`Window_x64`下的虚拟机`VMware Workstation`为例，VM连接摄像头请参考环境准备中的注释：

       **环境需求：**

       - 操作系统：`Ubuntu18.04 LTS Desktop`
       
       - 环境依赖： `Docker 18.06.1-ce`
       
       - 命令行工具：`Terminal`
       
       满足环境需求后执行以下命令：
       
       ```script
       # 1.在宿主机内安装xserver，并设置权限
       apt install x11-xserver-utils
       # 2.允许所有用户访问显示接口
       xhost +
       # 3.运行容器
       docker run -it --rm --device=/dev/video0 -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix tinyms/tinyms:0.3.1 /bin/bash
       # 4.在容器内下载tinyms项目
       git clone https://github.com/tinyms-ai/tinyms.git
       cd tinyms/tests/st/app/object_detection/
       # 5.运行摄像头采集的动态视频图像检测
       python opencv_camera_app.py
       ```
   - 环境二：Ubuntu

     - 若您采用主机测试：

       **环境需求：**

       - 操作系统：`Ubuntu 18.04 LTS Desktop`
       - 环境依赖：`Git` + `Python 3.7.5` +  `TinyMS 0.3.1`
       - 命令行工具：`Terminal`
       
       满足环境需求后执行以下命令
       
       ```script
       # 1.在容器内下载tinyms项目
       git clone https://github.com/tinyms-ai/tinyms.git
       cd tinyms/tests/st/app/object_detection/
       # 2.运行摄像头采集的动态视频图像检测
       python opencv_camera_app.py
       ```
       
     - 若您采用docker访问
     
       **环境需求：**
     
       - 操作系统：`Ubuntu 18.04 LTS Desktop`
       - 环境依赖： `Docker 18.06.1-ce`
       - 命令行工具：`Terminal`
     
       满足环境需求后执行以下命令：
     
       ```script
       # 1.在宿主机内安装xserver，并设置权限
       apt install x11-xserver-utils
       # 2.允许所有用户访问显示接口
       xhost +
       # 3.运行容器
       docker run -it --rm --device=/dev/video0 -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix tinyms/tinyms:0.3.1 /bin/bash
       # 4.在容器内下载tinyms项目
       git clone https://github.com/tinyms-ai/tinyms.git
       cd tinyms/tests/st/app/object_detection/
       # 5.运行摄像头采集的动态视频图像检测
       python opencv_camera_app.py
       ```
   
   目前文档还在完善中:smile:，如果您的环境不在以上参考中，在您尝试过后，依然有问题，欢迎您在我们的[社区](https://github.com/tinyms-ai/tinyms)，提出您的Issues和Pull requests，我们会及时回复您。
