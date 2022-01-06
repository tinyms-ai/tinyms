# 快速安装TinyMS

## 新用户上手安装

### Pypi

对于拥有干净环境的用户，在满足以下环境要求后，推荐使用 [pypi](https://pypi.org/) 安装TinyMS。安装python环境，可以使用 [Anaconda](https://www.anaconda.com/products/individual#Downloads)

环境要求

- OS: `Ubuntu 18.04` or `Windows 10`

- Python: `3.7.5`

中国国内用户可以运行以下代码配置国内镜像源，解决下载速度慢的问题

```shell
mkdir -pv /root/.pip \
&& echo "[global]" > /root/.pip/pip.conf \
&& echo "trusted-host=mirrors.aliyun.com" >> /root/.pip/pip.conf \
&& echo "index-url=http://mirrors.aliyun.com/pypi/simple/" >> /root/.pip/pip.conf
```

```shell
pip install tinyms==0.3.1
```

> 注：在执行过程中可能会出现一些问题，以下可能情况仅供参考。在安装过程中你可能会碰到其它问题，我们欢迎你在我们的[社区](https://github.com/tinyms-ai/tinyms)，提出您的Issues和Pull requests，我们会及时回复您。
>
> 1. Error 1：若使用镜像源执行安装命令可能会报 `Could not find a version that satisfies the requirement tinyms==0.3.1`
>
>    **解决方案：**
>
>    - 可以试试使用默认官方源，直接在末尾追加`-i https://pypi.python.org/simple/`，采用默认官方源下载速度可能较慢，请耐心等待:smile:
>
> 2. Error 2：如果是windows用户请确保是否安装了`Microsoft VC++ 14.0`，若没有，安装过程中可能会报`ERROR：Microsoft Visual C++ 14.0 or greater is required.Get it with “Microsoft C++ Build  Tools”: https://visualstudio.microsoft.com/visual-cpp-build-tools/`
>
>    **解决方案：**
>
>    - 因为TinyMS是对`Python3.7.5`环境依赖的，而Python3是通过`VC++ 14.0`编译的。我们可以根据错误提示，在提供的[链接](https://visualstudio.microsoft.com/visual-cpp-build-tools/ )下载`Microsoft C++ Build Tools`，注意在安装过程中需在`使用C++桌面的桌面开发模块`勾选`windows 10 SDK`和`用于Windows的C++ CMake工具`俩个组件，相关安装详情可以参考[Visual studio 生成工具安装]( https://docs.microsoft.com/zh-cn/archive/blogs/c/visual-studio-%E7%94%9F%E6%88%90%E5%B7%A5%E5%85%B7%E4%BB%8B%E7%BB%8D)。

### Docker

对于不想改变本地环境设置的用户，推荐使用 [docker](https://www.docker.com/) 进行安装

- docker: `v18.06.1-ce`

如果想要体验本教程中的`.ipynb`教程，建议拉取TinyMS jupyter版本的镜像，该镜像中除了tinyms外还内置了jupyter组件

如果想要在WEB界面体验图片可视化推理，建议拉取TinyMS nginx版本的镜像，该镜像中除了tinyms外还内置了nginx组件

* 普通版本

```shell
docker pull tinyms/tinyms:0.3.1
docker run -it tinyms/tinyms:0.3.1
```

* Jupyter版本

如果想体验jupyter教程，运行下列命令行

```shell
docker pull tinyms/tinyms:0.3.1-jupyter
docker run -it --net=host tinyms/tinyms:0.3.1-jupyter
```

在本地打开浏览器，输入

```
<公网IP地址>:8888
```

例如 `188.8.8.88:8888`，之后在弹出的页面中，密码输入`tinyms`，就可以远程登录`jupyter`了

* Nginx版本

如果想在可视化WEB界面体验图片推理服务，运行下列命令行

```shell
docker pull tinyms/tinyms:0.3.1-nginx
docker run -itd --name=tinyms-nginx -p 80:80 tinyms/tinyms:0.3.1-nginx /bin/bash

docker exec -it tinyms-nginx /bin/bash
entrypoint.sh <容器所在宿主机的公网IP地址>
```

在本地打开浏览器，输入

```
<容器所在宿主机的公网IP地址>:80
```

## 进阶用户源码安装

想针对TinyMS进行开发的开发者，可以通过源码安装

```shell
sudo apt-get install -y libssl-dev
git clone https://github.com/tinyms-ai/tinyms.git
cd tinyms
pip install -r requirements.txt
python setup.py install
```

## 验证

进入 `python`、 `jupyter` 或 `nginx` 环境，输入以下代码验证安装

```python
import tinyms as ts
from tinyms.primitives import tensor_add

x = ts.ones([2, 3])
y = ts.ones([2, 3])
print(tensor_add(x, y))
```

如果可以看到如下输出，则证明安装成功

```python
[[2. 2. 2.]
 [2. 2. 2.]]
```
