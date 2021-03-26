# 快速安装TinyMS

## 用户安装

### Pypi

对于拥有干净环境的用户，在满足以下环境要求后，推荐使用 [pypi](https://pypi.org/) 安装TinyMS。安装python环境，可以使用 [Anaconda](https://www.anaconda.com/products/individual#Downloads)

环境要求

- Ubuntu: `18.04`
- Python: `3.7.5`

国内用户可以运行以下代码配置国内镜像源，解决下载速度慢的问题

```shell
mkdir -pv /root/.pip \
&& echo "[global]" > /root/.pip/pip.conf \
&& echo "trusted-host=mirrors.aliyun.com" >> /root/.pip/pip.conf \
&& echo "index-url=http://mirrors.aliyun.com/pypi/simple/" >> /root/.pip/pip.conf
```

```shell
pip install tinyms==0.1.0
```

### Docker

对于不想改变本地环境设置的用户，推荐使用 [docker](https://www.docker.com/) 进行安装

- docker: `v18.06.1-ce`

如果想要体验本教程中的`.ipynb`教程，建议拉取TinyMS jupyter版本的镜像，该镜像中除了tinyms外还内置了jupyter组件

* 普通版本

```shell
docker pull tinyms/tinyms:0.1.0
docker run -it tinyms/tinyms:0.1.0
```

* Jupyter版本

如果想体验jupyter教程，运行下列命令行

```shell
docker pull tinyms/tinyms:0.1.0-jupyter
docker run -it --net=host tinyms/tinyms:0.1.0-jupyter
```

在本地打开浏览器，输入

```URL
<公网IP地址>:8888
```

例如 `188.8.8.88:8888`，之后在弹出的页面中，密码输入`tinyms`，就可以远程登录`jupyter`了

## 源码安装

想针对TinyMS进行开发的开发者，可以通过源码安装

```shell
sudo apt-get install -y libssl-dev
git clone https://github.com/tinyms-ai/tinyms.git
cd tinyms
pip install -r requirements.txt
python setup.py install
```

## 验证

进入 `python` 或 `jupyter` 环境，输入以下代码验证安装

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
