# Install TinyMS

## User install

### Pypi

For users who own a clean environment, it is recommended to use [pypi](https://pypi.org/) to install TinyMS given that the following requirements are meet. For those who don't, [Anaconda](https://www.anaconda.com/products/individual#Downloads) is a good choice for setting up the python environment.

Prerequisites  

- Ubuntu: `18.04`
- Python: `3.7.5`

For China based users it is recommended to run the following command lines to help with faster download

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

For those who don't want to affect the local develop environment due to difficulty of meeting the prerequisites, using [docker](https://www.docker.com/) to install is recommended

- docker: `v18.06.1-ce`

If user wants to try the tutorials that are written in `.ipynb` files，please pull jupyter version of TinyMS in which jupyter components are installed by default

* Default version

```shell
docker pull tinyms/tinyms:0.1.0
```

* Jupyter version

If user wants to try jupyter, run the following command line

```shell
docker pull tinyms/tinyms:0.1.0-jupyter
docker run -it --net=host tinyms/tinyms:0.1.0-jupyter
```

Open a browser on the local machine, type in

```URL
<Your_external_IP_address>:8888
```

Example: `188.8.8.88:8888`, the default password is `tinyms`，then user can log in to `jupyter`

## TinyMS Installation For Experienced Developers

For developers who want to develop based on TinyMS, install from source

```shell
sudo apt-get install -y libssl-dev
git clone https://github.com/tinyms-ai/tinyms.git
cd tinyms
pip install -r requirements.txt
python setup.py install
```

## Validate installation

Create a `python` or `jupyter` kernel, input the following codes

```python
import tinyms as ts
from tinyms.primitives import tensor_add

x = ts.ones([2, 3])
y = ts.ones([2, 3])
print(tensor_add(x, y))
```

If the output is similar to below, then the installation is valid

```python
[[2. 2. 2.]
 [2. 2. 2.]]
```
