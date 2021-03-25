# Install TinyMS

## User install TinyMS

### Pypi

For users owning a clean environment, recommend using [pypi](https://pypi.org/) to install TinyMS after satisfy the following prerequisites. For setting up python environment, [Anaconda](https://www.anaconda.com/products/individual#Downloads) is a good choice.

Prerequisites  

- Ubuntu: `18.04`
- Python: `3.7.5`

Chinese Domestic users recommend running the following code setting the mirror site of pypi to solve the downloading issue

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

For users' environment do not want to change the local env which also cannot satisfy the prerequisites, using [docker](https://www.docker.com/) to install is recommended

- docker: `v18.06.1-ce`

If user want to try the tutorials that are written in `.ipynb` files，please pull `tinyms-jupyter` which jupyter component are installed by default

```shell
docker pull tinyms/tinyms:0.1.0
```

If want to try jupyter, run the following command line

```shell
docker pull tinyms/tinyms:0.1.0-jupyter
docker run -it --net=host tinyms/tinyms:0.1.0-jupyter
```

Open a browser on the local machine, type in

```URL
<Your_external_IP_address>:8888
```

Example: `159.138.7.105:8888`, the default password is `tinyms`，then user can log in to `jupyter`

## Developer install TinyMS

For developers who want to develope based on TinyMS, install from source

### Install from source

```shell
sudo apt-get install -y libssl-dev
git clone https:github.com/tinyms-ai/tinyms.git
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
