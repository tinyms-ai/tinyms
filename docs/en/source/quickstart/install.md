# Install TinyMS

## Installation For Beginners

### Pypi

For users who own a clean environment, it is recommended to use [pypi](https://pypi.org/) to install TinyMS given that the following requirements are meet. For those who don't, [Anaconda](https://www.anaconda.com/products/individual#Downloads) is a good choice for setting up the python environment.

Prerequisites

- OS: `Ubuntu 18.04` or `Windows 10`
- Python: `3.7.5`

For China based users it is recommended to run the following command lines to help with faster download

```shell
mkdir -pv /root/.pip \
&& echo "[global]" > /root/.pip/pip.conf \
&& echo "trusted-host=mirrors.aliyun.com" >> /root/.pip/pip.conf \
&& echo "index-url=http://mirrors.aliyun.com/pypi/simple/" >> /root/.pip/pip.conf
```

```shell
pip install tinyms==0.3.1
```

> Note: There may be some problems during the installation process. The following possible situations are for reference only. If you encounter other problems during the installation process, we welcome you  to submit your issues and pull requests in our [community](https://github.com/tinyms-ai/tinyms), and we will reply you as soon as possible.
>
> 1. Error 1: If you use the mirror source to execute the installation command, it may report `Could not find a version that satisfies the requirement tinyms==0.3.1`
>
>    **Solution:**
>
>    - You can try to use the default official source, directly append `-i https://pypi.python.org/simple/` at the end, the download speed of the default official source may be slower, please be patient :smile:
>
> 2. Error 2: If you are a windows user, please make sure that `Microsoft VC++ 14.0` is installed. If not, it may report`ERROR: Microsoft Visual C++ 14.0 or greater is required. Get it with “Microsoft C++ Build Tools” may be reported during the installation process. : https://visualstudio.microsoft.com/visual-cpp-build-tools/`
>
>    **Solution:**
>
>    - Because TinyMS is dependent on the `Python3.7.5` environment, and Python3 is compiled with `VC++ 14.0`. According to the error prompt, download `Microsoft C++ Build Tools` at the provided [link](https://visualstudio.microsoft.com/visual-cpp-build-tools/) . Note that during the installation process, the two components `windows 10 SDK` and `C++ CMake Tools for Windows` need to be checked in `Desktop Development Module Using C++ Desktop`. For installation details, please refer to [Visual Studio Build Tool Installation](https://devblogs.microsoft.com/cppblog/introducing-the-visual-studio-build-tools/).
>

### Docker

For those who don't want to affect the local develop environment due to difficulty of meeting the prerequisites, using [docker](https://www.docker.com/) to install is recommended

- docker: `v18.06.1-ce`

If user wants to try the tutorials that are written in `.ipynb` files，please pull jupyter version of TinyMS in which jupyter components are installed by default

If user wants to experience the image inference service in a visual WEB UI，please pull nginx version of TinyMS in which nginx components are installed by default


* Default version

```shell
docker pull tinyms/tinyms:0.3.1
docker run -it tinyms/tinyms:0.3.1
```

* Jupyter version

If user wants to try jupyter, run the following command line

```shell
docker pull tinyms/tinyms:0.3.1-jupyter
docker run -it --net=host tinyms/tinyms:0.3.1-jupyter
```

Open a browser on the local machine, type in

```
<Your_external_IP_address>:8888
```

Example: `188.8.8.88:8888`, the default password is `tinyms`，then user can log in to `jupyter`

* Nginx version

If user wants to experience the image inference service in a visual WEB UI, run the following command line

```shell
docker pull tinyms/tinyms:0.3.1-nginx
docker run -itd --name=tinyms-nginx -p 80:80 tinyms/tinyms:0.3.1-nginx /bin/bash

docker exec -it tinyms-nginx /bin/bash
entrypoint.sh <Your_host_public_IP_address_not_docker_IP_address>
```

Open a browser on the local machine, type in

```
<Your_host_public_IP_address_not_docker_IP_address>:80
```

## Installation For Experienced Developers

For developers who want to develop based on TinyMS, install from source

```shell
sudo apt-get install -y libssl-dev
git clone https://github.com/tinyms-ai/tinyms.git
cd tinyms
pip install -r requirements.txt
python setup.py install
```

## Validate installation

Create a `python`, `jupyter` or `nginx` kernel, input the following codes

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

## Notes

When we use `TinyMS 0.3.1`, the following error may be reported

**Error Details:**

````
[ERROR] ME(24148:23792,MainProcess):2022-01-25-21:59:25.562.448 [mindspore\_extends\parse\parser.py:565] When eval 'P.tensor_add(identity, x)' by using Fallback feature, an error occurred: name 'identity' is not defined. You can try to turn off the Fallback feature by 'export MS_DEV_ENABLE_FALLBACK=0'.
````

**Solution:**

According to the error prompt, we can turn off the `Fallback feature` with the following command.

For general users, execute the following commands in the command line tool:

```shell
export MS_DEV_ENABLE_FALLBACK=0
````

For users using jupyter, execute the following command in the cell:

````python
!export MS_DEV_ENABLE_FALLBACK=0
````

If you report other error while using `TinyMS 0.3.1`, after you try to solve the error, there is still a problem, we welcome you to submit your issues and pull requests in our [community](https://github.com/tinyms-ai/tinyms), and we will reply you as soon as possible.

