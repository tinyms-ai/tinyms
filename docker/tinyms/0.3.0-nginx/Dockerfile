ARG BASE_CONTAINER=swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-cpu:1.5.0
FROM $BASE_CONTAINER

LABEL MAINTAINER="TinyMS Authors"

# Install base tools
RUN apt-get update

# Install TinyMS cpu whl package
RUN pip install --no-cache-dir tinyms==0.3.0
RUN git clone https://github.com/tinyms-ai/tinyms.git

# Ready for tinyms web frontend startup
# Install Nginx and opencv dependencies software
RUN apt-get install nginx=1.14.0-0ubuntu1.9 lsof libglib2.0-dev libsm6 libxrender1 -y

# Configure Nginx
RUN sed -i '/include \/etc\/nginx\/sites-enabled\/\*;/a\
            client_max_body_size 200M;\
            client_body_buffer_size 200M;\
            server {\
                    listen 80;\
                    server_name 127.0.0.1;\
                    root /tinyms/tinyms/serving/web;\
                    index index.html;\
                    location /predict {\
                            add_header Access-Control-Allow-Origin *;\
                            add_header Access-Control-Allow-Methods "GET,POST,OPTIONS";\
                            add_header Access-Control-Allow-Headers "DNT,X-Mx-ReqToken,Keep-Alive,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Authorization";\
                            proxy_pass http://localhost:5000/predict;\
                    }\
            }' /etc/nginx/nginx.conf &&\
     /etc/init.d/nginx start

# Ready for tinyms web backend startup
RUN mkdir -p /etc/tinyms/serving && cp /tinyms/tinyms/serving/config/servable.json /etc/tinyms/serving &&\
    mkdir /etc/tinyms/serving/lenet5_mnist && cd /etc/tinyms/serving/lenet5_mnist &&\
    wget https://tinyms.obs.cn-north-4.myhuaweicloud.com/ckpt_file/lenet5_mnist/lenet5.ckpt &&\
    mkdir /etc/tinyms/serving/cyclegan_cityscape && cd /etc/tinyms/serving/cyclegan_cityscape &&\
    wget https://tinyms.obs.cn-north-4.myhuaweicloud.com/ckpt_file/cyclegan_cityscape/G_A.ckpt &&\
    wget https://tinyms.obs.cn-north-4.myhuaweicloud.com/ckpt_file/cyclegan_cityscape/G_B.ckpt &&\
    mkdir /etc/tinyms/serving/ssd300_shanshui && cd /etc/tinyms/serving/ssd300_shanshui &&\
    wget https://tinyms.obs.cn-north-4.myhuaweicloud.com/ckpt_file/ssd300_shanshui/ssd300.ckpt

COPY ./entrypoint.sh /usr/local/bin/
CMD ["entrypoint.sh"]