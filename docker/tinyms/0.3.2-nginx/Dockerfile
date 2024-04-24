ARG BASE_CONTAINER=swr.cn-south-1.myhuaweicloud.com/mindspore/mindspore-cpu:1.6.0
FROM $BASE_CONTAINER

LABEL MAINTAINER="TinyMS Authors"

# Install base tools
RUN apt-get update

# Install TinyMS cpu whl package
RUN pip install --upgrade -i https://pypi.mirrors.ustc.edu.cn/simple pip && pip install --no-cache-dir -i https://pypi.mirrors.ustc.edu.cn/simple tinyms==0.3.2
RUN git clone https://github.com/tinyms-ai/tinyms.git

# Ready for tinyms web frontend startup
# Install Nginx and opencv dependencies software
RUN apt-get install nginx=1.14.0-0ubuntu1.11 lsof libglib2.0-dev libsm6 libxrender1 libgl1-mesa-glx -y

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

COPY ./entrypoint.sh /usr/local/bin/
CMD ["entrypoint.sh"]
