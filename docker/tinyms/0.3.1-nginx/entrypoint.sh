set -e

host_addr=$1

if [[ -z $host_addr ]];
then
    echo "nothing to be instead".
else
    sed -i 's/127.0.0.1/'$host_addr'/' /etc/nginx/nginx.conf
fi
/etc/init.d/nginx reload && /etc/init.d/nginx restart

python -c "from tinyms.serving import Server; server = Server(); server.start_server()" &
