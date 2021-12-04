FROM ubuntu:20.04

WORKDIR /backend

COPY start_rq_worker.sh /backend

RUN mkdir -p /backend/supervisor_out

RUN apt update
RUN apt install python3 python3-pip -y
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install supervisor
RUN python3 -m pip install rq

RUN mkdir -p /var/log/supervisor
COPY supervisord_rq.conf /etc/supervisor/conf.d/supervisord.conf

CMD ["/usr/local/bin/supervisord", "-c","/etc/supervisor/conf.d/supervisord.conf"]
