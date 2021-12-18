FROM ubuntu:20.04

WORKDIR /backend

COPY automl_api /backend/automl_api
COPY automl_scheduler /backend/automl_scheduler
COPY requirements.txt /backend
COPY telegram_bot.py /backend
COPY prom_tg_bot.py /backend

RUN mkdir -p /backend/supervisor_out

RUN apt update
RUN apt install python3 python3-pip -y
RUN python3 -m pip install --upgrade pip install -r requirements.txt
RUN python3 -m pip install supervisor

RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

CMD ["/usr/local/bin/supervisord", "-c","/etc/supervisor/conf.d/supervisord.conf"]
