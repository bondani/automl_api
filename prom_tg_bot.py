from aiohttp import web
import requests

from automl_api.settings import load_config

config = None

STATUS = {
    'firing': '🔥 firing'
}

async def alert(request):
    text = await request.json()

    api_key = config['telegram']['api_key']
    admin_id = config['telegram']['admin_id']

    alert = text['alerts'][-1]

    status = STATUS.get(alert['status']) or alert['status']

    message = f'Error occured:%0A' \
              f'status: {status}%0A' \
              f'endpoint: {alert["labels"]["endpoint"]}%0A' \
              f'status_code: {alert["labels"]["http_status"]}%0A' \
              f'message: {alert["labels"]["message"]}'

    url = f'https://api.telegram.org/bot{api_key}/sendMessage?chat_id={admin_id}&text={message}'

    print(url)

    requests.get(url)

    return 'Alert OK', 200



def main(configpath):

    global config

    config = load_config(configpath)

    app = web.Application()

    app.add_routes([
        web.post('/alert', alert)
    ])

    port = config['prom_tg_bot']['api_port']

    web.run_app(app, port='8081')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    
    args = parser.parse_args()

    if args.config:
        main(args.config)

    else:
        parser.print_help()