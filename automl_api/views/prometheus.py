from aiohttp import web
import prometheus_client
from prometheus_client import CONTENT_TYPE_LATEST

async def metrics(request):
    resp = web.Response(body=prometheus_client.generate_latest(), status=200)
    resp.content_type = CONTENT_TYPE_LATEST
    
    return resp