import time
import logging

from aiohttp import web


def prometheus_middleware(app_name):
    @web.middleware
    async def decorator_factory(request, handler):

        logging.debug('try calculate metrics for prometheus')

        try:
            response = await handler(request)
            request['start_time'] = time.time()
            request.app['REQUEST_IN_PROGRESS'].labels(
                app_name, request.path, request.method
            ).inc()

            resp_time = time.time() - request['start_time']

            request.app['REQUEST_LATENCY'].labels(
                app_name, request.path
                ).observe(resp_time)

            request.app['REQUEST_IN_PROGRESS'].labels(
                app_name, request.path, request.method
            ).dec()

            request.app['REQUEST_COUNT'].labels(
                app_name, request.method, 
                request.path, response.status
            ).inc()         

        except web.HTTPException as e:
            logging.error(e.reason)
            message = e.reason

            request.app['ERROR_REQUEST_COUNT'].labels(
                app_name, request.method, 
                request.path, e.status,
                message, time.time()
            ).inc()

            raise

            return web.json_response({'error': message})
        
        return response

    return decorator_factory