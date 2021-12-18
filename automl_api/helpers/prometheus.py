from prometheus_client import Counter, Gauge, Histogram

def setup_prometheus(app):
    app['REQUEST_COUNT'] = Counter(
        'request_total', 'Total Request Count',
        ['app_name', 'method', 'endpoint', 'http_status']
    )

    app['REQUEST_LATENCY'] = Histogram(
        'request_latency_seconds', 'Request latency',
        ['app_name', 'endpoint']
    )

    app['REQUEST_IN_PROGRESS'] = Gauge(
        'requests_in_progress_total', 'Requests in progress',
        ['app_name', 'endpoint', 'method']
    )

    app['FIT_JOB_ENQUEUED'] = Counter(
        'fit_job_equeued', 'Total job enqueued',
        ['app_name']
    )