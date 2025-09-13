import os

bind = f"0.0.0.0:{os.environ.get('PORT', 8080)}"
workers = int(os.environ.get('WEB_CONCURRENCY', 2))
timeout = int(os.environ.get('TIMEOUT', 120))
worker_class = 'gthread'
threads = int(os.environ.get('THREADS', 2))
max_requests = int(os.environ.get('MAX_REQUESTS', 1000))
max_requests_jitter = 50
preload_app = True
keepalive = 5

# Application module and variable
pythonpath = os.path.dirname(__file__)