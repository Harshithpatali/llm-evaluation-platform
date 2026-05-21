import json
import redis

from config import settings

redis_client = redis.Redis.from_url(
    settings.redis_url,
    decode_responses=True,
    ssl_cert_reqs=None
)

results = redis_client.lrange(
    "llm_evaluations",
    0,
    -1
)

for item in results:
    print(json.loads(item))