#!/usr/bin/env bash

celery -A tasks worker --loglevel=info &

uvicorn app:app --host 0.0.0.0 --port 10000