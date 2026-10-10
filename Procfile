# Heroku: single web dyno is enough for v1.
# With REDIS_URL unset, the webhook handler runs reviews inline as a
# background task (see src/prism/main.py), so no worker dyno or Redis
# addon is needed until you outgrow one dyno.
web: bash scripts/heroku_web.sh

# Scale-up path only (needs REDIS_URL + a Redis addon):
# worker: python -m prism.worker
