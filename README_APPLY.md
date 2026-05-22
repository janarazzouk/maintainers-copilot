# Auth + MinIO patch

Copy these files into the same paths in your repo.

Then run:

```bash
docker compose up -d vault vault-init db minio redis

docker compose run --rm migrate

docker compose up -d api