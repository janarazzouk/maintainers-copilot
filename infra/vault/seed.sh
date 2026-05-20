#!/bin/sh
set -e

export VAULT_ADDR="${VAULT_ADDR:-http://vault:8200}"
export VAULT_TOKEN="${VAULT_TOKEN:-root}"

echo "Waiting for Vault at $VAULT_ADDR..."

i=0
until vault status -address="$VAULT_ADDR"; do
  i=$((i + 1))

  if [ "$i" -gt 60 ]; then
    echo "Vault did not become ready after 60 seconds."
    exit 1
  fi

  echo "Vault not ready yet... retry $i"
  sleep 1
done

echo "Vault is ready."
echo "Seeding secrets into Vault..."

vault kv put secret/app \
  database_user="postgres" \
  database_password="postgres" \
  database_name="maintainers_copilot" \
  database_host="db" \
  database_port="5432" \
  database_url="postgresql+psycopg2://postgres:postgres@db:5432/maintainers_copilot" \
  jwt_signing_key="dev-super-secret-jwt-key-change-later" \
  minio_root_user="minioadmin" \
  minio_root_password="minioadmin" \
  minio_endpoint="http://minio:9000" \
  minio_bucket="maintainers-copilot" 

echo "Vault secrets seeded."
