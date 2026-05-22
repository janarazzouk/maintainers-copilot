#!/bin/sh
set -e

export VAULT_ADDR="${VAULT_ADDR:-http://vault:8200}"

if [ -n "${VAULT_TOKEN_FILE:-}" ] && [ -f "${VAULT_TOKEN_FILE}" ]; then
  export VAULT_TOKEN="$(cat "${VAULT_TOKEN_FILE}")"
else
  export VAULT_TOKEN="${VAULT_TOKEN:-${VAULT_DEV_ROOT_TOKEN_ID:-root}}"
fi

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

echo "Checking Vault token capabilities..."
vault token capabilities secret/data/app

echo "Ensuring KV v2 secret engine exists..."
vault secrets enable -path=secret kv-v2 2>/dev/null || true

echo "Seeding secrets into Vault..."

if vault kv get secret/app >/dev/null 2>&1; then
  echo "secret/app exists. Patching existing keys without deleting other secrets..."

  vault kv patch secret/app \
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
else
  echo "secret/app does not exist. Creating it for the first time..."

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
fi

echo "Vault secrets seeded."