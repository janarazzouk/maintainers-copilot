import json
import mimetypes
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError


class MinIOError(RuntimeError):
    pass


class MinIOObjectStore:
    """Small MinIO/S3 adapter.

    Online vector search stays in Postgres/pgvector.
    This adapter is for blob objects:
    - model artifacts
    - eval reports
    - retrieved-chunk snapshots
    - manifests
    - plots
    """

    def __init__(
        self,
        *,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        bucket: str,
    ) -> None:
        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="us-east-1",
        )

    @classmethod
    def from_secrets(cls, secrets: dict[str, Any]) -> "MinIOObjectStore":
        try:
            return cls(
                endpoint_url=str(secrets["minio_endpoint"]),
                access_key=str(secrets["minio_root_user"]),
                secret_key=str(secrets["minio_root_password"]),
                bucket=str(secrets["minio_bucket"]),
            )
        except KeyError as exc:
            raise MinIOError(f"Missing MinIO secret in Vault: {exc.args[0]}") from exc

    def ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except EndpointConnectionError as exc:
            raise MinIOError("MinIO is unreachable.") from exc
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))

            if code in {"404", "NoSuchBucket", "NotFound"}:
                self._client.create_bucket(Bucket=self.bucket)
                return

            raise MinIOError(f"Could not access MinIO bucket {self.bucket}: {code}") from exc

    def put_json(self, *, key: str, payload: dict[str, Any]) -> str:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

        try:
            self._client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
            )
        except Exception as exc:
            raise MinIOError(f"Failed to write object to MinIO: {key}") from exc

        return key

    def get_json(self, *, key: str) -> dict[str, Any]:
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            return json.loads(response["Body"].read().decode("utf-8"))
        except Exception as exc:
            raise MinIOError(f"Failed to read object from MinIO: {key}") from exc

    def upload_file(self, *, local_path: str | Path, key: str) -> str:
        path = Path(local_path)
        if not path.exists():
            raise MinIOError(f"Cannot upload missing file: {path}")
        if not path.is_file():
            raise MinIOError(f"upload_file expected a file, got: {path}")

        content_type, _ = mimetypes.guess_type(str(path))
        extra_args: dict[str, str] = {}
        if content_type:
            extra_args["ContentType"] = content_type

        try:
            self._client.upload_file(
                Filename=str(path),
                Bucket=self.bucket,
                Key=key,
                ExtraArgs=extra_args or None,
            )
        except Exception as exc:
            raise MinIOError(f"Failed to upload {path} to MinIO key {key}") from exc

        return key

    def list_keys(self, *, prefix: str = "") -> list[str]:
        keys: list[str] = []
        continuation_token: str | None = None

        while True:
            kwargs: dict[str, Any] = {
                "Bucket": self.bucket,
                "Prefix": prefix,
            }
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token

            response = self._client.list_objects_v2(**kwargs)

            for item in response.get("Contents", []):
                keys.append(str(item["Key"]))

            if not response.get("IsTruncated"):
                break

            continuation_token = response.get("NextContinuationToken")

        return keys