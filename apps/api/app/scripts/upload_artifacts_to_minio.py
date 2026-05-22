import argparse
import json
from pathlib import Path
from typing import Any

from app.infra.config import get_settings
from app.infra.minio import MinIOObjectStore
from app.infra.vault import VaultClient, resolve_vault_token


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("Manifest must be a JSON list.")

    return data


def _iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.is_file())

    raise FileNotFoundError(f"Path not found: {path}")


def _make_object_key(*, base_key: str, base_path: Path, file_path: Path) -> str:
    if base_path.is_file():
        return base_key

    relative = file_path.relative_to(base_path)
    return f"{base_key.rstrip('/')}/{relative.as_posix()}"


def upload_from_manifest(
    *,
    manifest_path: Path,
    root: Path,
    dry_run: bool,
    skip_missing: bool,
) -> None:
    settings = get_settings()

    vault = VaultClient(
        addr=settings.vault_addr,
        token=resolve_vault_token(settings),
    )
    app_secrets = vault.read_app_secrets()

    object_store = MinIOObjectStore.from_secrets(app_secrets)
    object_store.ensure_bucket()

    manifest = _load_manifest(manifest_path)

    uploaded: list[str] = []
    skipped: list[str] = []

    for item in manifest:
        local_path_value = item.get("local_path")
        object_key_value = item.get("object_key")

        if not local_path_value or not object_key_value:
            raise ValueError(f"Invalid manifest item: {item}")

        local_path = root / str(local_path_value)
        object_key = str(object_key_value)

        if not local_path.exists():
            message = f"{local_path}"
            if skip_missing:
                skipped.append(message)
                print(f"SKIP missing: {message}")
                continue
            raise FileNotFoundError(f"Manifest path not found: {local_path}")

        files = _iter_files(local_path)

        for file_path in files:
            key = _make_object_key(
                base_key=object_key,
                base_path=local_path,
                file_path=file_path,
            )

            if dry_run:
                print(f"DRY RUN upload: {file_path} -> s3://{object_store.bucket}/{key}")
            else:
                object_store.upload_file(local_path=file_path, key=key)
                print(f"Uploaded: {file_path} -> s3://{object_store.bucket}/{key}")

            uploaded.append(key)

    print("")
    print(f"Bucket: {object_store.bucket}")
    print(f"Uploaded count: {len(uploaded)}")
    print(f"Skipped count: {len(skipped)}")

    if uploaded:
        print("")
        print("Uploaded keys:")
        for key in uploaded:
            print(f"- {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload project artifacts to MinIO.")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the JSON manifest describing what to upload.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root used to resolve local_path values.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would upload without uploading.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing paths instead of failing.",
    )

    args = parser.parse_args()

    upload_from_manifest(
        manifest_path=Path(args.manifest).resolve(),
        root=Path(args.root).resolve(),
        dry_run=args.dry_run,
        skip_missing=args.skip_missing,
    )


if __name__ == "__main__":
    main()