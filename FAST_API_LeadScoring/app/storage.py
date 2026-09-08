from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import quote

from app.config import (
    AWS_REGION,
    AWS_S3_BUCKET_NAME,
    AWS_S3_ENDPOINT_URL,
    AWS_S3_PREFIX,
    AWS_S3_PUBLIC_BASE_URL,
    AWS_S3_SERVER_SIDE_ENCRYPTION,
)


@dataclass(frozen=True)
class SourceFileUploadResult:
    saved_to_s3: bool
    status: str
    bucket: str | None = None
    key: str | None = None
    uri: str | None = None
    url: str | None = None


def upload_source_file_to_s3(
    *,
    file_bytes: bytes,
    filename: str,
    content_type: str | None,
) -> SourceFileUploadResult:
    if not AWS_S3_BUCKET_NAME:
        return SourceFileUploadResult(
            saved_to_s3=False,
            status="S3_BUCKET_NAME is not configured; source Excel file was not uploaded to S3.",
        )

    try:
        import boto3
    except ImportError:
        return SourceFileUploadResult(
            saved_to_s3=False,
            status="boto3 is not installed. Install boto3 to enable S3 source file uploads.",
        )

    bucket = AWS_S3_BUCKET_NAME
    key = build_source_file_key(filename)
    put_object_args: dict[str, object] = {
        "Bucket": bucket,
        "Key": key,
        "Body": file_bytes,
        "ContentType": content_type or "application/octet-stream",
    }
    if AWS_S3_SERVER_SIDE_ENCRYPTION:
        put_object_args["ServerSideEncryption"] = AWS_S3_SERVER_SIDE_ENCRYPTION

    try:
        client = boto3.client(
            "s3",
            region_name=AWS_REGION,
            endpoint_url=AWS_S3_ENDPOINT_URL,
        )
        client.put_object(**put_object_args)
    except Exception as exc:
        return SourceFileUploadResult(
            saved_to_s3=False,
            status=f"S3 upload failed: {exc}",
            bucket=bucket,
            key=key,
            uri=f"s3://{bucket}/{key}",
            url=build_source_file_url(bucket, key),
        )

    return SourceFileUploadResult(
        saved_to_s3=True,
        status="Source Excel file was uploaded to S3.",
        bucket=bucket,
        key=key,
        uri=f"s3://{bucket}/{key}",
        url=build_source_file_url(bucket, key),
    )


def build_source_file_key(filename: str) -> str:
    date_path = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    safe_filename = sanitize_filename(filename)
    prefix = f"{AWS_S3_PREFIX}/" if AWS_S3_PREFIX else ""
    return f"{prefix}{date_path}/{uuid.uuid4()}-{safe_filename}"


def sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", filename.strip()).strip(".-")
    return cleaned or "employee-leads.xlsx"


def build_source_file_url(bucket: str, key: str) -> str | None:
    if AWS_S3_PUBLIC_BASE_URL:
        return f"{AWS_S3_PUBLIC_BASE_URL.rstrip('/')}/{quote(key)}"
    if AWS_S3_ENDPOINT_URL:
        return None
    return f"https://{bucket}.s3.{AWS_REGION}.amazonaws.com/{quote(key)}"
