#!/usr/bin/env python3
"""Validate SageMaker setup: credentials, S3, Secrets Manager, IAM role. No training launched."""
import sys
from pathlib import Path

import boto3
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_config() -> dict:
    cfg_path = ROOT / "config" / "sagemaker.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text()) or {}


def main():
    cfg = _load_config()
    bucket = cfg.get("s3_bucket")
    role_arn = cfg.get("iam_role")
    region = cfg.get("region", "us-east-1")
    secret_name = cfg.get("hf_secret_name")

    errors = []

    # 1. Credentials
    print("1. Checking AWS credentials...")
    try:
        sts = boto3.client("sts", region_name=region)
        identity = sts.get_caller_identity()
        print(f"   OK  Account: {identity['Account']}, ARN: {identity['Arn']}")
    except Exception as e:
        errors.append(f"Credentials: {e}")
        print(f"   FAIL {e}")
        sys.exit(1)

    # 2. IAM role ARN format
    print("\n2. Validating IAM role ARN...")
    if not role_arn or "ACCOUNT_ID" in role_arn:
        errors.append("iam_role not set or contains ACCOUNT_ID placeholder")
        print("   FAIL Replace ACCOUNT_ID in config/sagemaker.yaml")
    elif ":root:role/" in role_arn:
        suggested = role_arn.replace(":root:role/", ":role/")
        print(f"   WARN Role ARN has ':root:' - typical format is arn:aws:iam::ACCOUNT:role/NAME")
        print(f"        Suggested: {suggested}")
    else:
        print(f"   OK  {role_arn}")

    # 3. IAM role exists (optional: requires iam:GetRole; skip if AccessDenied)
    print("\n3. Checking IAM role exists...")
    role_name = None
    try:
        iam = boto3.client("iam")
        if "/" in role_arn:
            role_name = role_arn.split("/")[-1]
        else:
            role_name = role_arn
        iam.get_role(RoleName=role_name)
        print(f"   OK  Role '{role_name}' exists")
    except Exception as e:
        resp = getattr(e, "response", None)
        err_code = resp.get("Error", {}).get("Code", "") if resp else ""
        if err_code == "AccessDenied":
            print(f"   SKIP Cannot verify (sagemaker-user lacks iam:GetRole). Ensure role exists and you have iam:PassRole for training.")
        elif "NoSuchEntity" in str(e) or err_code == "NoSuchEntity":
            errors.append(f"IAM role '{role_name}' not found")
            print(f"   FAIL Role '{role_name}' not found")
        else:
            errors.append(f"IAM role check: {e}")
            print(f"   FAIL {e}")

    # 4. S3 bucket
    print("\n4. Checking S3 bucket...")
    if not bucket:
        errors.append("s3_bucket not set in config")
        print("   FAIL s3_bucket required")
    else:
        try:
            s3 = boto3.client("s3", region_name=region)
            s3.head_bucket(Bucket=bucket)
            print(f"   OK  Bucket: {bucket}")
            s3.list_objects_v2(Bucket=bucket, MaxKeys=1)
            print(f"   OK  Bucket accessible (list works)")
        except Exception as e:
            resp = getattr(e, "response", None)
            code = resp.get("Error", {}).get("Code", "") if resp else ""
            if code == "404" or "Not Found" in str(e):
                try:
                    create_kwargs = {"Bucket": bucket}
                    if region != "us-east-1":
                        create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
                    s3.create_bucket(**create_kwargs)
                    print(f"   OK  Created bucket: {bucket}")
                except Exception as create_err:
                    errors.append(f"S3 bucket create: {create_err}")
                    print(f"   FAIL Could not create bucket: {create_err}")
            else:
                errors.append(f"S3: {e}")
                print(f"   FAIL {e}")

    # 5. Secrets Manager (HF token)
    print("\n5. Checking Secrets Manager...")
    if not secret_name:
        print("   SKIP hf_secret_name not set")
    else:
        try:
            sm = boto3.client("secretsmanager", region_name=region)
            sm.get_secret_value(SecretId=secret_name)
            print(f"   OK  Secret '{secret_name}' accessible")
        except Exception as e:
            if "ResourceNotFoundException" in type(e).__name__ or "ResourceNotFound" in str(e):
                errors.append(f"Secret '{secret_name}' not found")
                print(f"   FAIL Secret not found. Create it: aws secretsmanager create-secret --name {secret_name} --secret-string 'hf_xxx'")
            else:
                errors.append(f"Secrets Manager: {e}")
                print(f"   FAIL {e}")

    # 6. SageMaker API (read-only)
    print("\n6. Checking SageMaker access...")
    try:
        sm = boto3.client("sagemaker", region_name=region)
        sm.list_training_jobs(MaxResults=1)
        print("   OK  SageMaker API accessible")
    except Exception as e:
        errors.append(f"SageMaker: {e}")
        print(f"   FAIL {e}")

    # Summary
    print("\n" + "=" * 50)
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("All checks passed. Ready for training.")
    sys.exit(0)


if __name__ == "__main__":
    main()
