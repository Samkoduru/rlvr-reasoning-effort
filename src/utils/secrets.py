import boto3
import json
import os

SECRET_NAME = "sandbox/llm-api-keys"
REGION = "us-east-2"

# Keys stored in AWS Secrets Manager (and mirrored as Modal secrets for training runs)
SECRET_KEYS = ["HF_TOKEN", "NVIDIA_API_KEY", "WANDB_API_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"]


def load_secrets(secret_name: str = SECRET_NAME, region: str = REGION) -> dict:
    """Pull all LLM API keys from AWS Secrets Manager into os.environ."""
    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_name)
    secrets = json.loads(response["SecretString"])
    for key, value in secrets.items():
        os.environ.setdefault(key, value)
    return secrets


def get_secret(key: str, secret_name: str = SECRET_NAME, region: str = REGION) -> str:
    """Retrieve a single key from Secrets Manager."""
    secrets = load_secrets(secret_name, region)
    if key not in secrets:
        raise KeyError(f"{key} not found in {secret_name}")
    return secrets[key]


if __name__ == "__main__":
    secrets = load_secrets()
    print("Loaded keys:", list(secrets.keys()))
