#!/bin/bash
# Run this after refreshing your AWS SSO session:
#   aws sso login (or copy fresh creds from the sandbox console)
# Then: bash scripts/store_secrets_to_aws.sh

set -e

AWS_REGION="us-east-2"
SECRET_ID="sandbox/llm-api-keys"

# Prompt for each key if not already in env
HF_TOKEN=${HF_TOKEN:-$(read -rp "HF_TOKEN: " v && echo "$v")}
NVIDIA_API_KEY=${NVIDIA_API_KEY:-$(read -rp "NVIDIA_API_KEY: " v && echo "$v")}
WANDB_API_KEY=${WANDB_API_KEY:-$(read -rp "WANDB_API_KEY: " v && echo "$v")}
MODAL_TOKEN_ID=${MODAL_TOKEN_ID:-$(read -rp "MODAL_TOKEN_ID: " v && echo "$v")}
MODAL_TOKEN_SECRET=${MODAL_TOKEN_SECRET:-$(read -rp "MODAL_TOKEN_SECRET: " v && echo "$v")}

PAYLOAD=$(python3 -c "
import json, sys
print(json.dumps({
  'HF_TOKEN': '$HF_TOKEN',
  'NVIDIA_API_KEY': '$NVIDIA_API_KEY',
  'WANDB_API_KEY': '$WANDB_API_KEY',
  'MODAL_TOKEN_ID': '$MODAL_TOKEN_ID',
  'MODAL_TOKEN_SECRET': '$MODAL_TOKEN_SECRET',
}))
")

aws secretsmanager put-secret-value \
  --region "$AWS_REGION" \
  --secret-id "$SECRET_ID" \
  --secret-string "$PAYLOAD"

echo "✓ Stored 5 keys in $SECRET_ID ($AWS_REGION)"
