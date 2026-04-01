#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Teardown all OCR comparison playground resources
# ============================================================

AWS_PROFILE="${AWS_PROFILE:-playground-dev}"
AWS_REGION="${AWS_REGION:-us-east-1}"
STACK_NAME="ocr-comparison"
ECR_REPOS=(
  ocr-comparison/paddle
  ocr-comparison/tesseract
  ocr-comparison/surya
  ocr-comparison/doctr
  ocr-comparison/gateway
)

export AWS_PROFILE AWS_REGION

info()  { echo -e "\033[1;34m▶ $*\033[0m"; }
ok()    { echo -e "\033[1;32m✔ $*\033[0m"; }
warn()  { echo -e "\033[1;33m⚠ $*\033[0m"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------
# Terminate EC2 instances
# ----------------------------------------------------------
info "Finding instances..."
instance_ids=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=${STACK_NAME}" "Name=instance-state-name,Values=running,stopped" \
  --query 'Reservations[].Instances[].InstanceId' --output text)

if [[ -n "${instance_ids}" && "${instance_ids}" != "None" ]]; then
  info "Terminating: ${instance_ids}"
  aws ec2 terminate-instances --instance-ids ${instance_ids} >/dev/null
  aws ec2 wait instance-terminated --instance-ids ${instance_ids}
  ok "Instances terminated"
else
  ok "No instances found"
fi

# ----------------------------------------------------------
# Delete security group
# ----------------------------------------------------------
sg_id=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=${STACK_NAME}-sg" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")

if [[ "${sg_id}" != "None" && -n "${sg_id}" ]]; then
  info "Deleting security group ${sg_id}..."
  for i in {1..6}; do
    aws ec2 delete-security-group --group-id "${sg_id}" 2>/dev/null && break
    echo "  Waiting for ENIs to detach (attempt ${i}/6)..."
    sleep 10
  done
  ok "Security group deleted"
fi

# ----------------------------------------------------------
# Remove IAM instance profile + role
# ----------------------------------------------------------
role_name="${STACK_NAME}-role"
profile_name="${STACK_NAME}-profile"

if aws iam get-instance-profile --instance-profile-name "${profile_name}" >/dev/null 2>&1; then
  info "Removing IAM profile and role..."
  aws iam remove-role-from-instance-profile \
    --instance-profile-name "${profile_name}" \
    --role-name "${role_name}" 2>/dev/null || true
  aws iam delete-instance-profile --instance-profile-name "${profile_name}"

  aws iam detach-role-policy --role-name "${role_name}" \
    --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly 2>/dev/null || true
  aws iam detach-role-policy --role-name "${role_name}" \
    --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess 2>/dev/null || true
  aws iam delete-role --role-name "${role_name}"
  ok "IAM resources deleted"
fi

# ----------------------------------------------------------
# Delete key pair
# ----------------------------------------------------------
if aws ec2 describe-key-pairs --key-names "${STACK_NAME}" >/dev/null 2>&1; then
  aws ec2 delete-key-pair --key-name "${STACK_NAME}"
  ok "Key pair deleted"
fi

# ----------------------------------------------------------
# Delete ECR repositories (optional)
# ----------------------------------------------------------
echo ""
read -rp "Delete all ECR repositories and images? [y/N] " del_ecr
if [[ "${del_ecr}" =~ ^[Yy]$ ]]; then
  for repo in "${ECR_REPOS[@]}"; do
    if aws ecr describe-repositories --repository-names "${repo}" >/dev/null 2>&1; then
      aws ecr delete-repository --repository-name "${repo}" --force >/dev/null
      ok "  Deleted ${repo}"
    fi
  done
  ok "ECR repositories deleted"
else
  warn "ECR repositories kept (images will persist and may incur storage costs)"
fi

# ----------------------------------------------------------
# Clean up local state
# ----------------------------------------------------------
rm -f "${SCRIPT_DIR}/.deploy-state"

echo ""
ok "All playground resources cleaned up!"
ok "Estimated cost stopped."
