#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# OCR Comparison Platform — Playground Deployment
#
# Deploys all 5 services (paddle, tesseract, surya, doctr, gateway)
# to a single g4dn.2xlarge EC2 instance with docker compose.
#
# Prerequisites:
#   aws sso login --profile playground-dev
#
# Usage:
#   ./infra/deploy.sh          # full deploy (build + push + launch + smoke test)
#   ./infra/deploy.sh push     # build & push all images only
#   ./infra/deploy.sh launch   # launch EC2 + run compose only
#   ./infra/deploy.sh test     # run smoke test against running instance
# ============================================================

AWS_PROFILE="${AWS_PROFILE:-playground-dev}"
AWS_REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="612232746043"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# g4dn.2xlarge: 1x T4 (16 GB VRAM), 8 vCPU, 32 GB RAM — ~$1.05/hr
# Needed: paddle 12G + surya 8G + doctr 6G + tesseract 2G + gateway 0.5G ≈ 29G
INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.2xlarge}"
KEY_NAME="${KEY_NAME:-ocr-comparison}"
STACK_NAME="ocr-comparison"
DISK_SIZE=80  # GB — models are large

ECR_BASE="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
SERVICES=(paddle tesseract surya doctr gateway)

export AWS_PROFILE AWS_REGION

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

info()  { echo -e "\033[1;34m▶ $*\033[0m"; }
ok()    { echo -e "\033[1;32m✔ $*\033[0m"; }
err()   { echo -e "\033[1;31m✘ $*\033[0m" >&2; }
warn()  { echo -e "\033[1;33m⚠ $*\033[0m"; }

# ----------------------------------------------------------
# State file: persist instance IP across commands
# ----------------------------------------------------------
STATE_FILE="${SCRIPT_DIR}/.deploy-state"

save_state() { echo "$1" > "${STATE_FILE}"; }
load_state() { [[ -f "${STATE_FILE}" ]] && cat "${STATE_FILE}" || echo ""; }

# ----------------------------------------------------------
# Step 1: Create ECR repositories (one per service)
# ----------------------------------------------------------
create_ecr_repos() {
  info "Ensuring ECR repositories..."
  for svc in "${SERVICES[@]}"; do
    local repo="ocr-comparison/${svc}"
    aws ecr describe-repositories --repository-names "${repo}" >/dev/null 2>&1 || \
      aws ecr create-repository --repository-name "${repo}" --image-scanning-configuration scanOnPush=true >/dev/null
    ok "  ${repo}"
  done
}

# ----------------------------------------------------------
# Step 2: Build & push all Docker images
# ----------------------------------------------------------
build_and_push() {
  info "Logging into ECR..."
  aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${ECR_BASE}"

  cd "${PROJECT_DIR}"

  for svc in "${SERVICES[@]}"; do
    local repo="ocr-comparison/${svc}"
    local ecr_uri="${ECR_BASE}/${repo}"
    local dockerfile="services/${svc}/Dockerfile"

    info "Building ${svc}..."

    local build_args=(
      --platform linux/amd64
      -f "${dockerfile}"
      -t "${repo}:${IMAGE_TAG}"
    )

    # PaddleOCR needs GPU target
    if [[ "${svc}" == "paddle" ]]; then
      build_args+=(--target gpu)
    fi

    docker build "${build_args[@]}" .

    docker tag "${repo}:${IMAGE_TAG}" "${ecr_uri}:${IMAGE_TAG}"
    docker push "${ecr_uri}:${IMAGE_TAG}"
    ok "  ${svc} pushed → ${ecr_uri}:${IMAGE_TAG}"
  done
}

# ----------------------------------------------------------
# Step 3: SSH key pair
# ----------------------------------------------------------
ensure_key_pair() {
  if ! aws ec2 describe-key-pairs --key-names "${KEY_NAME}" >/dev/null 2>&1; then
    info "Creating EC2 key pair '${KEY_NAME}'..."
    aws ec2 create-key-pair \
      --key-name "${KEY_NAME}" \
      --query 'KeyMaterial' \
      --output text > "${SCRIPT_DIR}/${KEY_NAME}.pem"
    chmod 400 "${SCRIPT_DIR}/${KEY_NAME}.pem"
    ok "Key saved to infra/${KEY_NAME}.pem"
  else
    ok "Key pair '${KEY_NAME}' already exists"
  fi
}

# ----------------------------------------------------------
# Step 4: GPU AMI
# ----------------------------------------------------------
get_gpu_ami() {
  aws ssm get-parameters \
    --names /aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended \
    --query 'Parameters[0].Value' \
    --output text | python3 -c "import sys,json; print(json.load(sys.stdin)['image_id'])"
}

# ----------------------------------------------------------
# Step 5: Security group (SSH + service ports)
# ----------------------------------------------------------
ensure_security_group() {
  local vpc_id sg_id

  vpc_id=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)

  sg_id=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=${STACK_NAME}-sg" "Name=vpc-id,Values=${vpc_id}" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")

  if [[ "${sg_id}" == "None" || -z "${sg_id}" ]]; then
    info "Creating security group..." >&2
    sg_id=$(aws ec2 create-security-group \
      --group-name "${STACK_NAME}-sg" \
      --description "OCR comparison - SSH and service ports" \
      --vpc-id "${vpc_id}" \
      --query 'GroupId' --output text)

    aws ec2 authorize-security-group-ingress \
      --group-id "${sg_id}" --protocol tcp --port 22 --cidr 0.0.0.0/0
    # Gateway (8080) + individual services (8081-8084)
    aws ec2 authorize-security-group-ingress \
      --group-id "${sg_id}" --protocol tcp --port 8080-8084 --cidr 0.0.0.0/0
  fi

  ok "Security group: ${sg_id}" >&2
  echo "${sg_id}"
}

# ----------------------------------------------------------
# Step 6: IAM instance profile
# ----------------------------------------------------------
ensure_instance_profile() {
  local role_name="${STACK_NAME}-role"
  local profile_name="${STACK_NAME}-profile"

  if aws iam get-instance-profile --instance-profile-name "${profile_name}" >/dev/null 2>&1; then
    return
  fi

  info "Creating IAM role for ECR pull..."

  aws iam create-role \
    --role-name "${role_name}" \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "ec2.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }]
    }' >/dev/null

  aws iam attach-role-policy \
    --role-name "${role_name}" \
    --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

  aws iam attach-role-policy \
    --role-name "${role_name}" \
    --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess

  aws iam create-instance-profile --instance-profile-name "${profile_name}" >/dev/null
  aws iam add-role-to-instance-profile \
    --instance-profile-name "${profile_name}" \
    --role-name "${role_name}"

  sleep 10
  ok "IAM instance profile ready"
}

# ----------------------------------------------------------
# Step 7: Launch EC2 + docker compose
# ----------------------------------------------------------
launch_instance() {
  local ami_id sg_id instance_id public_ip

  info "Resolving GPU AMI..."
  ami_id=$(get_gpu_ami)
  ok "AMI: ${ami_id}"

  ensure_key_pair
  sg_id=$(ensure_security_group)

  # Check if already running
  instance_id=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=${STACK_NAME}" "Name=instance-state-name,Values=running" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || echo "None")

  if [[ "${instance_id}" != "None" && -n "${instance_id}" ]]; then
    public_ip=$(aws ec2 describe-instances --instance-ids "${instance_id}" \
      --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    ok "Instance already running: ${instance_id} (${public_ip})"
    save_state "${public_ip}"
    echo "${public_ip}"
    return
  fi

  ensure_instance_profile

  # Generate docker-compose for remote (uses ECR image URIs)
  local compose_file
  compose_file=$(mktemp)
  generate_remote_compose > "${compose_file}"

  # User data: install compose, pull images, start services
  local user_data_file
  user_data_file=$(mktemp)
  cat > "${user_data_file}" <<USERDATA
#!/bin/bash
set -ex

# Wait for Docker + NVIDIA drivers
for i in {1..30}; do docker info >/dev/null 2>&1 && break; sleep 5; done
for i in {1..30}; do nvidia-smi >/dev/null 2>&1 && break; sleep 5; done

# Install AWS CLI
if ! command -v aws &>/dev/null; then
  yum install -y aws-cli unzip 2>/dev/null || true
  # Fallback: install v2
  if ! command -v aws &>/dev/null; then
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    cd /tmp && unzip -q awscliv2.zip && ./aws/install && cd -
  fi
fi

# Install docker compose plugin
DOCKER_CONFIG=\${DOCKER_CONFIG:-/usr/local/lib/docker}
mkdir -p \${DOCKER_CONFIG}/cli-plugins
curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 -o \${DOCKER_CONFIG}/cli-plugins/docker-compose
chmod +x \${DOCKER_CONFIG}/cli-plugins/docker-compose

# Login to ECR
REGION=\$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
aws ecr get-login-password --region "\${REGION}" | \
  docker login --username AWS --password-stdin "${ECR_BASE}"

# Write compose file
mkdir -p /opt/ocr
cat > /opt/ocr/docker-compose.yml <<'COMPOSE'
$(cat "${compose_file}")
COMPOSE

# Pull and start
cd /opt/ocr
docker compose pull
docker compose up -d

echo "OCR comparison platform started" > /opt/ocr/deploy-complete
USERDATA

  info "Launching ${INSTANCE_TYPE}..."

  instance_id=$(aws ec2 run-instances \
    --image-id "${ami_id}" \
    --instance-type "${INSTANCE_TYPE}" \
    --key-name "${KEY_NAME}" \
    --security-group-ids "${sg_id}" \
    --iam-instance-profile "Name=${STACK_NAME}-profile" \
    --user-data "file://${user_data_file}" \
    --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=${DISK_SIZE},VolumeType=gp3}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${STACK_NAME}}]" \
    --query 'Instances[0].InstanceId' --output text)

  rm -f "${user_data_file}" "${compose_file}"

  info "Waiting for instance ${instance_id}..."
  aws ec2 wait instance-running --instance-ids "${instance_id}"

  public_ip=$(aws ec2 describe-instances --instance-ids "${instance_id}" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

  save_state "${public_ip}"

  ok "Instance running: ${instance_id}"
  ok "Public IP: ${public_ip}"
  echo ""
  ok "Services (once healthy):"
  ok "  Gateway:   http://${public_ip}:8080"
  ok "  Paddle:    http://${public_ip}:8081"
  ok "  Tesseract: http://${public_ip}:8082"
  ok "  Surya:     http://${public_ip}:8083"
  ok "  docTR:     http://${public_ip}:8084"
  echo ""
  ok "SSH: ssh -i infra/${KEY_NAME}.pem ec2-user@${public_ip}"
  echo ""
  warn "First boot takes 5-10 min (pulling 5 images + loading models)"
  echo "${public_ip}"
}

# ----------------------------------------------------------
# Generate remote docker-compose.yml (ECR images, no build)
# ----------------------------------------------------------
generate_remote_compose() {
  cat <<YAML
services:
  paddle:
    image: ${ECR_BASE}/ocr-comparison/paddle:${IMAGE_TAG}
    ports:
      - "8081:8080"
    runtime: nvidia
    environment:
      - OCR_DEVICE=gpu
      - OCR_DPI=300
      - OCR_LOG_LEVEL=INFO
      - OCR_MAX_CONCURRENT=2
      - OCR_MODE=ocr
      - FLAGS_use_mkldnn=0
      - FLAGS_enable_pir_in_executor=0
      - FLAGS_enable_pir_api=0
      - FLAGS_fraction_of_cpu_memory_to_use=0.8
      - FLAGS_allocator_strategy=auto_growth
      - FLAGS_call_stack_level=2
      - MKL_NUM_THREADS=1
      - OMP_NUM_THREADS=1
      - OPENBLAS_NUM_THREADS=1
      - PYTHONFAULTHANDLER=1
      - GLOG_v=0
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        limits:
          memory: 12G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/healthz"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 120s
    restart: unless-stopped

  tesseract:
    image: ${ECR_BASE}/ocr-comparison/tesseract:${IMAGE_TAG}
    ports:
      - "8082:8080"
    environment:
      - OCR_TESSERACT_LANG=eng
      - OCR_TESSERACT_DPI=300
      - OCR_LOG_LEVEL=INFO
    deploy:
      resources:
        limits:
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/healthz"]
      interval: 15s
      timeout: 10s
      retries: 3
      start_period: 10s
    restart: unless-stopped

  surya:
    image: ${ECR_BASE}/ocr-comparison/surya:${IMAGE_TAG}
    ports:
      - "8083:8080"
    runtime: nvidia
    environment:
      - OCR_DEVICE=cuda
      - OCR_LOG_LEVEL=INFO
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        limits:
          memory: 8G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/healthz"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 120s
    restart: unless-stopped

  doctr:
    image: ${ECR_BASE}/ocr-comparison/doctr:${IMAGE_TAG}
    ports:
      - "8084:8080"
    runtime: nvidia
    environment:
      - OCR_DEVICE=cuda
      - OCR_LOG_LEVEL=INFO
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        limits:
          memory: 6G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/healthz"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 120s
    restart: unless-stopped

  gateway:
    image: ${ECR_BASE}/ocr-comparison/gateway:${IMAGE_TAG}
    ports:
      - "8080:8080"
    environment:
      - GATEWAY_PADDLE_URL=http://paddle:8080
      - GATEWAY_TESSERACT_URL=http://tesseract:8080
      - GATEWAY_SURYA_URL=http://surya:8080
      - GATEWAY_DOCTR_URL=http://doctr:8080
      - GATEWAY_REQUEST_TIMEOUT=120
    deploy:
      resources:
        limits:
          memory: 512M
    depends_on:
      paddle:
        condition: service_healthy
      tesseract:
        condition: service_healthy
      surya:
        condition: service_healthy
      doctr:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/healthz"]
      interval: 15s
      timeout: 5s
      retries: 3
      start_period: 5s
    restart: unless-stopped
YAML
}

# ----------------------------------------------------------
# Step 8: Smoke test
# ----------------------------------------------------------
smoke_test() {
  local ip="${1:-$(load_state)}"
  if [[ -z "${ip}" ]]; then
    err "No instance IP found. Run deploy first or pass IP as argument."
    exit 1
  fi

  info "Smoke testing against ${ip}..."
  echo ""

  # Wait for services to be healthy
  info "Waiting for services to come up (this may take a few minutes)..."
  local max_wait=600  # 10 min
  local waited=0
  while [[ ${waited} -lt ${max_wait} ]]; do
    local healthy=0
    for port in 8081 8082 8083 8084; do
      if curl -sf "http://${ip}:${port}/healthz" >/dev/null 2>&1; then
        healthy=$((healthy + 1))
      fi
    done
    if [[ ${healthy} -eq 4 ]]; then
      ok "All 4 OCR services healthy"
      break
    fi
    echo "  ${healthy}/4 services healthy... (${waited}s elapsed)"
    sleep 15
    waited=$((waited + 15))
  done

  if [[ ${waited} -ge ${max_wait} ]]; then
    err "Timeout waiting for services. Check logs with:"
    err "  ssh -i infra/${KEY_NAME}.pem ec2-user@${ip} 'cd /opt/ocr && docker compose logs'"
    exit 1
  fi

  # Test each service individually with simple-typed.png
  local test_file="${PROJECT_DIR}/test-fixtures/simple-typed.png"
  if [[ ! -f "${test_file}" ]]; then
    warn "Test fixture not found: ${test_file}"
    warn "Generating fixtures..."
    python3 "${PROJECT_DIR}/test-fixtures/generate.py"
  fi

  echo ""
  info "Testing each service with simple-typed.png..."
  echo ""

  for svc_port in "paddle:8081" "tesseract:8082" "surya:8083" "doctr:8084"; do
    local svc="${svc_port%%:*}"
    local port="${svc_port##*:}"

    echo -n "  ${svc} (port ${port})... "
    local resp
    resp=$(curl -sf -w "\n%{http_code}" \
      -F "file=@${test_file}" \
      "http://${ip}:${port}/api/v1/ocr" 2>&1) || true

    local http_code="${resp##*$'\n'}"
    local body="${resp%$'\n'*}"

    if [[ "${http_code}" == "200" ]]; then
      local pages words latency
      pages=$(echo "${body}" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('pages',[])))" 2>/dev/null || echo "?")
      words=$(echo "${body}" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(sum(len(w) for p in d.get('pages',[]) for b in p.get('blocks',[]) for l in b.get('lines',[]) for w in l.get('words',[])))
" 2>/dev/null || echo "?")
      latency=$(echo "${body}" | python3 -c "import sys,json; print(f\"{json.load(sys.stdin).get('latency_ms',0):.0f}\")" 2>/dev/null || echo "?")
      ok "${pages} page(s), ${words} words, ${latency}ms"
    else
      err "FAILED (HTTP ${http_code})"
    fi
  done

  # Test gateway compare endpoint
  echo ""
  echo -n "  gateway compare... "
  local gw_resp
  gw_resp=$(curl -sf -w "\n%{http_code}" \
    -F "file=@${test_file}" \
    "http://${ip}:8080/api/v1/compare" 2>&1) || true

  local gw_code="${gw_resp##*$'\n'}"
  if [[ "${gw_code}" == "200" ]]; then
    local gw_body="${gw_resp%$'\n'*}"
    local ok_count
    ok_count=$(echo "${gw_body}" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(sum(1 for r in d.get('results',{}).values() if r.get('status')=='ok'))
" 2>/dev/null || echo "?")
    ok "${ok_count}/4 providers returned results"
  else
    err "FAILED (HTTP ${gw_code})"
  fi

  echo ""
  echo "============================================"
  ok "Smoke tests complete!"
  echo ""
  ok "Comparison UI:  http://${ip}:8080"
  ok "SSH:            ssh -i infra/${KEY_NAME}.pem ec2-user@${ip}"
  echo "============================================"
}

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
main() {
  local cmd="${1:-all}"

  info "AWS Profile: ${AWS_PROFILE} | Region: ${AWS_REGION}"
  info "Stack: ${STACK_NAME} | Instance: ${INSTANCE_TYPE}"
  echo ""

  case "${cmd}" in
    push)
      create_ecr_repos
      build_and_push
      ;;
    launch)
      launch_instance
      ;;
    test)
      smoke_test "${2:-}"
      ;;
    all)
      create_ecr_repos
      build_and_push
      launch_instance
      echo ""
      info "Waiting 30s for instance user-data to start..."
      sleep 30
      smoke_test
      ;;
    *)
      err "Unknown command: ${cmd}"
      echo "Usage: $0 [push|launch|test|all]"
      exit 1
      ;;
  esac
}

main "$@"
