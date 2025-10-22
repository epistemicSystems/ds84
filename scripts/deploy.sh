#!/bin/bash
# Deployment script for Realtor AI Copilot

set -e  # Exit on error
set -o pipefail  # Exit on pipe failure

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="realtor-ai-copilot"
DOCKER_IMAGE="$APP_NAME"
ECR_REPOSITORY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${APP_NAME}"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed"
        exit 1
    fi

    # Check if required environment variables are set
    if [[ -z "$AWS_REGION" ]]; then
        log_error "AWS_REGION environment variable is not set"
        exit 1
    fi

    if [[ -z "$AWS_ACCOUNT_ID" ]]; then
        log_error "AWS_ACCOUNT_ID environment variable is not set"
        exit 1
    fi

    log_info "All requirements met"
}

build_image() {
    log_info "Building Docker image..."

    docker build -t "$DOCKER_IMAGE:latest" -f Dockerfile .

    if [ $? -eq 0 ]; then
        log_info "Docker image built successfully"
    else
        log_error "Docker build failed"
        exit 1
    fi
}

test_image() {
    log_info "Testing Docker image..."

    # Run a quick health check
    docker run --rm -d --name "$APP_NAME-test" -p 8001:8000 "$DOCKER_IMAGE:latest"
    sleep 10

    # Check health endpoint
    if curl -f http://localhost:8001/health > /dev/null 2>&1; then
        log_info "Health check passed"
        docker stop "$APP_NAME-test"
    else
        log_error "Health check failed"
        docker stop "$APP_NAME-test"
        exit 1
    fi
}

push_to_ecr() {
    log_info "Pushing image to ECR..."

    # Login to ECR
    aws ecr get-login-password --region "$AWS_REGION" | \
        docker login --username AWS --password-stdin "$ECR_REPOSITORY"

    # Tag image
    docker tag "$DOCKER_IMAGE:latest" "$ECR_REPOSITORY:latest"
    docker tag "$DOCKER_IMAGE:latest" "$ECR_REPOSITORY:$(git rev-parse --short HEAD)"

    # Push images
    docker push "$ECR_REPOSITORY:latest"
    docker push "$ECR_REPOSITORY:$(git rev-parse --short HEAD)"

    log_info "Images pushed to ECR successfully"
}

deploy_to_ecs() {
    log_info "Deploying to ECS..."

    CLUSTER_NAME="${1:-realtor-ai-production}"
    SERVICE_NAME="${2:-realtor-ai-api}"

    # Update ECS service to force new deployment
    aws ecs update-service \
        --cluster "$CLUSTER_NAME" \
        --service "$SERVICE_NAME" \
        --force-new-deployment \
        --region "$AWS_REGION"

    if [ $? -eq 0 ]; then
        log_info "ECS service update initiated"
    else
        log_error "ECS service update failed"
        exit 1
    fi

    # Wait for deployment to complete
    log_info "Waiting for deployment to complete..."
    aws ecs wait services-stable \
        --cluster "$CLUSTER_NAME" \
        --services "$SERVICE_NAME" \
        --region "$AWS_REGION"

    if [ $? -eq 0 ]; then
        log_info "Deployment completed successfully"
    else
        log_error "Deployment failed or timed out"
        exit 1
    fi
}

rollback() {
    log_warn "Rolling back deployment..."

    CLUSTER_NAME="${1:-realtor-ai-production}"
    SERVICE_NAME="${2:-realtor-ai-api}"

    # Get previous task definition
    CURRENT_TASK_DEF=$(aws ecs describe-services \
        --cluster "$CLUSTER_NAME" \
        --services "$SERVICE_NAME" \
        --region "$AWS_REGION" \
        --query 'services[0].taskDefinition' \
        --output text)

    TASK_FAMILY=$(echo "$CURRENT_TASK_DEF" | cut -d':' -f6 | cut -d'/' -f2)
    CURRENT_REVISION=$(echo "$CURRENT_TASK_DEF" | cut -d':' -f7)
    PREVIOUS_REVISION=$((CURRENT_REVISION - 1))

    if [ "$PREVIOUS_REVISION" -lt 1 ]; then
        log_error "No previous revision to rollback to"
        exit 1
    fi

    # Update service with previous task definition
    aws ecs update-service \
        --cluster "$CLUSTER_NAME" \
        --service "$SERVICE_NAME" \
        --task-definition "$TASK_FAMILY:$PREVIOUS_REVISION" \
        --force-new-deployment \
        --region "$AWS_REGION"

    if [ $? -eq 0 ]; then
        log_info "Rollback initiated"
    else
        log_error "Rollback failed"
        exit 1
    fi
}

# Main deployment flow
main() {
    log_info "Starting deployment for $APP_NAME"

    # Parse command line arguments
    COMMAND="${1:-deploy}"
    ENVIRONMENT="${2:-production}"

    case "$COMMAND" in
        build)
            check_requirements
            build_image
            ;;
        test)
            check_requirements
            build_image
            test_image
            ;;
        deploy)
            check_requirements
            build_image
            test_image
            push_to_ecr
            deploy_to_ecs "realtor-ai-$ENVIRONMENT" "realtor-ai-api"
            ;;
        rollback)
            check_requirements
            rollback "realtor-ai-$ENVIRONMENT" "realtor-ai-api"
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            echo "Usage: $0 {build|test|deploy|rollback} [environment]"
            echo "  build       - Build Docker image only"
            echo "  test        - Build and test Docker image"
            echo "  deploy      - Full deployment (build, test, push, deploy)"
            echo "  rollback    - Rollback to previous version"
            echo ""
            echo "  environment - staging or production (default: production)"
            exit 1
            ;;
    esac

    log_info "Operation completed successfully"
}

# Run main function
main "$@"
