# Week 7: Production Infrastructure

## Overview

Week 7 establishes production-ready infrastructure with Docker containerization, CI/CD pipelines, comprehensive health checks, structured logging, and deployment automation. This ensures the Realtor AI Copilot can be deployed, monitored, and maintained in production environments.

---

## Table of Contents

1. [Docker Containerization](#docker-containerization)
2. [Docker Compose for Development](#docker-compose-for-development)
3. [CI/CD Pipeline](#cicd-pipeline)
4. [Health Checks & Readiness Probes](#health-checks--readiness-probes)
5. [Structured Logging](#structured-logging)
6. [Monitoring & Metrics](#monitoring--metrics)
7. [Deployment](#deployment)
8. [Environment Configuration](#environment-configuration)
9. [Development Workflow](#development-workflow)
10. [Production Deployment Guide](#production-deployment-guide)

---

## Docker Containerization

### Multi-Stage Dockerfile

The production Dockerfile uses multi-stage builds to minimize image size and improve security:

**Stage 1: Builder**
- Installs system dependencies
- Creates virtual environment
- Installs Python dependencies

**Stage 2: Runtime**
- Minimal production image
- Copies only virtual environment
- Runs as non-root user for security
- Includes health check

### Key Features

- **Minimal Base Image**: python:3.11-slim
- **Non-Root User**: Runs as `appuser` for security
- **Multi-Worker**: 4 Uvicorn workers by default
- **Health Check**: Built-in health check for container orchestrators
- **Size Optimized**: Uses multi-stage build to reduce image size

### Building the Image

```bash
# Build locally
docker build -t realtor-ai-copilot:latest .

# Build with cache
docker build --cache-from realtor-ai-copilot:latest -t realtor-ai-copilot:latest .

# Build for specific platform
docker build --platform linux/amd64 -t realtor-ai-copilot:latest .
```

### Running the Container

```bash
# Run with environment variables
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e DATABASE_URL=postgresql://... \
  --name realtor-ai-api \
  realtor-ai-copilot:latest

# Run with environment file
docker run -d \
  -p 8000:8000 \
  --env-file .env.production \
  --name realtor-ai-api \
  realtor-ai-copilot:latest
```

---

## Docker Compose for Development

### Services

The `docker-compose.yml` includes:

1. **db**: PostgreSQL with pgvector extension
2. **redis**: Redis for caching
3. **app**: Realtor AI Copilot API
4. **prometheus**: Metrics collection (optional, with `monitoring` profile)
5. **grafana**: Metrics visualization (optional, with `monitoring` profile)
6. **nginx**: Reverse proxy (optional, with `production` profile)

### Starting Services

```bash
# Start core services (db, redis, app)
docker-compose up -d

# Start with monitoring
docker-compose --profile monitoring up -d

# Start with production setup
docker-compose --profile production up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Development Mode

The app service mounts source code for hot reloading:

```yaml
volumes:
  - ./app:/app/app  # Hot reload enabled
  - ./workflows:/app/workflows
  - ./prompts:/app/prompts
```

Make changes to Python files and the server automatically reloads.

### Service URLs

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379
- **Prometheus** (optional): http://localhost:9090
- **Grafana** (optional): http://localhost:3000 (admin/admin)
- **Nginx** (optional): http://localhost:80

---

## CI/CD Pipeline

### GitHub Actions Workflow

The CI/CD pipeline (`.github/workflows/ci-cd.yml`) includes 7 jobs:

#### 1. Lint (Code Quality)
- Black (code formatting)
- isort (import sorting)
- flake8 (linting)
- pylint (static analysis)

#### 2. Security Scan
- Safety (dependency vulnerabilities)
- Bandit (security linting)

#### 3. Unit Tests
- Pytest with coverage
- PostgreSQL and Redis services
- Coverage upload to Codecov

#### 4. Build Docker Image
- Docker Buildx with caching
- Multi-platform support

#### 5. Integration Tests
- Tests with real services
- Docker Compose integration

#### 6. Deploy to Staging
- Triggers on `develop` branch
- Pushes to Amazon ECR
- Deploys to ECS staging cluster

#### 7. Deploy to Production
- Triggers on `main` branch (with manual approval)
- Pushes to Amazon ECR
- Deploys to ECS production cluster
- Creates GitHub release

### Branch Strategy

```
main (production)
  ├── develop (staging)
  └── feature/* (development)
```

### Running CI Locally

```bash
# Lint
make ci-lint

# Test
make ci-test

# Build
make ci-build
```

### Required Secrets

Configure these in GitHub repository settings:

- `OPENAI_API_KEY`: OpenAI API key
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_ACCOUNT_ID`: AWS account ID

---

## Health Checks & Readiness Probes

### Endpoints

#### `/health` - Basic Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "version": "0.1.0",
  "environment": "production"
}
```

**Use Case**: Load balancer health checks, simple uptime monitoring

#### `/health/liveness` - Liveness Probe
```bash
curl http://localhost:8000/health/liveness
```

Response:
```json
{
  "status": "alive",
  "timestamp": "2025-10-22T10:30:00Z"
}
```

**Use Case**: Kubernetes liveness probe, container restart detection

#### `/health/readiness` - Readiness Probe
```bash
curl http://localhost:8000/health/readiness
```

Response:
```json
{
  "status": "ready",
  "checks": {
    "app": "ok",
    "workflows": "ok",
    "cache": "ok"
  },
  "timestamp": "2025-10-22T10:30:00Z"
}
```

**Use Case**: Kubernetes readiness probe, traffic routing decisions

Returns 503 if not ready to accept traffic.

#### `/health/detailed` - Detailed Health
```bash
curl http://localhost:8000/health/detailed
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-22T10:30:00Z",
  "version": "0.1.0",
  "environment": "production",
  "services": {
    "workflow_engine": {
      "status": "ok",
      "workflows_loaded": 3,
      "workflow_ids": ["property_search", "..."]
    },
    "cache": {
      "status": "ok",
      "stats": { "hit_rate": 0.85, ... }
    },
    "ab_testing": {
      "status": "ok",
      "active_tests": 2
    },
    "context_manager": {
      "status": "ok",
      "active_sessions": 15
    }
  }
}
```

**Use Case**: Monitoring dashboards, debugging, status pages

### Docker Health Check

Dockerfile includes built-in health check:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

### Kubernetes Probes

```yaml
livenessProbe:
  httpGet:
    path: /health/liveness
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health/readiness
    port: 8000
  initialDelaySeconds: 20
  periodSeconds: 5
```

---

## Structured Logging

### JSON Logging

Production uses structured JSON logging for easy parsing by log aggregation tools (CloudWatch, ELK, Datadog).

### Usage

```python
from app.services.logging_service import logging_service

# Basic logging
logging_service.info("User login successful", user_id="user_123")
logging_service.error("Payment failed", exc_info=True, user_id="user_123")

# Request logging
logging_service.log_request(
    method="POST",
    path="/api/property-search",
    status_code=200,
    duration_ms=245.3,
    user_id="user_123",
    request_id="req_abc"
)

# Workflow logging
logging_service.log_workflow_execution(
    workflow_id="property_search",
    execution_id="exec_123",
    status="completed",
    duration_ms=1250.5,
    user_id="user_123"
)

# Security logging
logging_service.log_security_event(
    event_type="rate_limit_exceeded",
    severity="high",
    description="User exceeded rate limit",
    user_id="user_123",
    ip_address="192.168.1.1"
)
```

### Log Format

```json
{
  "timestamp": "2025-10-22T10:30:00.123Z",
  "level": "INFO",
  "logger": "realtor-ai-copilot",
  "message": "Workflow execution completed",
  "module": "workflow_engine",
  "function": "execute_workflow",
  "line": 123,
  "user_id": "user_123",
  "request_id": "req_abc",
  "extra": {
    "workflow_id": "property_search",
    "duration_ms": 1250.5
  }
}
```

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical errors requiring immediate attention

### Log Rotation

Logs automatically rotate when they reach 100MB (configurable):
- Max size: 100MB
- Backup count: 10 files
- Total max size: 1GB

### Viewing Logs

```bash
# Docker logs
docker-compose logs -f app

# Local log file
tail -f logs/app.log

# Pretty-print JSON logs
tail -f logs/app.log | jq '.'

# Filter by level
tail -f logs/app.log | jq 'select(.level == "ERROR")'

# Filter by user
tail -f logs/app.log | jq 'select(.user_id == "user_123")'
```

---

## Monitoring & Metrics

### Prometheus

Prometheus configuration is in `monitoring/prometheus.yml`.

**Scrape Targets**:
- API metrics (port 8000/metrics)
- PostgreSQL metrics (postgres_exporter)
- Redis metrics (redis_exporter)
- System metrics (node_exporter)

**Accessing Prometheus**:
```bash
# Start with monitoring profile
docker-compose --profile monitoring up -d

# Access UI
open http://localhost:9090
```

### Grafana

Grafana dashboards for visualization.

**Accessing Grafana**:
```bash
# Access UI
open http://localhost:3000

# Login
Username: admin
Password: admin
```

**Pre-configured Dashboards**:
- API Performance (requests, latency, errors)
- Cache Metrics (hit rate, size, evictions)
- Database Metrics (connections, queries, slow queries)
- System Metrics (CPU, memory, disk)

### Key Metrics

**API Metrics**:
- Request rate (requests/second)
- Response time (p50, p95, p99)
- Error rate
- Active connections

**Cache Metrics**:
- Hit rate
- Miss rate
- Eviction rate
- Memory usage

**Workflow Metrics**:
- Execution count
- Success rate
- Average duration
- Failed executions

**A/B Test Metrics**:
- Active tests
- Variant assignments
- Statistical significance

---

## Deployment

### Deployment Scripts

#### `scripts/deploy.sh`

Automated deployment script with commands:

```bash
# Build only
./scripts/deploy.sh build

# Build and test
./scripts/deploy.sh test

# Full deployment to production
./scripts/deploy.sh deploy production

# Deploy to staging
./scripts/deploy.sh deploy staging

# Rollback
./scripts/deploy.sh rollback production
```

### Makefile Commands

```bash
# Development
make install          # Install dependencies
make dev              # Run development server
make test             # Run tests
make lint             # Lint code
make format           # Format code

# Docker
make docker-build     # Build Docker image
make docker-up        # Start services
make docker-down      # Stop services
make docker-logs      # View logs

# Deployment
make deploy-staging   # Deploy to staging
make deploy-prod      # Deploy to production
make rollback         # Rollback deployment
```

### AWS ECS Deployment

The CI/CD pipeline automatically deploys to AWS ECS:

**Staging**: Deploys on merge to `develop`
**Production**: Deploys on merge to `main` (requires manual approval)

**Architecture**:
- ECS Fargate for serverless containers
- Application Load Balancer
- RDS PostgreSQL with pgvector
- ElastiCache Redis
- CloudWatch for logs and metrics

---

## Environment Configuration

### `.env.production`

Production environment template:

```env
# Application
APP_NAME=Realtor AI Copilot
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Database
DATABASE_URL=postgresql://user:password@rds-instance.region.rds.amazonaws.com:5432/realtor_ai

# Redis
REDIS_URL=redis://redis-cluster.region.cache.amazonaws.com:6379/0

# OpenAI
OPENAI_API_KEY=${OPENAI_API_KEY}

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=120
RATE_LIMIT_REQUESTS_PER_HOUR=5000

# Caching
CACHE_EMBEDDING_SIZE=50000
CACHE_PROMPT_SIZE=5000
```

### Secrets Management

**Development**: Use `.env` files (not committed)
**Production**: Use AWS Secrets Manager or Parameter Store

```bash
# Retrieve secret from AWS Secrets Manager
aws secretsmanager get-secret-value \
    --secret-id realtor-ai/openai-api-key \
    --query SecretString \
    --output text
```

---

## Development Workflow

### Local Development

```bash
# 1. Clone repository
git clone https://github.com/yourusername/realtor-ai-copilot.git
cd realtor-ai-copilot

# 2. Install dependencies
make install

# 3. Start services
docker-compose up -d db redis

# 4. Run development server
make dev

# 5. Access API
open http://localhost:8000/docs
```

### Making Changes

```bash
# 1. Create feature branch
git checkout -b feature/your-feature

# 2. Make changes
# ... edit files ...

# 3. Format code
make format

# 4. Lint code
make lint

# 5. Run tests
make test

# 6. Commit changes
git add .
git commit -m "Add your feature"

# 7. Push and create PR
git push origin feature/your-feature
```

### Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test
pytest tests/test_week7.py -v

# Run in watch mode
make test-watch
```

---

## Production Deployment Guide

### Prerequisites

1. **AWS Account**: With ECS, RDS, ElastiCache access
2. **Docker Hub/ECR**: Container registry
3. **Domain**: For production API
4. **SSL Certificate**: For HTTPS

### Step-by-Step Deployment

#### 1. Infrastructure Setup

```bash
# Create VPC, subnets, security groups
# (Use Terraform or CloudFormation)

# Create RDS PostgreSQL instance with pgvector
# Create ElastiCache Redis cluster
# Create ECS cluster
# Create ECR repository
```

#### 2. Configure Secrets

```bash
# Store secrets in AWS Secrets Manager
aws secretsmanager create-secret \
    --name realtor-ai/openai-api-key \
    --secret-string "your-openai-key"

aws secretsmanager create-secret \
    --name realtor-ai/database-url \
    --secret-string "postgresql://..."
```

#### 3. Build and Push Image

```bash
# Set environment variables
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=123456789012

# Build and push
./scripts/deploy.sh deploy production
```

#### 4. Monitor Deployment

```bash
# Watch ECS service
aws ecs describe-services \
    --cluster realtor-ai-production \
    --services realtor-ai-api

# View logs
aws logs tail /ecs/realtor-ai-api --follow
```

#### 5. Verify Deployment

```bash
# Check health
curl https://api.yourdomain.com/health

# Check detailed health
curl https://api.yourdomain.com/health/detailed

# Test API
curl https://api.yourdomain.com/api/property-search \
    -H "X-API-Key: your-key" \
    -H "Content-Type: application/json" \
    -d '{"query": "3-bedroom house"}'
```

### Rollback Procedure

If deployment fails:

```bash
# Automatic rollback (if health checks fail)
# ECS will automatically rollback

# Manual rollback
./scripts/deploy.sh rollback production

# Or via AWS CLI
aws ecs update-service \
    --cluster realtor-ai-production \
    --service realtor-ai-api \
    --task-definition realtor-ai-api:PREVIOUS_REVISION
```

---

## Monitoring Production

### CloudWatch Dashboards

Create dashboards for:
- API request rate and latency
- Error rate and types
- Cache hit rate
- Database connections
- ECS task health

### Alarms

Set up CloudWatch alarms for:
- High error rate (> 5%)
- Slow response time (p95 > 2s)
- High CPU usage (> 80%)
- High memory usage (> 80%)
- Failed health checks

### Log Analysis

```bash
# View recent errors
aws logs filter-pattern '{ $.level = "ERROR" }' \
    --log-group-name /ecs/realtor-ai-api \
    --start-time $(date -d '1 hour ago' +%s)000

# View slow requests
aws logs filter-pattern '{ $.duration_ms > 2000 }' \
    --log-group-name /ecs/realtor-ai-api
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs realtor-ai-api

# Check health
docker inspect --format='{{json .State.Health}}' realtor-ai-api | jq '.'

# Enter container
docker exec -it realtor-ai-api /bin/bash
```

### Database Connection Issues

```bash
# Test connection from container
docker exec realtor-ai-api psql $DATABASE_URL -c "SELECT 1"

# Check security groups
aws ec2 describe-security-groups --group-ids sg-xxxxx
```

### High Memory Usage

```bash
# Check cache sizes
curl http://localhost:8000/health/detailed | jq '.services.cache.stats'

# Adjust cache sizes in environment variables
CACHE_EMBEDDING_SIZE=10000  # Reduce from 50000
```

### Deployment Failures

```bash
# Check ECS events
aws ecs describe-services \
    --cluster realtor-ai-production \
    --services realtor-ai-api \
    | jq '.services[0].events'

# Check task definition
aws ecs describe-task-definition \
    --task-definition realtor-ai-api
```

---

## Best Practices

### Security
- Always use HTTPS in production
- Rotate API keys regularly
- Use AWS Secrets Manager for secrets
- Run containers as non-root user
- Keep dependencies updated

### Performance
- Use connection pooling for database
- Enable Redis caching
- Monitor and tune worker count
- Use CDN for static assets

### Reliability
- Set up health checks
- Configure auto-scaling
- Use multiple availability zones
- Regular backups of database

### Cost Optimization
- Use Fargate Spot for non-critical workloads
- Right-size ECS tasks
- Enable RDS performance insights
- Monitor and optimize cache hit rates

---

## Summary

Week 7 delivers production-ready infrastructure:

✅ **Docker Containerization**: Multi-stage, secure, optimized
✅ **Docker Compose**: Full local development environment
✅ **CI/CD Pipeline**: Automated testing and deployment
✅ **Health Checks**: Comprehensive readiness and liveness probes
✅ **Structured Logging**: JSON logging for easy analysis
✅ **Monitoring**: Prometheus and Grafana integration
✅ **Deployment**: Automated scripts and Makefile
✅ **AWS Integration**: ECS, RDS, ElastiCache ready

The system is now production-ready with enterprise-grade infrastructure, automated deployment, and comprehensive monitoring. 🚀
