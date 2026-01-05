# AWS Deployment Plan

## Architecture Overview

**Recommended**: Lambda + API Gateway + S3 (cost-effective for low-medium traffic)

**Alternative**: SageMaker Endpoint (better for high traffic, more expensive)

## Option 1: Lambda + API Gateway (Recommended)

### Components
- **Lambda Function**: Model inference (Python runtime, ~500MB)
- **API Gateway**: REST API endpoint
- **S3**: Model storage (MLflow artifacts)
- **CloudWatch**: Logging and monitoring

### Architecture
```
Client → API Gateway → Lambda → S3 (model) → Response
```

### Implementation Steps
1. Package model + dependencies in Lambda layer
2. Store models in S3 (MLflow artifacts)
3. Lambda loads model from S3 on cold start
4. API Gateway routes requests to Lambda
5. CloudWatch logs for monitoring

### Cost Estimate
- Lambda: $0.20 per 1M requests (first 1M free)
- API Gateway: $3.50 per 1M requests (first 1M free)
- S3: $0.023 per GB/month
- **Total**: ~$5-10/month for low traffic

## Option 2: SageMaker Endpoint

### Components
- **SageMaker Endpoint**: Managed model serving
- **SageMaker Model**: Containerized model
- **S3**: Model storage
- **CloudWatch**: Monitoring

### Architecture
```
Client → API Gateway → SageMaker Endpoint → Response
```

### Implementation Steps
1. Build Docker container with model + dependencies
2. Push container to ECR
3. Create SageMaker model from S3 artifacts
4. Deploy endpoint (instance type: ml.t2.medium)
5. Create API Gateway integration

### Cost Estimate
- SageMaker Endpoint: ~$30-50/month (ml.t2.medium, always-on)
- API Gateway: $3.50 per 1M requests
- **Total**: ~$35-55/month

## Model Versioning

- **MLflow + S3**: Store model artifacts in S3 with MLflow tracking
- **Model Registry**: Use MLflow model registry or S3 versioning
- **A/B Testing**: Deploy multiple endpoints, route traffic

## Infrastructure as Code

**Terraform** (recommended) or **CloudFormation**:
- Lambda function + API Gateway
- S3 bucket for models
- IAM roles and policies
- CloudWatch alarms

## Security

- API Gateway: API keys or Cognito authentication
- Lambda: VPC (optional) for private resources
- S3: Bucket policies, encryption at rest
- IAM: Least privilege access

## Monitoring

- **CloudWatch Metrics**: Latency, errors, invocations
- **CloudWatch Logs**: Request/response logging
- **X-Ray**: Distributed tracing (optional)
- **Alarms**: Error rate, latency thresholds

## Deployment Pipeline

1. **CI/CD**: GitHub Actions or CodePipeline
2. **Build**: Package model + dependencies
3. **Test**: Run integration tests
4. **Deploy**: Update Lambda/SageMaker endpoint
5. **Monitor**: CloudWatch dashboards

## Recommendation

**Start with Lambda + API Gateway**:
- Lower cost for portfolio project
- Sufficient for demonstration
- Easy to scale if needed
- Can migrate to SageMaker later if traffic increases

