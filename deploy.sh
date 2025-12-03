#!/bin/bash
# ===========================================
# Cloud Run Deployment Script
# ===========================================

# Configuration - CHANGE THESE VALUES
PROJECT_ID="your-gcp-project-id"
SERVICE_NAME="video-talent-analyzer"
REGION="us-central1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🚀 Deploying Video Talent Analyzer to Cloud Run${NC}"
echo "================================================"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI not found. Please install it first.${NC}"
    exit 1
fi

# Set project
echo -e "\n${GREEN}1. Setting GCP project...${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "\n${GREEN}2. Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Build and push container
echo -e "\n${GREEN}3. Building and pushing container...${NC}"
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

# Deploy to Cloud Run
echo -e "\n${GREEN}4. Deploying to Cloud Run...${NC}"
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 600 \
    --concurrency 10 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars "OPENAI_API_KEY=$OPENAI_API_KEY,AZURE_SPEECH_KEY=$AZURE_SPEECH_KEY,AZURE_SERVICE_REGION=$AZURE_SERVICE_REGION"

echo -e "\n${GREEN}✅ Deployment complete!${NC}"
echo -e "Your service URL will be displayed above."

