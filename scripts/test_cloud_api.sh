#!/bin/bash
# Helper script to test Cloud Run endpoints
# Usage: ./test_cloud_api.sh [CLOUD_RUN_URL]

CLOUD_RUN_URL="${1:-http://localhost:8080}"

echo "==================================="
echo "Testing Cloud Run Analysis Pipeline"
echo "URL: $CLOUD_RUN_URL"
echo "==================================="

echo ""
echo "1. Health Check (GET /)"
echo "-----------------------------------"
curl -s "$CLOUD_RUN_URL/"
echo ""

echo ""
echo "2. Test Mode - Default Fortune 100, 50 records (GET)"
echo "-----------------------------------"
curl -s "$CLOUD_RUN_URL/run?mode=test"
echo ""

echo ""
echo "3. Specific Companies via GET"
echo "-----------------------------------"
curl -s "$CLOUD_RUN_URL/run?companies=AAPL,MSFT,GOOGL&mode=test"
echo ""

echo ""
echo "4. Date Range via GET"
echo "-----------------------------------"
curl -s "$CLOUD_RUN_URL/run?start_date=2024-01-01&end_date=2024-12-31&mode=test"
echo ""

echo ""
echo "5. Full Configuration via POST (JSON)"
echo "-----------------------------------"
curl -s -X POST "$CLOUD_RUN_URL/run" \
  -H "Content-Type: application/json" \
  -d '{
    "companies": "AAPL,MSFT,GOOGL,AMZN,NVDA",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "mode": "test",
    "limit": 100
  }'
echo ""

echo ""
echo "6. Full Mode - Fortune 100 (POST)"
echo "-----------------------------------"
echo "WARNING: This will process ALL records for Fortune 100 companies"
echo "Uncomment the curl command below to run:"
echo ""
echo "# curl -s -X POST \"$CLOUD_RUN_URL/run\" \\"
echo "#   -H \"Content-Type: application/json\" \\"
echo "#   -d '{\"mode\": \"full\"}'"
echo ""
