# PowerShell script to test Cloud Run endpoints
# Usage: .\test_cloud_api.ps1 [CLOUD_RUN_URL]

param(
    [string]$CloudRunUrl = "http://localhost:8080"
)

Write-Host "===================================" -ForegroundColor Cyan
Write-Host "Testing Cloud Run Analysis Pipeline" -ForegroundColor Cyan
Write-Host "URL: $CloudRunUrl" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan

Write-Host "`n1. Health Check (GET /)" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Yellow
Invoke-WebRequest -Uri "$CloudRunUrl/" -Method GET | Select-Object -ExpandProperty Content
Write-Host ""

Write-Host "`n2. Test Mode - Default Fortune 100, 50 records (GET)" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Yellow
Invoke-WebRequest -Uri "$CloudRunUrl/run?mode=test" -Method GET | Select-Object -ExpandProperty Content
Write-Host ""

Write-Host "`n3. Specific Companies via GET" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Yellow
Invoke-WebRequest -Uri "$CloudRunUrl/run?companies=AAPL,MSFT,GOOGL&mode=test" -Method GET | Select-Object -ExpandProperty Content
Write-Host ""

Write-Host "`n4. Date Range via GET" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Yellow
Invoke-WebRequest -Uri "$CloudRunUrl/run?start_date=2024-01-01&end_date=2024-12-31&mode=test" -Method GET | Select-Object -ExpandProperty Content
Write-Host ""

Write-Host "`n5. Full Configuration via POST (JSON)" -ForegroundColor Yellow
Write-Host "-----------------------------------" -ForegroundColor Yellow
$body = @{
    companies = "AAPL,MSFT,GOOGL,AMZN,NVDA"
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    mode = "test"
    limit = 100
} | ConvertTo-Json

Invoke-WebRequest -Uri "$CloudRunUrl/run" -Method POST -ContentType "application/json" -Body $body | Select-Object -ExpandProperty Content
Write-Host ""

Write-Host "`n6. Full Mode - Fortune 100 (POST)" -ForegroundColor Red
Write-Host "-----------------------------------" -ForegroundColor Red
Write-Host "WARNING: This will process ALL records for Fortune 100 companies" -ForegroundColor Red
Write-Host "Uncomment the code below to run:" -ForegroundColor Red
Write-Host ""
Write-Host "# `$body = @{ mode = 'full' } | ConvertTo-Json" -ForegroundColor Gray
Write-Host "# Invoke-WebRequest -Uri '$CloudRunUrl/run' -Method POST -ContentType 'application/json' -Body `$body" -ForegroundColor Gray
Write-Host ""
