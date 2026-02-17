# Run full pipeline: build features -> train models -> generate map
# Usage: .\run_all.ps1

$ErrorActionPreference = "Stop"
$roads = (Get-ChildItem "data\raw\*.shp").FullName
if (-not $roads) { throw "No .shp file found in data\raw\" }

Write-Host "=== 1. Build features ===" -ForegroundColor Cyan
python -m src.features.build `
  --collisions data/raw/Traffic_Collision_Data.csv `
  --roads $roads `
  --construction "data/raw/Upcoming_Construction_facilities%2C_culverts%2C_parks%2C_bridges.csv" `
  --buffer-m 12 `
  --out data/processed/segment_year_features.parquet

Write-Host "`n=== 2. Train models ===" -ForegroundColor Cyan
python -m src.models.train --config configs/logreg.yaml
python -m src.models.train --config configs/lightgbm.yaml

Write-Host "`n=== 3. Generate map ===" -ForegroundColor Cyan
python -m src.viz.map_results `
  --features data/processed/segment_year_features.parquet `
  --models-dir models `
  --roads $roads `
  --out reports/risk_map.html

Write-Host "`nDone. Open reports\risk_map.html in a browser." -ForegroundColor Green
