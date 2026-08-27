# Grafana Dashboard (Real-time) for Suspicious-Activity-Detection

## Key point: this is *not* static data
The dashboard panels query Prometheus using live PromQL. Prometheus continuously scrapes `GET /metrics` from your app, and Grafana refreshes the panels periodically.

This dashboard JSON sets:
- `refresh`: **10s**
- Gauges/stat panels show the **latest** scraped value.

So it updates in real time as long as Prometheus has fresh samples.

## Steps to import
1. Run your app and ensure metrics are exposed at:
   - `http://<host>:8000/metrics` (default: `PORT=8000`)
2. In Grafana: **Dashboards → New → Import**
3. Upload `grafana/sad-dashboard.json`
4. When prompted, select your Prometheus datasource for the variable `DS_PROMETHEUS`.

## Verify it’s updating
- Open one Stat panel (e.g. **Camera FPS**).
- If the app is running and Prometheus is scraping, the value should change within ~10–20 seconds.

## Common troubleshooting
- **No data in panels**:
  - Check Prometheus is scraping `.../metrics`.
  - In Prometheus UI, run: `sad_camera_fps` and ensure it returns a value.
- **Variables/datasource**:
  - Ensure the Prometheus datasource you select matches what Grafana should query.

