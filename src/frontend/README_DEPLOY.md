# LightMem-Ego Frontend Deployment

This document describes how to deploy the LightMem-Ego web frontend as static files behind Nginx.

## Build Output

The frontend is a Vite + React app. Production files are generated into:

```text
online_web/dist/
```

The deployment server only needs the files in `dist/` plus an Nginx configuration that serves them and routes API requests to the backend.

## Build

```bash
cd src/frontend/online_web
npm ci
npm run build
```

If you need to point the frontend at a different backend, set `VITE_API_BASE_URL` before building:

```bash
VITE_API_BASE_URL=https://your-domain.example.com/api npm run build
```

On Windows PowerShell:

```powershell
$env:VITE_API_BASE_URL="https://your-domain.example.com/api"
npm run build
```

## Nginx Static Deployment

Copy the generated files to the web root:

```bash
sudo mkdir -p /var/www/lightmem-ego
sudo rsync -a online_web/dist/ /var/www/lightmem-ego/
```

Install the example Nginx config:

```bash
sudo cp deploy/nginx-online-web.conf.example /etc/nginx/conf.d/lightmem-ego.conf
sudo nginx -t
sudo systemctl reload nginx
```

Update `server_name`, TLS paths, and proxy settings in the Nginx config for your environment.

## HTTPS Requirement

Browser camera, microphone, and WebRTC features require a secure context in production. Use HTTPS for public deployments. `localhost` is the usual browser exception for development.

## Backend Requirements

The backend must allow the frontend origin in its CORS configuration. For example, if the site is served from:

```text
https://lightmem-ego.zjukg.cn
```

then the backend CORS allowlist must include that origin.

The frontend expects the backend to expose the LightMem-Ego stream and query APIs:

```text
POST /stream/start
POST /stream/{session_id}/frame
POST /stream/{session_id}/audio_chunk
POST /stream/{session_id}/live/ingest/start
POST /stream/{session_id}/live/ingest/stop
GET  /stream/{session_id}/status
POST /ask/{session_id}
GET  /query_task/{task_id}
GET  /session/{session_id}/file?path=...
```

## Troubleshooting

- If camera or microphone access is unavailable, check HTTPS and browser permissions.
- If API requests fail, check `VITE_API_BASE_URL`, Nginx proxy routing, backend availability, and backend CORS settings.
- If refreshing a frontend route returns 404, ensure Nginx uses `try_files $uri $uri/ /index.html;`.
- If static assets return 404, confirm that the contents of `online_web/dist/` were copied to the configured web root.
