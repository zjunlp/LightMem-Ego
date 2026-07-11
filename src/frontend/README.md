# LightMem-Ego Frontend

This directory contains the deployed web frontend source for LightMem-Ego.

## Layout

```text
online_web/          # Vite + React web application
deploy/             # Example nginx deployment configuration
README_DEPLOY.md    # Server deployment notes
```

## Development

```bash
cd online_web
npm install
npm run dev
```

## Build

```bash
cd online_web
npm run build
```

The production build is generated in `online_web/dist/`. It is intentionally not committed because it can be recreated from the source code and dependency lockfile.
