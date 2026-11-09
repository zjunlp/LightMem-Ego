FROM node:22-bookworm-slim AS build

WORKDIR /app/frontend/online_web
COPY src/frontend/online_web/package.json src/frontend/online_web/package-lock.json ./
COPY src/frontend/online_web/.npmrc ./
RUN npm ci
COPY src/frontend/online_web/ ./
COPY src/frontend/history_demo/ ../history_demo/
ARG VITE_API_BASE_URL=/api
ARG VITE_DEMO_API_BASE_URL=
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
ENV VITE_DEMO_API_BASE_URL=${VITE_DEMO_API_BASE_URL}
RUN npm run build

FROM nginx:1.27-alpine
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/frontend/online_web/dist /usr/share/nginx/html
EXPOSE 80
