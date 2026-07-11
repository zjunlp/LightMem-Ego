# online_web 阿里云部署说明

生成日期：2026-06-07

这个压缩包用于把 `online_web` 前端部署到新的阿里云 ECS 服务器。项目是 Vite + React 前端，最终生产环境只需要把 `online_web/dist` 目录交给 Nginx 托管。

## 目录说明

```text
online_web_deploy_20260607/
├── README_DEPLOY.md
├── deploy/
│   └── nginx-online-web.conf.example
└── online_web/
    ├── dist/                 # 已构建好的静态文件，可直接部署
    ├── src/                  # 前端源码
    ├── index.html
    ├── package.json
    ├── package-lock.json
    └── vite.config.js
```

压缩包中不包含 `node_modules`。如果需要在服务器重新构建，请使用 `npm ci` 安装依赖。

## 一、服务器准备

1. 在阿里云安全组放行入站端口：
   - `80/tcp`：HTTP
   - `443/tcp`：HTTPS，推荐必须开启

2. 如果有域名，把域名 A 记录解析到 ECS 公网 IP。

3. 安装 Nginx 和 unzip。

Ubuntu / Debian：

```bash
sudo apt update
sudo apt install -y nginx unzip
```

CentOS / Alibaba Cloud Linux：

```bash
sudo yum install -y nginx unzip
sudo systemctl enable --now nginx
```

## 二、最快部署方式：直接部署 dist

把 zip 上传到服务器，例如上传到 `/tmp/online_web_deploy_20260607.zip`。

```bash
cd /tmp
unzip online_web_deploy_20260607.zip

sudo mkdir -p /var/www/online_web
sudo rsync -a /tmp/online_web_deploy_20260607/online_web/dist/ /var/www/online_web/

sudo cp /tmp/online_web_deploy_20260607/deploy/nginx-online-web.conf.example /etc/nginx/conf.d/online_web.conf
sudo sed -i 's/your-domain.example.com/你的域名/g' /etc/nginx/conf.d/online_web.conf

sudo nginx -t
sudo systemctl reload nginx
```

如果暂时没有域名，可以先把 Nginx 配置中的 `server_name` 改成服务器公网 IP。但摄像头、麦克风、WebRTC 功能在公网访问时需要 HTTPS，只有 `localhost` 例外，所以正式使用建议配置域名和 HTTPS。

## 三、配置 HTTPS

前端会使用摄像头、麦克风和 WebRTC。浏览器要求这些能力运行在安全上下文中，因此公网部署请使用 HTTPS。

如果域名已经解析到 ECS，可以使用 Certbot：

Ubuntu / Debian：

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d 你的域名
```

CentOS / Alibaba Cloud Linux 可使用阿里云 SSL 证书，或按当前系统版本安装 Certbot。证书部署完成后，再执行：

```bash
sudo nginx -t
sudo systemctl reload nginx
```

## 四、接口地址配置

当前前端默认请求后端：

```text
https://api.worldmm.xyz
```

这个地址写在：

```text
online_web/src/api/worldmmApi.js
```

第一行：

```js
const API_BASE_URL = 'https://api.worldmm.xyz'
```

如果新服务器仍然访问这个后端，则不需要修改。若后端地址变了，需要在服务器或本地改这一行，然后重新构建：

```bash
cd /tmp/online_web_deploy_20260607/online_web
npm ci
npm run build

sudo rsync -a dist/ /var/www/online_web/
sudo nginx -t
sudo systemctl reload nginx
```

注意：后端必须允许当前前端域名跨域访问，也就是后端 CORS 需要放行：

```text
https://你的域名
```

如果只是用公网 IP 测试，也需要后端放行对应的 `http://公网IP` 或 `https://公网IP`。

## 五、后端接口依赖

前端会调用以下类型的接口，部署前请确认后端可用：

```text
POST /stream/start
POST /stream/{sessionId}/frame
POST /stream/{sessionId}/audio_chunk
POST /stream/{sessionId}/live/ingest/start
POST /stream/{sessionId}/live/ingest/stop
GET  /stream/{sessionId}/status
POST /ask/{sessionId}
GET  /query_task/{taskId}
GET  /session/{sessionId}/file?path=...
```

WebRTC 模式依赖后端返回可访问的 `whip_url`。该地址也需要满足浏览器跨域和 HTTPS 要求。

## 六、常见问题排查

1. 页面能打开，但摄像头或麦克风不可用：
   - 检查是否使用 HTTPS。
   - 检查浏览器是否允许当前站点访问摄像头和麦克风。

2. 页面请求接口失败：
   - 打开浏览器开发者工具，看 Network 里的请求 URL 是否是预期后端。
   - 检查后端 CORS 是否放行前端域名。
   - 检查阿里云安全组、防火墙、后端服务端口是否开放。

3. 刷新页面出现 404：
   - 确认 Nginx 配置里有 `try_files $uri $uri/ /index.html;`。

4. 静态文件 404：
   - 当前构建默认部署在网站根路径 `/`。
   - 如果要部署到子路径，例如 `https://域名/online/`，需要在 `vite.config.js` 增加 `base: '/online/'` 后重新构建。

## 七、本次构建信息

本压缩包中的 `dist` 已重新构建并验证通过：

```text
Node.js: v26.0.0
npm: 11.12.1
Vite: 6.4.3
```

生产构建命令：

```bash
npm run build
```
