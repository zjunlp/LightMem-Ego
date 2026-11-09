import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'

export default defineConfig({
  plugins: [react()],
  build: {
    target: 'es2022',
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        memories: fileURLToPath(new URL('./memories/index.html', import.meta.url)),
        history: fileURLToPath(new URL('./history-demo/index.html', import.meta.url))
      },
      output: {
        manualChunks(id) {
          if (/node_modules\/(cytoscape-cose-bilkent|cose-base|layout-base)\//.test(id)) return 'memory-graph-layout'
          if (/node_modules\/(cytoscape|react-cytoscapejs)\//.test(id)) return 'memory-graph-core'
          return undefined
        }
      }
    }
  },
  server: {
    host: '0.0.0.0',
    fs: { allow: [fileURLToPath(new URL('../', import.meta.url))] },
    proxy: { '/api': { target: process.env.LIGHTMEM_BACKEND_URL || 'http://127.0.0.1:8000', changeOrigin: true, rewrite: path => path.replace(/^\/api/, '') } }
  }
})
