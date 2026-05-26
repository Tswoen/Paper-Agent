import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const runtimeConfigPath = path.resolve(__dirname, '..', '.runtime', 'backend.json')
const defaultBackendTarget = 'http://127.0.0.1:8000'

function resolveBackendTarget() {
  if (process.env.VITE_BACKEND_URL) {
    return process.env.VITE_BACKEND_URL
  }

  try {
    const runtimeConfig = JSON.parse(fs.readFileSync(runtimeConfigPath, 'utf-8'))
    if (typeof runtimeConfig.url === 'string' && runtimeConfig.url.trim()) {
      return runtimeConfig.url
    }
  } catch {
    // The backend may not have been started yet; fall back to the default port.
  }

  return defaultBackendTarget
}

const backendTarget = resolveBackendTarget()

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/api': {
        target: backendTarget,
        changeOrigin: true
      },
      '/knowledge': {
        target: backendTarget,
        changeOrigin: true
      },
      '/send_input': {
        target: backendTarget,
        changeOrigin: true
      }
    }
  }
})
