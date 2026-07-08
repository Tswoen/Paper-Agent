import fs from 'fs'
import path from 'path'
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

function getBackendUrl(): string {
  try {
    const runtimeConfig = JSON.parse(
      fs.readFileSync(path.resolve(__dirname, '../.runtime/backend.json'), 'utf-8')
    )
    return runtimeConfig.url
  } catch {
    return 'http://localhost:8000'
  }
}

const backendUrl = getBackendUrl()

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': { target: backendUrl, changeOrigin: true },
      '/knowledge': { target: backendUrl, changeOrigin: true },
      '/config': { target: backendUrl, changeOrigin: true },
      '/send_input': { target: backendUrl, changeOrigin: true },
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    globals: true,
  },
})
