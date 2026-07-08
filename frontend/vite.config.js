import { defineConfig } from 'vite'

export default defineConfig({
  // Set base to your repo name for GitHub Pages, e.g. '/bos/'
  // Override with VITE_BASE env var in CI if the repo name differs.
  base: process.env.VITE_BASE ?? '/',
  build: {
    outDir: 'dist',
  },
})
