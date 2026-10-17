import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';

// The `assert` and `events` packages in package.json are what stop Vite from externalizing
// those Node built-ins (kepler.gl and thrift import them); no alias is needed.
export default defineConfig({
  plugins: [react()],
  define: {
    global: 'globalThis',
    'process.env': {}
  },
  build: {chunkSizeWarningLimit: 20000}
});