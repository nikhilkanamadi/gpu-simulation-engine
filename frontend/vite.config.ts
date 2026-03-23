import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // GitHub Pages serves the site from a subpath (e.g. /gpu-simulation-engine/).
  // Using relative asset paths keeps the built app working when deployed via `/docs`.
  base: "./",
  server: {
    port: 5173,
    host: true,
    proxy: {
      "/api": "http://localhost:3001",
      "/ws": {
        target: "http://localhost:3001",
        ws: true
      }
    }
  }
});

