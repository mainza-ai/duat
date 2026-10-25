import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// TEMPORARY local-test config (untracked helper): API on :8001, web on :5174.
// The repo's vite.config.ts targets :8000, which is occupied on this machine.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: { "/api": "http://127.0.0.1:8001" },
  },
});
