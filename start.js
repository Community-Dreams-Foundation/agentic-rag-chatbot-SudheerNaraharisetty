#!/usr/bin/env node
/**
 * Cross-platform start script for Agentic RAG Chatbot.
 * Starts both the Python backend (FastAPI) and Next.js frontend.
 *
 * Usage: node start.js
 */

const { spawn } = require("child_process");
const path = require("path");
const http = require("http");

const ROOT = __dirname;
const FRONTEND_DIR = path.join(ROOT, "frontend");

const isWindows = process.platform === "win32";
const pythonCmd = isWindows ? "python" : "python3";
const npmCmd = isWindows ? "npm.cmd" : "npm";

let backendProc = null;
let frontendProc = null;
let shuttingDown = false;

function log(tag, msg) {
  const time = new Date().toLocaleTimeString();
  console.log(`[${time}] [${tag}] ${msg}`);
}

function startBackend() {
  log("BACKEND", "Starting FastAPI on port 8000...");

  backendProc = spawn(
    pythonCmd,
    ["-m", "uvicorn", "src.api.server:app", "--reload", "--port", "8000"],
    {
      cwd: ROOT,
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env },
      shell: isWindows,
    }
  );

  backendProc.stdout.on("data", (data) => {
    const lines = data.toString().trim().split("\n");
    lines.forEach((line) => log("BACKEND", line));
  });

  backendProc.stderr.on("data", (data) => {
    const lines = data.toString().trim().split("\n");
    lines.forEach((line) => log("BACKEND", line));
  });

  backendProc.on("close", (code) => {
    if (!shuttingDown) {
      log("BACKEND", `Process exited with code ${code}`);
    }
  });

  backendProc.on("error", (err) => {
    log("BACKEND", `Failed to start: ${err.message}`);
    log("BACKEND", `Make sure '${pythonCmd}' is available and dependencies are installed.`);
  });
}

function startFrontend() {
  log("FRONTEND", "Starting Next.js on port 3000...");

  frontendProc = spawn(npmCmd, ["run", "dev"], {
    cwd: FRONTEND_DIR,
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env },
    shell: isWindows,
  });

  frontendProc.stdout.on("data", (data) => {
    const lines = data.toString().trim().split("\n");
    lines.forEach((line) => log("FRONTEND", line));
  });

  frontendProc.stderr.on("data", (data) => {
    const lines = data.toString().trim().split("\n");
    lines.forEach((line) => log("FRONTEND", line));
  });

  frontendProc.on("close", (code) => {
    if (!shuttingDown) {
      log("FRONTEND", `Process exited with code ${code}`);
    }
  });

  frontendProc.on("error", (err) => {
    log("FRONTEND", `Failed to start: ${err.message}`);
    log("FRONTEND", `Make sure 'npm' is available and 'npm install' has been run in frontend/.`);
  });
}

function waitForBackend(maxAttempts = 60) {
  return new Promise((resolve) => {
    let attempts = 0;
    const check = () => {
      attempts++;
      const req = http.get("http://localhost:8000/api/health", (res) => {
        if (res.statusCode === 200) {
          log("BACKEND", "Ready!");
          resolve(true);
        } else {
          retry();
        }
      });
      req.on("error", retry);
      req.setTimeout(2000, retry);

      function retry() {
        if (attempts >= maxAttempts) {
          log("BACKEND", "Timed out waiting for backend. Starting frontend anyway.");
          resolve(false);
        } else {
          setTimeout(check, 1000);
        }
      }
    };
    check();
  });
}

function cleanup() {
  if (shuttingDown) return;
  shuttingDown = true;
  log("SYSTEM", "Shutting down...");

  if (backendProc && !backendProc.killed) {
    if (isWindows) {
      spawn("taskkill", ["/pid", String(backendProc.pid), "/f", "/t"], { shell: true });
    } else {
      backendProc.kill("SIGTERM");
    }
  }
  if (frontendProc && !frontendProc.killed) {
    if (isWindows) {
      spawn("taskkill", ["/pid", String(frontendProc.pid), "/f", "/t"], { shell: true });
    } else {
      frontendProc.kill("SIGTERM");
    }
  }

  setTimeout(() => process.exit(0), 1000);
}

// Handle Ctrl+C
process.on("SIGINT", cleanup);
process.on("SIGTERM", cleanup);

async function main() {
  console.log("");
  console.log("  ===================================");
  console.log("   Agentic RAG Chatbot - Starting...");
  console.log("  ===================================");
  console.log("");

  startBackend();
  await waitForBackend();
  startFrontend();

  console.log("");
  console.log("  ===================================");
  console.log("   Backend:  http://localhost:8000");
  console.log("   Frontend: http://localhost:3000");
  console.log("   API Docs: http://localhost:8000/docs");
  console.log("  ===================================");
  console.log("");
  console.log("  Press Ctrl+C to stop both servers.");
  console.log("");
}

main();
