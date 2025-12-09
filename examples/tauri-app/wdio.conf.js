import path from "path";
import { spawn, spawnSync } from "child_process";
import { fileURLToPath } from "url";
import { Socket } from "net";
import { CrabNebulaCloudReporter } from "@crabnebula/webdriverio-cloud-reporter";
import { waitTauriDriverReady } from "@crabnebula/tauri-driver";
import { waitTestRunnerBackendReady } from "@crabnebula/test-runner-backend";

const __dirname = fileURLToPath(new URL(".", import.meta.url));

// TODO: relative path
const applicationPath = "/Users/fabianlars/dev/_target/debug/tauri-app";

let wkwebdriver;
// keep track of the `tauri-driver` child process
let tauriDriver;
let killedTauriDriver = false;
// keep track of the `test-runner-backend` child process
let testRunnerBackend;
let killedTestRunnerBackend = false;

export const config = {
  host: "127.0.0.1",
  port: 4444,
  specs: ["./test/specs/**/*.js"],
  maxInstances: 1,

  capabilities: [
    {
      maxInstances: 1,
      "tauri:options": {
        application: applicationPath,
      },
      "wkwebview:options": {
        binary: `${applicationPath}.app`,
      },
    },
  ],
  framework: "mocha",
  reporters: [CrabNebulaCloudReporter],
  mochaOpts: {
    ui: "bdd",
    timeout: 60000,
  },
  connectionRetryCount: 0,

  onPrepare: async () => {
    // ensure the Tauri app is built since we expect this binary to exist for the webdriver sessions
    spawnSync("pnpm", ["tauri", "build", "--debug", "--no-bundle"], {
      cwd: path.resolve(__dirname, "src-tauri"),
      stdio: "inherit",
      shell: true,
    });

    if (process.platform === "darwin") {
      // CN_API_KEY is required to run macOS tests via CrabNebula Webdriver for Tauri
      if (!process.env.CN_API_KEY) {
        console.error(
          "CN_API_KEY is not set, required for CrabNebula Webdriver"
        );
        process.exit(1);
      }

      testRunnerBackend = spawn("pnpm", ["test-runner-backend"], {
        stdio: "inherit",
        shell: true,
      });

      testRunnerBackend.on("error", (error) => {
        console.error("test-runner-backend error:", error);
        process.exit(1);
      });
      testRunnerBackend.on("exit", (code) => {
        if (!killedTestRunnerBackend) {
          console.error("test-runner-backend exited with code:", code);
          process.exit(1);
        }
      });

      await waitTestRunnerBackendReady();

      // instruct tauri-driver to connect to the test-runner-backend
      process.env.REMOTE_WEBDRIVER_URL = `http://127.0.0.1:3000`;
    }
  },

  // ensure we are running `tauri-driver` before the session starts so that we can proxy the webdriver requests
  beforeSession: async () => {
    tauriDriver = spawn("pnpm", ["tauri-driver"], {
      stdio: [null, process.stdout, process.stderr],
      shell: true,
    });
    tauriDriver.on("error", (error) => {
      console.error("tauri-driver error:", error);
      process.exit(1);
    });
    tauriDriver.on("exit", (code) => {
      if (!killedTauriDriver) {
        console.error("tauri-driver exited with code:", code);
        process.exit(1);
      }
    });

    // wait for tauri-driver to initialize its proxy server
    await waitTauriDriverReady();
  },

  // clean up the `tauri-driver` process we spawned at the start of the session
  afterSession: () => {
    closeTauriDriver();
  },

  onComplete: () => {
    killedTestRunnerBackend = true;
    testRunnerBackend?.kill();
  },
};

function closeTauriDriver() {
  killedTauriDriver = true;
  tauriDriver?.kill();
  killedTestRunnerBackend = true;
  testRunnerBackend?.kill();
}

export function onShutdown(fn) {
  const cleanup = () => {
    try {
      fn();
    } finally {
      process.exit();
    }
  };

  process.on("exit", cleanup);
  process.on("SIGINT", cleanup);
  process.on("SIGTERM", cleanup);
  process.on("SIGHUP", cleanup);
  process.on("SIGBREAK", cleanup);
}

onShutdown(closeTauriDriver);
