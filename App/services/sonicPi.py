from pathlib import Path
import json
import os
import re
import subprocess
import sys
import threading
import time

from pythonosc import dispatcher, osc_server, udp_client


class SonicPi:
    def __init__(
        self,
        logger,
        feedback_host="127.0.0.1",
        feedback_port=4559,
        feedback_timeout=20,
        startup_timeout=30,
    ):
        self.logger = logger
        self.feedback_host = feedback_host
        self.feedback_port = feedback_port
        self.feedback_timeout = feedback_timeout
        self.startup_timeout = startup_timeout

    def _build_song_script_path(self, song):
        return Path(song.song_dir) / f"{song.name}.rb"

    def _get_configured_sonic_pi_executable(self):
        env_path = os.environ.get("SONIC_PI_EXECUTABLE")
        if env_path:
            return env_path

        settings_path = Path(__file__).resolve().parents[1] / "static" / "config" / "settings.json"
        try:
            with open(settings_path, "r", encoding="utf-8") as file:
                settings = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

        return settings.get("sonic_pi_executable")

    def _read_script(self, script_path):
        with open(script_path, "r", encoding="utf-8") as file:
            return file.read()

    def _resolve_script(self, song_or_path=None, full_script=None):
        if full_script is not None:
            return full_script

        if song_or_path is None:
            raise ValueError("Either full_script or song/song_file_path must be provided.")

        if hasattr(song_or_path, "song_dir") and hasattr(song_or_path, "name"):
            script_path = self._build_song_script_path(song_or_path)
            self.logger.info(f"Script in path {script_path}")
            return self._read_script(script_path)

        script_path = Path(song_or_path)
        self.logger.info(f"Script in path {script_path}")
        return self._read_script(script_path)

    def _start_feedback_server(self):
        feedback_event = threading.Event()
        feedback_data = {"message": None}

        def handle_message(address, *args):
            feedback_data["message"] = f"Received message from {address}: {args}"
            feedback_event.set()
            self.logger.info(feedback_data["message"])

        disp = dispatcher.Dispatcher()
        disp.map("/feedback", handle_message)

        server = osc_server.ThreadingOSCUDPServer((self.feedback_host, self.feedback_port), disp)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        return server, server_thread, feedback_event, feedback_data

    def _stop_feedback_server(self, server, server_thread):
        if server is None or server_thread is None:
            return

        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1)

    def _get_spider_command_line(self):
        if sys.platform.startswith("win"):
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*spider-server.rb*' } | "
                    "Select-Object -First 1 -ExpandProperty CommandLine)",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            command_line = result.stdout.strip()
            if command_line:
                return command_line
        else:
            result = subprocess.run(
                ["ps", "-ax", "-o", "command="],
                capture_output=True,
                text=True,
                check=True,
            )
            for line in result.stdout.splitlines():
                if "spider-server.rb" in line:
                    return line.strip()

        return None

    def _find_sonic_pi_executable(self):
        candidate_paths = []
        configured_path = self._get_configured_sonic_pi_executable()
        if configured_path:
            candidate_paths.append(configured_path)
        if sys.platform.startswith("win"):
            candidate_paths.extend(
                [
                    r"C:\Program Files\Sonic Pi\app\gui\build\Release\sonic-pi.exe",
                    r"C:\Program Files\Sonic Pi\Sonic Pi.exe",
                    r"C:\Program Files\Sonic Pi\sonic-pi.exe",
                    r"C:\Program Files (x86)\Sonic Pi\Sonic Pi.exe",
                    r"C:\Program Files (x86)\Sonic Pi\sonic-pi.exe",
                ]
            )

        env_path = os.environ.get("SONIC_PI_EXECUTABLE")
        if env_path:
            candidate_paths.insert(0, env_path)

        for candidate in candidate_paths:
            if Path(candidate).exists():
                return candidate

        return None

    def _launch_sonic_pi(self):
        executable = self._find_sonic_pi_executable()
        if not executable:
            raise RuntimeError(
                "Sonic Pi is not running and no executable was found. "
                "Set SONIC_PI_EXECUTABLE or install Sonic Pi in the default location."
            )

        self.logger.info(f"Starting Sonic Pi from {executable}")
        if sys.platform.startswith("win"):
            subprocess.Popen([executable], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen([executable], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _ensure_runtime_connection_details(self):
        runtime_details = self._get_runtime_connection_details()
        if runtime_details is not None:
            return runtime_details

        self._launch_sonic_pi()
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            runtime_details = self._get_runtime_connection_details()
            if runtime_details is not None:
                self.logger.info("Sonic Pi runtime is ready.")
                return runtime_details
            time.sleep(1)

        raise RuntimeError(
            f"Sonic Pi did not become ready within {self.startup_timeout} seconds."
        )

    def _get_runtime_connection_details(self):
        command_line = self._get_spider_command_line()
        if not command_line:
            return None
        match = re.search(r"\s-u\s+((?:-?\d+\s+){7}-?\d+)", command_line)
        if not match:
            raise RuntimeError(f"Could not parse Sonic Pi runtime ports/token from: {command_line}")

        values = [int(value) for value in match.group(1).split()]
        return {
            "server_port": values[0],
            "cue_port": values[4],
            "token": values[-1],
        }

    def _send_internal_run_code(self, runtime_details, code):
        client = udp_client.SimpleUDPClient("127.0.0.1", runtime_details["server_port"])
        client.send_message("/run-code", [runtime_details["token"], code])

    def _build_feedback_wrapper(self, full_script):
        encoded_script = json.dumps(full_script)
        return f"""
osc_send '{self.feedback_host}', {self.feedback_port}, '/feedback', 'ACK: received script'

in_thread do
  begin
    eval({encoded_script})
  rescue Exception => e
    osc_send '{self.feedback_host}', {self.feedback_port}, '/feedback', "ERROR: #{{e.message}}"
  end
end
"""

    def call_sonicpi(self, song_or_path=None, ip_address="127.0.0.1", port=4557, full_script=None):
        full_script = self._resolve_script(song_or_path=song_or_path, full_script=full_script)
        runtime_details = self._ensure_runtime_connection_details()
        wrapped_script = self._build_feedback_wrapper(full_script)

        self.logger.info(
            f"Running code in Sonic PI on {ip_address}:{runtime_details['server_port']} "
            f"(server_port={runtime_details['server_port']})"
        )

        server = None
        server_thread = None
        try:
            server, server_thread, feedback_event, feedback_data = self._start_feedback_server()
            self.logger.info("Script before sending:\n" + full_script)
            self._send_internal_run_code(runtime_details, wrapped_script)

            if feedback_event.wait(timeout=self.feedback_timeout):
                return feedback_data["message"]

            timeout_message = (
                f"Timeout waiting for Sonic Pi feedback after {self.feedback_timeout} seconds."
            )
            self.logger.warning(timeout_message)
            return timeout_message
        finally:
            self._stop_feedback_server(server, server_thread)
