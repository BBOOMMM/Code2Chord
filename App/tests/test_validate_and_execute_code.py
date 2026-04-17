import importlib.util
import logging
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SONIC_PI_MODULE_PATH = Path(__file__).resolve().parents[1] / "services" / "sonicPi.py"
spec = importlib.util.spec_from_file_location("musicagent_sonicpi", SONIC_PI_MODULE_PATH)
sonicpi_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(sonicpi_module)
SonicPi = sonicpi_module.SonicPi


class TestSonicPiContract(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("ValidateExecuteCodeTest")
        self.logger.setLevel(logging.INFO)
        self.sonic_pi = SonicPi(self.logger, feedback_timeout=0.01)

    def _feedback_tuple(self, message="Received message from /feedback: ('ok',)"):
        feedback_event = threading.Event()
        feedback_event.set()
        return object(), object(), feedback_event, {"message": message}

    @patch.object(SonicPi, "_get_runtime_connection_details")
    @patch.object(SonicPi, "_send_internal_run_code")
    @patch.object(SonicPi, "_stop_feedback_server")
    @patch.object(SonicPi, "_start_feedback_server")
    def test_call_sonicpi_accepts_file_path(
        self, start_server, stop_server, send_internal_run_code, get_runtime_connection_details
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "demo.rb"
            script_path.write_text("play 60", encoding="utf-8")
            start_server.return_value = self._feedback_tuple()
            get_runtime_connection_details.return_value = {
                "server_port": 35583,
                "cue_port": 4560,
                "token": 123,
            }

            message = self.sonic_pi.call_sonicpi(str(script_path), "127.0.0.1", 4557)

        self.assertIn("Received message from /feedback", message)
        wrapped_script = send_internal_run_code.call_args.args[1]
        self.assertIn("eval(\"play 60\")", wrapped_script)
        stop_server.assert_called_once()

    @patch.object(SonicPi, "_get_runtime_connection_details")
    @patch.object(SonicPi, "_send_internal_run_code")
    @patch.object(SonicPi, "_stop_feedback_server")
    @patch.object(SonicPi, "_start_feedback_server")
    def test_call_sonicpi_accepts_song_object(
        self, start_server, stop_server, send_internal_run_code, get_runtime_connection_details
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            song = SimpleNamespace(name="demo", song_dir=temp_dir)
            script_path = Path(temp_dir) / "demo.rb"
            script_path.write_text("play 65", encoding="utf-8")
            start_server.return_value = self._feedback_tuple("Received message from /feedback: ('song',)")
            get_runtime_connection_details.return_value = {
                "server_port": 35583,
                "cue_port": 4560,
                "token": 123,
            }

            message = self.sonic_pi.call_sonicpi(song, "127.0.0.1", 4557)

        self.assertIn("song", message)
        wrapped_script = send_internal_run_code.call_args.args[1]
        self.assertIn("eval(\"play 65\")", wrapped_script)
        stop_server.assert_called_once()

    @patch.object(SonicPi, "_get_runtime_connection_details")
    @patch.object(SonicPi, "_send_internal_run_code")
    @patch.object(SonicPi, "_stop_feedback_server")
    @patch.object(SonicPi, "_start_feedback_server")
    def test_call_sonicpi_returns_timeout_message_when_no_feedback(
        self, start_server, stop_server, send_internal_run_code, get_runtime_connection_details
    ):
        feedback_event = threading.Event()
        start_server.return_value = object(), object(), feedback_event, {"message": None}
        get_runtime_connection_details.return_value = {
            "server_port": 35583,
            "cue_port": 4560,
            "token": 123,
        }

        message = self.sonic_pi.call_sonicpi(
            None,
            "127.0.0.1",
            4557,
            full_script="play 70",
        )

        self.assertIn("Timeout waiting for Sonic Pi feedback", message)
        stop_server.assert_called_once()


if __name__ == "__main__":
    unittest.main()
