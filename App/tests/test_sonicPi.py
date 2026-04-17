import importlib.util
import logging
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SONIC_PI_MODULE_PATH = Path(__file__).resolve().parents[1] / "services" / "sonicPi.py"
spec = importlib.util.spec_from_file_location("musicagent_sonicpi", SONIC_PI_MODULE_PATH)
sonicpi_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(sonicpi_module)
SonicPi = sonicpi_module.SonicPi


class TestSonicPiHelpers(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("SonicPiHelperTest")
        self.logger.setLevel(logging.INFO)
        self.sonic_pi = SonicPi(self.logger)

    def test_resolve_script_from_full_script(self):
        self.assertEqual(self.sonic_pi._resolve_script(full_script="play 60"), "play 60")

    def test_resolve_script_from_file_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "demo.rb"
            script_path.write_text("play 72", encoding="utf-8")

            result = self.sonic_pi._resolve_script(song_or_path=script_path)

        self.assertEqual(result, "play 72")

    def test_build_feedback_wrapper_contains_ack_and_eval(self):
        wrapped = self.sonic_pi._build_feedback_wrapper("play 72")

        self.assertIn("ACK: received script", wrapped)
        self.assertIn('eval("play 72")', wrapped)


if __name__ == "__main__":
    unittest.main()
