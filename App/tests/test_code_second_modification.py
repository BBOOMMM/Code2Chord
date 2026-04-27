import importlib.util
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool


os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGSMITH_API_KEY", "")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SETTINGS_PATH = PROJECT_ROOT / "App" / "static" / "config" / "settings.json"
with open(SETTINGS_PATH, "r", encoding="utf-8") as settings_file:
    SETTINGS = json.load(settings_file)

NODE_PROMPTS_PATH = PROJECT_ROOT / "prompts" / "node_prompts.json"
with open(NODE_PROMPTS_PATH, "r", encoding="utf-8") as node_prompts_file:
    NODE_PROMPTS = json.load(node_prompts_file)

SYSTEM_PROMPTS_PATH = PROJECT_ROOT / "prompts" / "system_prompts.json"
with open(SYSTEM_PROMPTS_PATH, "r", encoding="utf-8") as system_prompts_file:
    SYSTEM_PROMPTS = json.load(system_prompts_file)

SONIC_PI_MODULE_PATH = PROJECT_ROOT / "App" / "services" / "sonicPi.py"
spec = importlib.util.spec_from_file_location("musicagent_sonicpi", SONIC_PI_MODULE_PATH)
sonicpi_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(sonicpi_module)
SonicPi = sonicpi_module.SonicPi


class CodeSecondModificationState(TypedDict):
    song_name: str
    theme: str
    melody: str
    rhythm: str
    structure: str
    total_duration: int
    segments: dict
    arrangements: str
    samples: str
    sonicpi_code: str
    raw_response: str
    feedback_message: str


class CodeSecondModificationGraphTest:
    def __init__(self):
        self.logger = logging.getLogger("code-second-modification-graph-test")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())

        self.song_name = "code_second_modification_test"
        self.song_directory = PROJECT_ROOT / "Songs" / self.song_name
        self.song_directory.mkdir(parents=True, exist_ok=True)

        api_key = os.environ.get("OPENAI_API_KEY") or SETTINGS.get("OPENAI_API_KEY")
        if not api_key or "your " in api_key.lower():
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            base_url="https://z.apiyihe.org/v1",
        )
        self.agent = self._build_agent()

    def create_song_file(self, sonicpi_code: str) -> Path:
        header = f"# --{self.song_name.upper()}-- \n\n"
        final_code = sonicpi_code if sonicpi_code.startswith(header) else header + sonicpi_code
        song_file = self.song_directory / f"{self.song_name}.rb"
        song_file.write_text(final_code, encoding="utf-8")
        return song_file

    def _build_validate_sonicpi_tool(self):
        @tool
        def validate_sonicpi_code(sonicpi_code: str) -> str:
            """Run the provided Sonic Pi code locally and return the feedback message from Sonic Pi."""
            song_file = self.create_song_file(sonicpi_code)
            sonic_pi = SonicPi(self.logger)
            feedback_message = sonic_pi.call_sonicpi(
                str(song_file),
                SETTINGS.get("sonic_pi_IP", "127.0.0.1"),
                int(SETTINGS.get("sonic_pi_port", 4560)),
            )
            return feedback_message or "No feedback returned from Sonic Pi."

        return validate_sonicpi_code

    def _build_human_review_tool(self):
        @tool
        def Human_Review(query: str) -> str:
            """Request suggestions from a human reviewer and return the human's comments."""
            print("\n=== Human Review Requested ===")
            print(query)
            human_response = input("\nPlease enter your review comments: ").strip()
            if not human_response:
                return "The human reviewer is satisfied. Keep the code unchanged."
            return human_response

        return Human_Review

    def _build_agent(self):
        validate_tool = self._build_validate_sonicpi_tool()
        human_review_tool = self._build_human_review_tool()
        system_prompt = SYSTEM_PROMPTS["Human Review"]
        return create_agent(
            model=self.model,
            tools=[validate_tool, human_review_tool],
            system_prompt=system_prompt,
        )

    def code_second_modification(self, state: CodeSecondModificationState):
        prompt_info = NODE_PROMPTS["Code Second Modification"]
        user_prompt = "".join(prompt_info["user_prompt"])
        for key, value in prompt_info["input"].items():
            user_prompt = user_prompt.replace(f"{{{key}}}", str(state.get(value, "")))

        response = self.agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
        response_text = response["messages"][-1].content
        response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            raise ValueError(f"Agent did not return JSON. Raw response: {response_text}")

        response_data = json.loads(match.group(0))
        sonicpi_code = response_data.get("sonicpi_code", "")
        feedback_message = ""

        if sonicpi_code:
            song_file = self.create_song_file(sonicpi_code)
            sonic_pi = SonicPi(self.logger)
            feedback_message = sonic_pi.call_sonicpi(
                str(song_file),
                SETTINGS.get("sonic_pi_IP", "127.0.0.1"),
                int(SETTINGS.get("sonic_pi_port", 4560)),
            ) or ""

        return {
            "sonicpi_code": sonicpi_code,
            "raw_response": response_text,
            "feedback_message": feedback_message,
        }

    def build_graph(self):
        graph_builder = StateGraph(CodeSecondModificationState)
        graph_builder.add_node("Code_Second_Modification", self.code_second_modification)
        graph_builder.add_edge(START, "Code_Second_Modification")
        graph_builder.add_edge("Code_Second_Modification", END)
        return graph_builder.compile()

    def run(self):
        graph = self.build_graph()
        initial_state = {
            "song_name": self.song_name,
            "theme": "romantic longing",
            "melody": "warm and dreamy with a simple hook",
            "rhythm": "steady pop groove",
            "structure": "Intro - Verse - Chorus - Outro",
            "total_duration": 20,
            "segments": {"Intro": 4, "Verse": 8, "Chorus": 6, "Outro": 2},
            "arrangements": "soft synth pad, kick, snare, simple lead",
            "samples": "[]",
            "sonicpi_code": "play 60\nsleep 0.5\nplay 64\nsleep 0.5\nplay 67",
            "raw_response": "",
            "feedback_message": "",
        }
        result = graph.invoke(initial_state)
        print("\n=== Code Second Modification Result ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result


if __name__ == "__main__":
    CodeSecondModificationGraphTest().run()
