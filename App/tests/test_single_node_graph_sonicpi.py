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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGSMITH_API_KEY", "")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SETTINGS_PATH = PROJECT_ROOT / "App" / "static" / "config" / "settings.json"
with open(SETTINGS_PATH, "r", encoding="utf-8") as settings_file:
    SETTINGS = json.load(settings_file)

SONIC_PI_MODULE_PATH = PROJECT_ROOT / "App" / "services" / "sonicPi.py"
spec = importlib.util.spec_from_file_location("musicagent_sonicpi", SONIC_PI_MODULE_PATH)
sonicpi_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(sonicpi_module)
SonicPi = sonicpi_module.SonicPi


class SingleNodeState(TypedDict):
    song_name: str
    sonicpi_code: str
    feedback_message: str
    raw_response: str


class SingleNodeSonicPiGraphTest:
    def __init__(self):
        self.logger = logging.getLogger("single-node-sonicpi-graph-test")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())

        self.song_name = "single_node_graph_test"
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

    def _build_agent(self):
        validate_tool = self._build_validate_sonicpi_tool()
        system_prompt = "You are a Sonic Pi coding assistant. "
        "Write the smallest valid Sonic Pi snippet you can. "
        "After drafting code, call the validate_sonicpi_code tool to get local Sonic Pi feedback. "
        "Then return strict JSON only with keys sonicpi_code and feedback_message."
        return create_agent(
            model=self.model,
            tools=[validate_tool],
            system_prompt=system_prompt
        )

    def generate_and_validate(self, state: SingleNodeState):
        user_prompt = (
            "Write a very short Sonic Pi program that plays three notes. "
            "Use the local validation tool to get feedback message and modify your code. Your code should pass the local examination, and return JSON in this format: "
            '{"sonicpi_code": "..."}'
        )
        response = self.agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
        response_text = response["messages"][-1].content
        response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            raise ValueError(f"Agent did not return JSON. Raw response: {response_text}")

        response_data = json.loads(match.group(0))
        return {
            "sonicpi_code": response_data.get("sonicpi_code", ""),
            "feedback_message": response_data.get("feedback_message", ""),
            "raw_response": response_text,
        }

    def build_graph(self):
        graph_builder = StateGraph(SingleNodeState)
        graph_builder.add_node("Generate_And_Validate", self.generate_and_validate)
        graph_builder.add_edge(START, "Generate_And_Validate")
        graph_builder.add_edge("Generate_And_Validate", END)
        return graph_builder.compile()

    def run(self):
        graph = self.build_graph()
        initial_state = {
            "song_name": self.song_name,
            "sonicpi_code": "",
            "feedback_message": "",
            "raw_response": "",
        }
        result = graph.invoke(initial_state)
        print("\n=== Single Node Graph Result ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result


if __name__ == "__main__":
    SingleNodeSonicPiGraphTest().run()
