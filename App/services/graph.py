from typing import Annotated, Literal
from typing_extensions import TypedDict
import json
import re
import os
import faiss
import numpy as np
import requests
import imghdr
from sentence_transformers import SentenceTransformer

from langgraph.graph import START, END, StateGraph
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver


from .songCreationData import SongCreationData
from App.config import Config
from .sonicPi import SonicPi

class MusicState(TypedDict):
    song_name: str
    song_description: str
    genre: str
    theme: str
    melody: str
    rhythm: str
    lyrics: str
    structure: str
    segments: dict
    total_duration: str
    arrangements: str
    sonicpi_code: str
    review: str
    album_url: str
    samples: str
    code_review_passed: bool
    code_review_loop_count: int
    max_code_review_loops: int


class MusicGraph:
    def __init__(self, selected_model, provider, logger, song_name, genre, duration, additional_information):
        self.selected_model = selected_model
        self.provider = provider
        self.logger = logger
        self.MAX_CODE_REVIEW_LOOPS = 3
        self.song_name = song_name
        self.song_description = f"I want to compose a brand new song. I like the " + genre + " genre. If I would describe the song, I would say: " + additional_information
        self.initial_state = self.build_initial_state(song_name, genre, duration)
        self.project_root = Config.PROJECT_ROOT
        self.songs_directory = os.path.join(self.project_root, 'Songs')
        if not os.path.exists(self.songs_directory):
            os.makedirs(self.songs_directory)
        self.song_directory = os.path.join(self.songs_directory, song_name)
        if not os.path.exists(self.song_directory):
           os.makedirs(self.song_directory)
        user_prompt_file = f"{self.project_root}/prompts/node_prompts.json"
        with open(user_prompt_file, "r") as f:
            self.user_prompts = json.load(f)
        system_prompt_file = f"{self.project_root}/prompts/system_prompts.json"
        with open(system_prompt_file, "r") as f:
            self.system_prompts = json.load(f)
        self.checkpoint_db_path = os.path.join(self.project_root, "music_graph_checkpoints.sqlite")
        self._checkpointer_cm = None
        self.checkpointer = None
        self.build_multi_agents()
    
    
    def build_initial_state(self, song_name, genre, duration) -> MusicState:
        return MusicState(
            song_name=song_name,
            song_description=self.song_description,
            genre=genre,
            theme="",
            melody="",
            rhythm="",
            lyrics="",
            structure="",
            segments={},
            total_duration=duration,
            arrangements="",
            sonicpi_code="",
            review="",
            album_url="",
            samples="",
            code_review_passed=False,
            code_review_loop_count=0,
            max_code_review_loops=self.MAX_CODE_REVIEW_LOOPS,
        )
    
    def build_multi_agents(self):
        if self.provider == "openai":
            self.model = ChatOpenAI(
                model=self.selected_model,
                api_key=Config.API_KEYS['openai'],
                base_url="https://z.apiyihe.org/v1")
            # tool_search = TavilySearch(max_results=2)
            # self.tools = [tool_search]
            self.tools = []
            self.agents = {}
            for agent_name in self.system_prompts:
                system_prompt = self.system_prompts[agent_name]
                agent = create_agent(
                    self.model,
                    tools=self.tools,
                    system_prompt=system_prompt,
                )
                self.agents[agent_name] = agent
        else:
            raise NotImplementedError


    def run(self):
        graph = self.build_graph()
        config = {"configurable": {"thread_id": self.song_name}}
        checkpoint = self._get_checkpointer().get(config)

        if checkpoint:
            self.logger.info(f"Resuming existing graph run for song: {self.song_name}")
            return graph.invoke(None, config=config)

        self.logger.info(f"Starting new graph run for song: {self.song_name}")
        return graph.invoke(self.initial_state, config=config)


    def _get_checkpointer(self):
        if self.checkpointer is not None:
            return self.checkpointer

        self._checkpointer_cm = SqliteSaver.from_conn_string(self.checkpoint_db_path)
        self.checkpointer = self._checkpointer_cm.__enter__()
        return self.checkpointer
    
    
    def agent_run(self, node_name, agent_name, state: MusicState, codeValidation=False):
        if node_name not in self.user_prompts:
            raise ValueError(f"No prompt found for node: {node_name}")

        prompt_info = self.user_prompts[node_name]
        user_prompt = ''.join(prompt_info["user_prompt"])
        for key, value in prompt_info["input"].items():
            param_value = state.get(value)
            user_prompt = user_prompt.replace(f"{{{key}}}", str(param_value))
        
        retry_count = 0
        max_retries = 3
        while retry_count < max_retries:
            messages = [
                {"role": "user", "content": user_prompt}
            ]
            response = self.agents[agent_name].invoke({"messages": messages})
            # response_text = response.choices[0].message.content # TODO
            response_text = response["messages"][-1].content
            response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
            
            user_prompt_single_line = user_prompt.replace('\n', ' ')
            self.logger.info(f"[Questioner]({node_name}):[[{user_prompt_single_line}]]")
            self.logger.info(f"Response (retry {retry_count})")
            response_text_single_line = response_text.replace('\n', ' ')
            self.logger.info(f"[Assistant]({agent_name}):[[{response_text_single_line}]]")
            
            # extract JSON object using regex if the response contains extra text
            try:
                match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if match:
                    response_data = json.loads(match.group(0))
                else:
                    response_data = json.loads(response_text)
                    
                if 'sonicpi_code' in response_data:
                    if isinstance(response_data['sonicpi_code'], list):
                        code_to_retrieve = '\n'.join(response_data['sonicpi_code'])
                    elif isinstance(response_data['sonicpi_code'], str):
                        code_to_retrieve = response_data['sonicpi_code']
                    
                    if code_to_retrieve and isinstance(code_to_retrieve, str):
                        fixed_code = re.sub(r":([A-G])#(\d)", lambda m: f":{m.group(1).lower()}s{m.group(2)}", code_to_retrieve)
                    else:
                        print(f"Warning: code_to_retrieve is not a valid string. Value: {code_to_retrieve}")
                        fixed_code = code_to_retrieve 
                    
                    self.logger.info(f"Code successfully retrieved: {fixed_code}")
                    response_data["sonicpi_code"] = fixed_code
                    songfile_path = self.create_song_file(state['song_name'], fixed_code)
                    
                    if codeValidation:
                        if not self.validate_and_execute_code(songfile_path):
                            self.logger.info(f"Error detected in Sonic Pi code, should be corrected before continuing.")
                            retry_count += 1
                            continue

                if isinstance(response_data, dict) and response_data.keys() == self.user_prompts[node_name]["outcome"].keys():
                    return response_data
            
            except json.JSONDecodeError as e:
                self.logger.info(f"Failed to parse the response as JSON: {e}")
                retry_count += 1
        
        if retry_count >= max_retries:
            self.logger.info("Maximum number of retries reached, unable to parse JSON")
    
            
    def samples_retrieve(self, k, state: MusicState):
        faiss_index_path = os.path.join(self.project_root, 'Samples', 'sample_index.faiss')
        metadata_path = os.path.join(self.project_root, 'Samples', 'sample_metadata.json')
        index = faiss.read_index(faiss_index_path)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            
        model = SentenceTransformer('all-MiniLM-L6-v2')

        query = f"We are creating a new song with the following details: - Theme: {state['theme']}. " \
                f"- Melody: {state['melody']}. " \
                f"- Rhythm: {state['rhythm']}. " \
                f"- Song Description: {state['song_description']}. " \
                "Please suggest suitable samples with tags matching the mood, progression, and instrumentation described above."
        query_embedding = model.encode(query)
        query_embedding = np.array([query_embedding]).astype("float32")
        
        # Search for the nearest neighbors
        self.logger.info(f"Number of samples that will be retrieved: {k}")
        distances, indices = index.search(query_embedding, k)

        results = []
        for i in range(k):
            results.append(metadata[indices[0][i]])

        samples = json.dumps(results, separators=(',', ':'))
        
        return samples
    
    
    def create_song_file(self, song_name, code):
        self.logger.info("\nCreating the song file.")

        header = f"# --{song_name.upper()}-- \n\n"
        if not code.startswith(header):
            code = header + code

        # Full path samples in the sonic pi code for song (excluded from parameter to avoid sending to openai, anthropic, ...)
        project_directory = os.path.join(self.project_root, "Samples")
        def replace_sample_path(match):
            full_path = os.path.normpath(os.path.join(project_directory, match.group(1)))
            full_path = full_path.replace("\\", "\\\\")
            return f'sample "{full_path}"'

        finalcode = re.sub(
            r'sample\s+"([^"]+)"',
            replace_sample_path,
            code
        )

        self.logger.info("Writing song to directory: " + self.song_directory)
        self.logger.info("sonic pi code " + finalcode)

        song_file = os.path.join(self.song_directory, song_name + '.rb')

        if os.path.exists(song_file):
            # Compare the existing file content with the new content
            with open(song_file, 'r') as existing_file:
                existing_content = existing_file.read()
            if existing_content != finalcode:
                # Find the next available index for renaming
                index = 1
                while True:
                    new_song_file = os.path.join(self.song_directory, f"{song_name}_{index}.rb")
                    if not os.path.exists(new_song_file):
                        os.rename(song_file, new_song_file)
                        break
                    index += 1
                    
        with open(song_file, 'w') as f:
            f.write(finalcode)
            
        return song_file
            
    
    def validate_and_execute_code(self, songfile_path):
        sonic_pi = SonicPi(self.logger)
        feedback_message = sonic_pi.call_sonicpi(songfile_path, Config.SONIC_PI_HOST, Config.SONIC_PI_PORT)

        if feedback_message is not None and "error" in feedback_message.lower():
            self.logger.info(f"Error detected in Sonic Pi execution: {feedback_message}")
            return False
        return True
    

    def Conceptualization(self, state: MusicState):
        self.logger.info("Starting Conceptualization")
        node_name = "Conceptualization"
        agent_name = "Composer"
        response_data = self.agent_run(node_name, agent_name, state)
        if response_data:
            return {
                "theme": response_data.get("theme", ""),
                "melody": response_data.get("melody", ""),
                "rhythm": response_data.get("rhythm", ""),
            }
        return {}

    def Songwriting(self, state: MusicState):
        self.logger.info("Starting Songwriting")
        node_name = "Songwriting"
        agent_name = "Songwriter"
        response_data = self.agent_run(node_name, agent_name, state)
        if response_data:
            return {
                "lyrics": response_data.get("lyrics", ""),
                "structure": response_data.get("structure", "")
            }
        return {}

    def Segmentation(self, state: MusicState):
        self.logger.info("Starting Segmentation")
        node_name = "Segmentation"
        agent_name = "Arranger"
        response_data = self.agent_run(node_name, agent_name, state)
        if response_data:
            return {
                "segments": response_data.get("segments", ""),
            }
        return {}

    def Arrangements(self, state: MusicState):
        self.logger.info("Starting Arrangements")
        node_name = "Arrangements"
        agent_name = "Arranger"
        response_data = self.agent_run(node_name, agent_name, state)
        if response_data:
            return {
                "arrangements": response_data.get("arrangements", "")
            }
        return {}

    def Sampling(self, state: MusicState):
        self.logger.info("Starting Sampling")
        samples = self.samples_retrieve(5, state)
        if samples:
            return {
                "samples": samples
            }
        return {}

    def Initial_Song_Coding(self, state: MusicState):
        self.logger.info("Starting Initial Song Coding")
        node_name = "Initial Song Coding"
        agent_name = "Sonic PI coder"
        response_data = self.agent_run(node_name, agent_name, state)
        if response_data:
            return {
                "sonicpi_code": response_data.get("sonicpi_code", "")
            }
        return {}

    def Code_Review(self, state: MusicState):
        self.logger.info("Starting Code Review")
        node_name = "Code Review"
        agent_name = "Sonic PI reviewer"
        response_data = self.agent_run(node_name, agent_name, state)
        if response_data and "review" in response_data and isinstance(response_data["review"], str):
            if 'no further code changes are required' in response_data["review"].lower():
                return {
                    "review": response_data.get("review", ""),
                    "code_review_passed": True
                }
            else:
                return {
                    "review": response_data.get("review", ""),
                    "code_review_passed": False
                }
        return {}
    
    
    def route_after_code_review(self, state: MusicState):
        if state.get("code_review_passed", False):
            return "Song_mixing"

        if state.get("code_review_loop_count", 0) >= self.MAX_CODE_REVIEW_LOOPS:
            return "Song_mixing"

        return "Code_Modification"


    def Code_Modification(self, state: MusicState):
        self.logger.info("Starting Code Modification")
        node_name = "Code Modification"
        agent_name = "Sonic PI coder"
        response_data = self.agent_run(node_name, agent_name, state)
        if response_data:
            return {
                "sonicpi_code": response_data.get("sonicpi_code", ""),
                "code_review_passed": False,
                "code_review_loop_count": state.get("code_review_loop_count", 0) + 1
            }
        return {}
    

    def Song_mixing(self, state: MusicState):
        self.logger.info("Starting Song Mixing")
        node_name = "Song Mixing"
        agent_name = "Sonic PI coder"
        response_data = self.agent_run(node_name, agent_name, state)
        if response_data:
            return {
                "sonicpi_code": response_data.get("sonicpi_code", "")
            }
        return {}
    

    def Cover_Art(self, state: MusicState):
        self.logger.info(f"Starting cover art generation phase.")
        album_cover_style = ''.join(self.user_prompts["Cover Art"]["user_prompt"])
        image_prompt = "Album cover style defined as: " + album_cover_style + " / Song on the album described as " + state['song_description'] + "Please create an album cover for this song. The image should be in a square format."
        
        if self.provider == "openai":
            model = ChatOpenAI(
                model='gpt-image-1.5',
                api_key=Config.API_KEYS['openai'],
                base_url="https://z.apiyihe.org/v1"
            )
        else:
            raise NotImplementedError
        
        response = model.invoke(
            {"messages": [{"role": "user", "content": image_prompt}]}
        )
        image_url = response["messages"][-1].content.strip()
        self.logger.info("Image URL: " + image_url)
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            # Detect image type
            image_type = imghdr.what(None, h=image_response.content)
            filename = state['song_name'] + "_cover"
            if image_type:
                self.image_type = image_type
                filename = f"{filename}.{image_type}"
            file_path = os.path.join(self.song_directory, filename)
            with open(file_path, 'wb') as file:
                file.write(image_response.content)
            self.logger.info(f"Image downloaded as {file_path}")
            return {
                "album_url": file_path
            }
        else:
            self.logger.info("Failed to download image")
            return {}


    def Booklet_Creation(self, state: MusicState):
        self.logger.info(f"Starting booklet creation phase.")
        readme_file = os.path.join(self.song_directory, 'README.md')
        readme_content = (
            f"# --- {state['song_name']} ---"
            f"![Album Cover]({state['song_name']}_cover.{self.image_type})\n"
            f"\n\nLyrics: \n{state['lyrics']}\n\n"
            f"---\n\n"
            f"## Song Parameters\n"
            f"Song Prompt: {state['song_description']}\n\n"
            f"Theme: {state['theme']}\n\n"
            f"Melody: {state['melody']}\n\n"
            f"Rhythm: {state['rhythm']}\n\n"
            f"Structure: {state['structure']}\n\n"
            f"Segments: {state['segments']}\n\n"
            f"Total Duration: {state['total_duration']} seconds\n\n\n"
            f"Arrangements: {state['arrangements']}\n\n\n"
        )
        with open(readme_file, 'w', encoding='utf-8') as file:
            file.write(readme_content)


    def build_graph(self):
        graph_builder = StateGraph(MusicState)

        graph_builder.add_node("Conceptualization", self.Conceptualization)
        graph_builder.add_node("Songwriting", self.Songwriting)
        graph_builder.add_node("Segmentation", self.Segmentation)
        graph_builder.add_node("Arrangements", self.Arrangements)
        graph_builder.add_node("Sampling", self.Sampling)
        graph_builder.add_node("Initial_Song_Coding", self.Initial_Song_Coding)

        graph_builder.add_node("Code_Review", self.Code_Review)
        graph_builder.add_node("Code_Modification", self.Code_Modification)

        graph_builder.add_node("Song_mixing", self.Song_mixing)
        graph_builder.add_node("Cover_Art", self.Cover_Art)
        graph_builder.add_node("Booklet_Creation", self.Booklet_Creation)

        graph_builder.add_edge(START, "Conceptualization")
        graph_builder.add_edge("Conceptualization", "Songwriting")
        graph_builder.add_edge("Songwriting", "Segmentation")
        graph_builder.add_edge("Segmentation", "Arrangements")
        graph_builder.add_edge("Arrangements", "Sampling")
        graph_builder.add_edge("Sampling", "Initial_Song_Coding")
        graph_builder.add_edge("Initial_Song_Coding", "Code_Review")

        graph_builder.add_conditional_edges(
            "Code_Review",
            self.route_after_code_review,
            {
                "Song_mixing": "Song_mixing",
                "Code_Modification": "Code_Modification",
            },
        )

        graph_builder.add_edge("Code_Modification", "Code_Review")
        graph_builder.add_edge("Song_mixing", "Cover_Art")
        graph_builder.add_edge("Cover_Art", "Booklet_Creation")
        graph_builder.add_edge("Booklet_Creation", END)

        graph = graph_builder.compile(checkpointer=self._get_checkpointer())
        
        try:
            png_bytes = graph.get_graph().draw_mermaid_png()
            with open("graph.png", "wb") as f:
                f.write(png_bytes)
            print("Saved to graph.png")
        except Exception as e:
            print(f"Graph visualization is not available: {e}")
        
        return graph
