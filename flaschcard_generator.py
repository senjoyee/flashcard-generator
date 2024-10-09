import io
import json
from typing import Dict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
import PyPDF2
from docx import Document
import logging
from dotenv import load_dotenv
import os
import re

# Load environment variables from .env file
load_dotenv()


class FlashcardGenerator:
    """
    A class to generate flashcards from study materials using a language model and a workflow.
    """

    def __init__(self):
        """
        Initialize the FlashcardGenerator with a language model, workflow, and logger.
        """
        #self.llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
        #self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002")
        self.workflow = self._create_workflow()
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def _create_workflow(self):
        """
        Create and return a compiled workflow for flashcard generation.

        Returns:
            A compiled StateGraph object representing the flashcard generation workflow.
        """

        class State(Dict):
            study_materials: str
            categories: List[str]
            flashcards: Dict[str, List[Dict[str, str]]]
            current_category: str
            end: bool

        workflow = StateGraph(State)

        # Add nodes to the workflow
        workflow.add_node("create_categories", self._create_categories)
        workflow.add_node("create_flashcards", self._create_flashcards_for_category)
        workflow.add_node("select_next_category", self._select_next_category)

        # Add edges to connect the nodes
        workflow.add_edge("create_categories", "select_next_category")
        # workflow.add_edge("select_next_category", "create_flashcards")
        workflow.add_edge("create_flashcards", "select_next_category")

        def should_continue(state):
            return not state["end"]

        # Add conditional edges to control the flow
        workflow.add_conditional_edges(
            "select_next_category",
            should_continue,
            {True: "create_flashcards", False: END},
        )

        workflow.add_edge("create_flashcards", "select_next_category")

        # Set the entry point for the workflow
        workflow.set_entry_point("create_categories")

        return workflow.compile()

    def read_file_contents(self, file):
        """
        Read the contents of a file based on its extension.

        Args:
            file: A file object to read from.

        Returns:
            str: The contents of the file as a string.

        Raises:
            ValueError: If the file format is not supported.
        """
        file_extension = file.name.split(".")[-1].lower()

        if file_extension == "pdf":
            return self._read_pdf(file)
        elif file_extension == "docx":
            return self._read_docx(file)
        else:
            raise ValueError(
                "Unsupported file format. Please provide a PDF or Word document."
            )

    def _read_pdf(self, file):
        """
        Read the contents of a PDF file.

        Args:
            file: A file object containing a PDF.

        Returns:
            str: The text content of the PDF.
        """
        pdf_reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() for page in pdf_reader.pages)

    def _read_docx(self, file):
        """
        Read the contents of a Word document.

        Args:
            file: A file object containing a Word document.

        Returns:
            str: The text content of the Word document.
        """
        doc = Document(io.BytesIO(file.getvalue()))
        return " ".join(paragraph.text for paragraph in doc.paragraphs)

    def _create_categories(self, state: Dict) -> Dict:
        """
        Create categories from the study materials.

        Args:
            state (Dict): The current state of the workflow.

        Returns:
            Dict: A dictionary containing the list of created categories.
        """
        self.logger.info("Creating categories")
        prompt = """
        Generate categories from study materials for flashcard creation

        You will be provided with study materials on a specific topic. Your task is to:

        1. Analyze the content thoroughly.
        2. Identify key themes, concepts, and subject areas within the material.
        3. Create a list of distinct categories that comprehensively cover the main points of the study materials.
        4. Ensure the categories are broad enough to encompass multiple related facts or concepts, but specific enough to be meaningful for flashcard creation.
        5. Aim for between 7-10 main categories, depending on the breadth and depth of the material.
        6. Output only the names of the categories, one per line.

        Your output will be used in a subsequent step to create question/answer style flashcards for students, so keep this end goal in mind when formulating your categories.
        """

        response = self.llm.invoke(
            prompt + "\n\nStudy materials:\n" + state["study_materials"]
        )
        self.logger.info(f"LLM Response: {response.content}")
        categories = [
            cat.strip() for cat in response.content.split("\n") if cat.strip()
        ]
        self.logger.info(f"Extracted categories: {categories}")
        return {"categories": categories}

    def _create_flashcards_for_category(self, state: Dict) -> Dict:
        """
        Create flashcards for a specific category.

        Args:
            state (Dict): The current state of the workflow.

        Returns:
            Dict: A dictionary containing the newly created flashcards for the current category.
        """
        self.logger.info(
            f"Creating flashcards for category: {state['current_category']}"
        )
        prompt = """
        Using the provided study materials and categories, your task is to create detailed yet easy-to-understand question/answer style flashcards. Follow these guidelines:

        1. Ensure the answers are thorough but not overwhelming. Break down complex ideas into simpler terms.

        2. Use simple language, analogies, and a narrative structure to make the concept understandable to someone with no technical background. Focus on its main purpose and benefits, avoiding jargon and technical details.

        3. Vary the types of questions to include:
        - Definition questions
        - "How" and "Why" questions to test understanding
        - Comparison questions
        - Application questions that relate concepts to real-world scenarios

        4. Include mnemonics, diagrams, or other memory aids when helpful.

        5. Ensure each flashcard can stand alone without requiring information from other cards.


        Example output format:

        Category: [Name of Category]

        Flashcard 1:
        Q: [Clear, concise question]
        A: [Detailed, easy-to-understand answer. Include a brief story or example if applicable.]

        Flashcard 2:
        Q: [Another question related to the category]
        A: [Thorough answer with an interesting fact or analogy to aid memory]

        Flashcard 3 (Misconception):
        Q: [Common misunderstanding about a concept]
        A: [Correction of the misconception with a clear explanation]

        Flashcard 4 (Multi-step):
        Q: [Question requiring multiple steps to answer]
        A:
        Step 1: [First part of the answer]
        Step 2: [Second part of the answer]
        Step 3: [Final part of the answer]

        [Continue with additional flashcards for each category]

        Remember, the goal is to create flashcards that are not only informative but also engaging and memorable for students.


        Provide the flashcards in the following JSON format **without** any code fences, backticks, or additional text:

        [
            {
                "question": "Clear, concise question",
                "answer": "Detailed, easy-to-understand answer."
            },
            // ... more flashcards
        ]

        Ensure your response is valid JSON and contains **only** the JSON data, with no additional text or formatting.
        """

        response = self.llm.invoke(
            prompt
            + f"\n\nStudy materials:\n{state['study_materials']}\n\nCategory:\n{state['current_category']}"
        )

        # Preprocess the response to remove code fences
        content = response.content.strip()

        # Remove code fences if they exist
        if content.startswith("```") and content.endswith("```"):

            # Remove the first line (e.g., ```json) and the last line (```)
            content = "\n".join(content.split("\n")[1:-1]).strip()

        # Alternatively, use regex to extract JSON content
        # match = re.search(r'(\{.*\}|$.*$)', content, re.DOTALL)
        # if match:
        #    content = match.group(0)

        try:
            flashcards = json.loads(content)
            if not isinstance(flashcards, list):
                raise ValueError("Flashcards should be a list")
            for card in flashcards:
                if (
                    not isinstance(card, dict)
                    or "question" not in card
                    or "answer" not in card
                ):
                    raise ValueError("Invalid flashcard format")
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing flashcards: {str(e)}")
            self.logger.error(f"LLM response: {response.content}")
            flashcards = []

        return {
            "flashcards": {
                **state.get("flashcards", {}),
                state["current_category"]: flashcards,
            }
        }

    def _select_next_category(self, state: Dict) -> Dict:
        """
        Select the next category to process or end the workflow if all categories have been processed.

        Args:
            state (Dict): The current state of the workflow.

        Returns:
            Dict: An updated state dictionary with the next category or end flag set.
        """
        self.logger.info(
            f"Selecting next category. Remaining categories: {len(state['categories'])}"
        )
        if state["categories"]:
            next_category = state["categories"].pop(0)
            self.logger.info(f"Selected category: {next_category}")
            return {
                "categories": state["categories"],
                "current_category": next_category,
                "flashcards": state.get("flashcards", {}),
                "end": False,
            }
        else:
            self.logger.info("No more categories. Ending process.")
            return {
                "categories": [],
                "current_category": "",
                "flashcards": state.get("flashcards", {}),
                "end": True,
            }

    def generate_flashcards(self, study_materials: str) -> Dict:
        """
        Generate flashcards from the provided study materials.

        Args:
            study_materials (str): The study materials to generate flashcards from.

        Returns:
            Dict: A dictionary containing the generated flashcards or an error message.
        """
        if not study_materials or not isinstance(study_materials, str):
            raise ValueError("Invalid study materials provided")
        try:
            self.logger.info("Starting flashcard generation")
            initial_state = {
                "study_materials": study_materials,
                "categories": [],
                "flashcards": {},
                "current_category": "",
                "end": False,
            }
            self.logger.info(f"Initial state: {initial_state}")
            result = self.workflow.invoke(initial_state, {"recursion_limit": 25})
            self.logger.info(f"Workflow result: {result}")
            self.logger.info("Flashcard generation completed successfully")

            # Write results to a file
            file_path = self._write_results_to_file(result.get("flashcards", {}))

            return {"flashcards": result.get("flashcards", {})}
        except Exception as e:
            self.logger.error(
                f"An error occurred during flashcard generation: {str(e)}",
                exc_info=True,
            )
            return {"flashcards": {}, "error": str(e)}

    def _write_results_to_file(self, flashcards: Dict):
        """
        Write the generated flashcards to a JSON file.

        Args:
            flashcards (Dict): The flashcards to write to the file.
        """
        file_path = os.path.abspath("flashcards_output.txt")
        try:
            with open(file_path, "w") as f:
                for category, cards in flashcards.items():
                    f.write(f"Category: {category}\n\n")
                    for i, card in enumerate(cards, 1):
                        f.write(f"  {i}. Question: {card['question']}\n")
                        f.write(f"     Answer: {card['answer']}\n\n")
                    f.write("\n")
            self.logger.info(f"Flashcards written to {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"Error writing flashcards to file: {str(e)}")
            return ""


if __name__ == "__main__":
    generator = FlashcardGenerator()
    # Assuming you have a way to get study_materials, replace the following line accordingly
    study_materials = "Your study materials here"
    result = generator.generate_flashcards(study_materials)
    if "error" in result:
        print(f"An error occurred: {result['error']}")
    else:
        print(f"Flashcard generation complete. Results saved to: {result['file_path']}")
