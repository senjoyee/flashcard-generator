import io
import json
from typing import Dict, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
import PyPDF2
from docx import Document
import logging
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class FlashcardGenerator:
    def __init__(self):
        #self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002")
        #self.llm = ChatAnthropic(model="claude-3-5-sonnet")
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.workflow = self._create_workflow()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def _create_workflow(self):
        class State(Dict):
            study_materials: str
            flashcards: List[Dict[str, str]]
            reflection: str
            end: bool

        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("create_flashcards", self._create_flashcards)
        workflow.add_node("reflect_on_coverage", self._reflect_on_coverage)
        workflow.add_node("update_flashcards", self._update_flashcards)
        
        # Set entry point
        workflow.set_entry_point("create_flashcards")
        
        # Add edges
        workflow.add_edge("create_flashcards", "reflect_on_coverage")
        workflow.add_edge("reflect_on_coverage", "update_flashcards")
        workflow.add_edge("update_flashcards", END)

        return workflow.compile()

    def read_file_contents(self, file):
        file_extension = file.name.split(".")[-1].lower()

        if file_extension == "pdf":
            return self._read_pdf(file)
        elif file_extension == "docx":
            return self._read_docx(file)
        else:
            raise ValueError("Unsupported file format. Please provide a PDF or Word document.")

    def _read_pdf(self, file):
        pdf_reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() for page in pdf_reader.pages)

    def _read_docx(self, file):
        doc = Document(io.BytesIO(file.getvalue()))
        return " ".join(paragraph.text for paragraph in doc.paragraphs)

    def _create_flashcards(self, state: Dict) -> Dict:
        self.logger.info("Creating flashcards")
        prompt = """
        Using the provided study materials, your task is to create comprehensive, engaging, and easy-to-understand question/answer style flashcards. Follow these guidelines:

        1. Content Analysis and Coverage:
        - Thoroughly analyze the study materials to identify ALL key concepts, principles, and important details
        - Create as many flashcards as necessary to cover the material completely
        - Ensure no important concept is left uncovered

        2. Answer Structure:
        - Begin with a clear, direct answer to the question
        - Follow with a relatable real-world analogy (e.g., comparing technical concepts to everyday situations like cooking, sports, or nature)
        - Expand with detailed explanations using simple language
        - End with practical examples or applications where relevant

        3. Analogy Requirements:
        - Include at least one creative, memorable analogy in each answer
        - Use analogies that relate to common experiences (cooking, driving, building, nature, etc.)
        - Ensure analogies accurately represent the concept's core mechanics or principles
        - Keep analogies culturally neutral and widely understandable

        4. Question Variety:
        Include a mix of:
        - Definition questions ("What is...?")
        - Process questions ("How does...?")
        - Reasoning questions ("Why does...?")
        - Comparison questions ("What's the difference between...?")
        - Application questions ("In what scenario would you use...?")
        - Problem-solving questions ("How would you handle...?")

        5. Enhancement Elements:
        Where beneficial, include:
        - Mnemonics for complex terms or lists
        - Simple diagrams or visual descriptions
        - Step-by-step breakdowns of processes
        - Cause-and-effect relationships
        - Common pitfalls or misconceptions to avoid

        6. Quality Standards:
        Each flashcard must:
        - Be self-contained and independently comprehensible
        - Use clear, jargon-free language
        - Break down complex ideas into digestible parts
        - Include both theoretical understanding and practical application
        - Maintain technical accuracy while being accessible

        Provide the flashcards in the following JSON format without any code fences, backticks, or additional text:

        [
        {
        "question": "Clear, focused question",
        "answer": "Direct Answer:
        A clear, initial explanation of the concept.

        Analogy:
        A relatable real-world comparison that makes the concept easier to understand.

        Detailed Explanation:
        A comprehensive breakdown with examples and additional context.

        Practical Application:
        A concrete real-world scenario where this concept is applied."
        },
        // ... continue for all necessary concepts
        ]

        Important formatting rules:
        1. Each section of the answer (Direct Answer, Analogy, etc.) should start on a new line
        2. Include a blank line between sections for better readability
        3. Begin each section with its header (e.g., "Direct Answer:", "Analogy:")
        4. Ensure the content is properly indented within the JSON structure
        5. The answer should be one continuous string with line breaks (\n) for formatting

        Ensure your response contains only valid JSON data with no additional text or formatting.
        """

        response = self.llm.invoke(prompt + "\n\nStudy materials:\n" + state["study_materials"])
        content = response.content.strip()

        if content.startswith("```") and content.endswith("```"):
            content = "\n".join(content.split("\n")[1:-1]).strip()

        try:
            flashcards = json.loads(content)
            if not isinstance(flashcards, list):
                raise ValueError("Flashcards should be a list")
            for card in flashcards:
                if not isinstance(card, dict) or "question" not in card or "answer" not in card:
                    raise ValueError("Invalid flashcard format")
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing flashcards: {str(e)}")
            self.logger.error(f"LLM response: {response.content}")
            flashcards = []

        return {"flashcards": flashcards}

    def _reflect_on_coverage(self, state: Dict) -> Dict:
        self.logger.info("Reflecting on flashcard coverage")
        prompt = """
        You are tasked with reviewing the generated flashcards and the original study materials to ensure comprehensive coverage. Please analyze both and provide a reflection on the following:

        1. Are there any important concepts or topics from the study materials that are not covered in the flashcards?
        2. Are there any areas where the flashcards could be more detailed or provide better explanations?
        3. Are there any redundant or overly similar flashcards that could be combined or removed?
        4. Are there any complex topics that might benefit from being broken down into multiple flashcards?

        Provide your reflection in a clear, concise manner, highlighting specific areas for improvement or addition.
        """

        response = self.llm.invoke(prompt + "\n\nStudy materials:\n" + state["study_materials"] + "\n\nFlashcards:\n" + json.dumps(state["flashcards"], indent=2))
        reflection = response.content.strip()

        return {"reflection": reflection}

    def _update_flashcards(self, state: Dict) -> Dict:
        self.logger.info("Updating flashcards based on reflection")
        prompt = """
        Based on the original study materials, the initially generated flashcards, and the reflection provided, please update the flashcards to ensure comprehensive coverage of all concepts. You should:

        1. Add new flashcards for any missing concepts identified in the reflection.
        2. Modify existing flashcards to provide more detailed explanations where needed.
        3. Remove or combine any redundant flashcards.
        4. Break down complex topics into multiple flashcards if necessary.

        Most importantly, ensure that each flashcard strictly follows this format:

        {
        "question": "Clear, focused question",
        "answer": "Direct Answer:
        A clear, initial explanation of the concept.

        Analogy:
        A relatable real-world comparison that makes the concept easier to understand.

        Detailed Explanation:
        A comprehensive breakdown with examples and additional context.

        Practical Application:
        A concrete real-world scenario where this concept is applied."
        }

        Remember to:
        - Include all four sections (Direct Answer, Analogy, Detailed Explanation, Practical Application) for each flashcard.
        - Ensure each section starts with its header (e.g., "Direct Answer:", "Analogy:") on a new line.
        - Include a blank line between sections for better readability.
        - Make the answer a continuous string with line breaks (\n) for formatting.
        - Use clear, jargon-free language and break down complex ideas.
        - Include creative, memorable analogies related to common experiences.
        - Provide a mix of question types (definition, process, reasoning, comparison, application, problem-solving).
        - Ensure each flashcard is self-contained and independently comprehensible.

        Provide the updated flashcards in valid JSON format without any additional text or formatting.
        """

        response = self.llm.invoke(prompt + "\n\nStudy materials:\n" + state["study_materials"] + 
                                "\n\nInitial Flashcards:\n" + json.dumps(state["flashcards"], indent=2) + 
                                "\n\nReflection:\n" + state["reflection"])
    
        content = response.content.strip()
        if content.startswith("```") and content.endswith("```"):
            content = "\n".join(content.split("\n")[1:-1]).strip()

        try:
            updated_flashcards = json.loads(content)
            if not isinstance(updated_flashcards, list):
                raise ValueError("Updated flashcards should be a list")
            for card in updated_flashcards:
                if not isinstance(card, dict) or "question" not in card or "answer" not in card:
                    raise ValueError("Invalid flashcard format")
                # Check if the answer contains all required sections
                required_sections = ["Direct Answer:", "Analogy:", "Detailed Explanation:", "Practical Application:"]
                if not all(section in card["answer"] for section in required_sections):
                    raise ValueError(f"Flashcard missing required sections: {card['question']}")
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing updated flashcards: {str(e)}")
            self.logger.error(f"LLM response: {response.content}")
            updated_flashcards = state["flashcards"]  # Fallback to original flashcards

        return {"flashcards": updated_flashcards, "end": True}

    def generate_flashcards(self, study_materials: str, file_name: str) -> Dict:
        if not study_materials or not isinstance(study_materials, str):
            raise ValueError("Invalid study materials provided")
        try:
            self.logger.info("Starting flashcard generation")
            initial_state = {
                "study_materials": study_materials,
                "flashcards": [],
                "reflection": "",
                "end": False,
            }
            self.logger.info(f"Initial state: {initial_state}")
            result = self.workflow.invoke(initial_state)
            self.logger.info(f"Workflow result: {result}")
            self.logger.info("Flashcard generation completed successfully")

            file_path = self._write_results_to_file(result.get("flashcards", []), file_name)

            return {"flashcards": result.get("flashcards", []), "file_path": file_path}
        except Exception as e:
            self.logger.error(f"An error occurred during flashcard generation: {str(e)}", exc_info=True)
            return {"flashcards": [], "error": str(e)}

    def _write_results_to_file(self, flashcards: list, file_name: str):
        base_name = os.path.splitext(file_name)[0]
        file_path = os.path.abspath(f"{base_name}_flashcards.txt")
        try:
            with open(file_path, "w") as f:
                for i, card in enumerate(flashcards, 1):
                    f.write(f"{i}. Question: {card['question']}\n")
                    f.write(f"   Answer: {card['answer']}\n\n")
            self.logger.info(f"Flashcards written to {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"Error writing flashcards to file: {str(e)}")
            return ""