import streamlit as st
from flashcard_generator_1 import FlashcardGenerator
import logging


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    st.title("Flashcard Generator")

    uploaded_file = st.file_uploader("Choose a PDF or Word file", type=["pdf", "docx"])

    if uploaded_file is not None:
        generator = FlashcardGenerator()

        with st.spinner("Processing file..."):
            try:
                study_materials = generator.read_file_contents(uploaded_file)
                logger.info(f"File processed successfully: {uploaded_file.name}")
            except Exception as e:
                logger.error(
                    f"Error processing file {uploaded_file.name}: {str(e)}",
                    exc_info=True,
                )
                st.error(f"Error processing file: {str(e)}")
                return

        if st.button("Generate Flashcards"):
            with st.spinner("Generating flashcards..."):
                try:
                    result = generator.generate_flashcards(study_materials)
                    logger.info("Flashcards generated successfully")

                    if "error" in result:
                        logger.error(
                            f"Error in flashcard generation: {result['error']}"
                        )
                        st.error(f"Error generating flashcards: {result['error']}")

                    else:
                        st.success("Flashcards generated successfully!")

                        # Display flashcards
                        for category, flashcards in result["flashcards"].items():
                            st.subheader(category)
                            for i, flashcard in enumerate(flashcards, 1):
                                with st.expander(f"Flashcard {i}"):
                                    st.write(f"Q: {flashcard['question']}")
                                    st.write(f"A: {flashcard['answer']}")

                except Exception as e:
                    logger.error(
                        f"Unexpected error during flashcard generation: {str(e)}",
                        exc_info=True,
                    )
                    st.error(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()
