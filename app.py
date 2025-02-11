import os
from pathlib import Path

import fitz
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from config import DocumentProcessingConfig, PathConfig
from document_processor import DocumentProcessor
from query_engine import QueryEngine


def initialize_session():
    if "initialized" not in st.session_state:
        load_dotenv()

        path_config = PathConfig()
        doc_config = DocumentProcessingConfig()

        for dir_path in [
            path_config.data_dir,
            path_config.temp_dir,
            doc_config.cache_dir,
            doc_config.tracking_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        processor = DocumentProcessor(doc_config)

        st.session_state.path_config = path_config
        st.session_state.doc_config = doc_config
        st.session_state.processor = processor
        st.session_state.initialized = True

        if processor.process_documents(path_config.data_dir):
            st.success("New or modified documents have been processed!")
        else:
            st.info("No document changes detected.")


def display_pdf_page(pdf_path: Path, page_num: int, temp_dir: Path):
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

        img_path = temp_dir / "temp_page.png"
        pix.save(str(img_path))

        st.image(str(img_path), width=600)

        img_path.unlink()
        doc.close()
    except Exception as e:
        st.error(f"Error displaying PDF page: {str(e)}")


def get_pdf_download_link(filename: str) -> str:
    """Generate the download link for the full PDF."""
    base_url = "https://raw.githubusercontent.com/AdeptTechSolutions/haystack-rag/refs/heads/main/data/"
    return f"{base_url}{filename}"


def main():
    st.set_page_config(
        page_title="Islamic Texts",
        page_icon="🔍",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        "<h1 style='text-align: center;'>📚 Islamic Texts</h1>",
        unsafe_allow_html=True,
    )

    initialize_session()

    if not st.session_state.initialized:
        st.stop()

    st.write("")

    st.warning(
        "⚠️ Please refer to the original book in case the answer given is incorrect or incomplete due to artificial intelligence."
    )

    with st.form("query_form"):
        query = st.text_input(
            "Enter your question:",
            placeholder="Ask a question about the documents...",
        )
        submit_button = st.form_submit_button("Generate response")

    if submit_button and query:
        with st.spinner("Generating response..."):
            query_engine = QueryEngine(
                st.session_state.processor.store, st.session_state.doc_config
            )
            result = query_engine.query(query)

            st.write("#### Answer")
            st.write(result["answer"])

            st.write("")

            with st.expander("📑 Source Information", expanded=False):
                source = result["source"]
                if source:
                    st.info(
                        """
                        ```
                        Document: {}
                        Page: {}
                        Relevance Score: {:.4f}
                        ```
                        """.format(
                            source["source"], source["page"], source["score"]
                        )
                    )

                    st.info("**Context**")
                    st.markdown(f"{source['content']}")

                    st.info("**Page Preview**")

                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        pdf_path = (
                            st.session_state.path_config.data_dir / source["source"]
                        )
                        display_pdf_page(
                            pdf_path,
                            source["page"],
                            st.session_state.path_config.temp_dir,
                        )

                    download_link = get_pdf_download_link(source["source"])
                    st.markdown(f"[📥 Download PDF]({download_link})")
                else:
                    st.warning("⚠️ No specific source found for this response.")


if __name__ == "__main__":
    main()
