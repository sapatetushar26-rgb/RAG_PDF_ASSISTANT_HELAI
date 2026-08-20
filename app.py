import os
import shutil
import tempfile

import streamlit as st

from rag import (
    load_pdf,
    split_documents,
    create_vectorstore,
    search_documents,
    generate_answer,
)

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="HelAI · Chat with your PDFs",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

CHROMA_DIR = "./chroma_db"


# --------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------

if "files" not in st.session_state:
    st.session_state.files = {}

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------

st.title("✨ HelAI")
st.caption("🤖 Upload your PDFs and chat with them — powered by RAG 📄💬")

st.divider()


# --------------------------------------------------------------------------
# SIDEBAR — UPLOAD & MANAGE DOCUMENTS
# --------------------------------------------------------------------------

with st.sidebar:

    st.header("📚 Your Documents")

    uploaded = st.file_uploader(
        "Upload your PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    # Store uploaded files
    if uploaded:
        for file in uploaded:
            st.session_state.files[file.name] = file.getvalue()

    # ----------------------------------------------------------------------
    # SHOW UPLOADED FILES
    # ----------------------------------------------------------------------

    if st.session_state.files:

        st.caption(
            f"{len(st.session_state.files)} "
            f"file(s) ready"
        )

        for filename in st.session_state.files:

            if filename in st.session_state.processed_files:
                st.success(
                    f"✅ {filename}",
                    icon="📄",
                )
            else:
                st.info(
                    f"🕓 {filename}",
                    icon="📄",
                )

    else:
        st.info("Upload one or more PDFs to get started.")


    st.divider()


    # ----------------------------------------------------------------------
    # PROCESS / RESET BUTTONS
    # ----------------------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        process_clicked = st.button(
            "⚙️ Process",
            use_container_width=True,
        )

    with col2:
        reset_clicked = st.button(
            "🗑️ Reset",
            use_container_width=True,
        )


    # ----------------------------------------------------------------------
    # RESET
    # ----------------------------------------------------------------------

    if reset_clicked:

        shutil.rmtree(
            CHROMA_DIR,
            ignore_errors=True,
        )

        st.session_state.files = {}
        st.session_state.processed_files = []
        st.session_state.vectorstore = None
        st.session_state.chat_history = []

        st.rerun()


    # ----------------------------------------------------------------------
    # PROCESS PDFS
    # ----------------------------------------------------------------------

    if process_clicked:

        if not st.session_state.files:

            st.warning(
                "Please upload at least one PDF first. 📎"
            )

        else:

            with st.spinner(
                "Reading and indexing your PDFs... 🧠"
            ):

                all_chunks = []

                with tempfile.TemporaryDirectory() as tmp_dir:

                    for filename, data in st.session_state.files.items():

                        tmp_path = os.path.join(
                            tmp_dir,
                            filename,
                        )

                        with open(
                            tmp_path,
                            "wb",
                        ) as file:

                            file.write(data)

                        documents = load_pdf(
                            tmp_path,
                            filename,
                        )

                        chunks = split_documents(
                            documents
                        )

                        all_chunks.extend(chunks)


                # Re-create vector database
                shutil.rmtree(
                    CHROMA_DIR,
                    ignore_errors=True,
                )

                st.session_state.vectorstore = (
                    create_vectorstore(all_chunks)
                )

                st.session_state.processed_files = list(
                    st.session_state.files.keys()
                )

            st.success(
                "PDFs indexed successfully! 🎉"
            )


    st.divider()


    # ----------------------------------------------------------------------
    # SEARCH SCOPE
    # ----------------------------------------------------------------------

    st.header("🔎 Search Scope")

    if st.session_state.processed_files:

        selected_pdfs = st.multiselect(
            "Which PDFs should I look into?",
            options=st.session_state.processed_files,
            default=st.session_state.processed_files,
        )

    else:

        selected_pdfs = []

        st.info(
            "Process your PDFs to enable search scope. 👆"
        )


    st.divider()


    # ----------------------------------------------------------------------
    # API STATUS
    # ----------------------------------------------------------------------

    st.header("🔑 API Status")

    api_key_present = bool(
        os.getenv("OPENROUTER_API_KEY")
    )

    if api_key_present:

        st.success(
            "🟢 API key detected"
        )

    else:

        st.warning(
            "🔴 No API key found in environment"
        )


# --------------------------------------------------------------------------
# MAIN CHAT AREA
# --------------------------------------------------------------------------

if not st.session_state.processed_files:

    # ----------------------------------------------------------------------
    # EMPTY STATE
    # ----------------------------------------------------------------------

    st.info(
        """
        ### 🗂️✨ No PDFs indexed yet

        Upload your PDFs using the sidebar and click
        **⚙️ Process** to start chatting with your documents.
        """
    )

    st.markdown(
        """
        #### How it works

        1. 📎 Upload one or more PDF files
        2. ⚙️ Click **Process**
        3. 🔎 Choose which PDFs to search
        4. 💬 Ask questions about your documents
        """
    )


else:

    # ----------------------------------------------------------------------
    # CHAT HISTORY
    # ----------------------------------------------------------------------

    for message in st.session_state.chat_history:

        if message["role"] == "user":

            with st.chat_message(
                "user",
                avatar="🧑‍💻",
            ):

                st.markdown(
                    message["content"]
                )

        else:

            with st.chat_message(
                "assistant",
                avatar="🤖",
            ):

                st.markdown(
                    message["content"]
                )

                sources = message.get(
                    "sources",
                    [],
                )

                if sources:

                    with st.expander(
                        "📎 Sources"
                    ):

                        for source in sources:

                            st.write(
                                f"📄 {source}"
                            )


    # ----------------------------------------------------------------------
    # CHAT INPUT
    # ----------------------------------------------------------------------

    question = st.chat_input(
        "Ask something about your PDFs..."
    )


    # ----------------------------------------------------------------------
    # PROCESS QUESTION
    # ----------------------------------------------------------------------

    if question:

        # --------------------------------------------------------------
        # USER MESSAGE
        # --------------------------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message(
            "user",
            avatar="🧑‍💻",
        ):

            st.markdown(question)


        # --------------------------------------------------------------
        # ASSISTANT MESSAGE
        # --------------------------------------------------------------

        with st.chat_message(
            "assistant",
            avatar="🤖",
        ):

            if not selected_pdfs:

                answer = (
                    "⚠️ Please select at least one PDF "
                    "in the sidebar to search."
                )

                sources = []

                st.warning(answer)

            else:

                with st.spinner(
                    "Thinking... 💭"
                ):

                    try:

                        results = search_documents(
                            st.session_state.vectorstore,
                            question,
                            selected_pdfs,
                        )

                        answer = generate_answer(
                            question,
                            results,
                        )

                        sources = sorted(
                            {
                                doc.metadata.get(
                                    "source",
                                    "unknown",
                                )
                                for doc in results
                            }
                        )

                    except Exception as error:

                        answer = (
                            "❌ Something went wrong while "
                            "answering your question."
                        )

                        sources = []

                        st.error(
                            f"Error: {error}"
                        )

                st.markdown(answer)

                if sources:

                    with st.expander(
                        "📎 Sources"
                    ):

                        for source in sources:

                            st.write(
                                f"📄 {source}"
                            )


        # --------------------------------------------------------------
        # SAVE ASSISTANT RESPONSE
        # --------------------------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )