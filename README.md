# USTC-RAG: Retrieval-Augmented Generation for USTC Admissions

Welcome to **USTC-RAG**, a Retrieval-Augmented Generation (RAG) application powered by Large Language Models (LLMs). This tool is designed to assist the admissions office at the University of Science and Technology of Chittagong (USTC) by providing quick and accurate information to applicants and staff. 

By combining efficient retrieval mechanisms with the generative capabilities of LLMs, USTC-RAG ensures seamless and informed interactions.

---

## Features 🚀

- **Intelligent Information Retrieval**: Extracts precise data from a knowledge base to address user queries.
- **Enhanced Generative Responses**: Combines retrieved information with LLM capabilities for comprehensive and contextual replies.
- **Streamlined Admissions Support**: Tailored to handle USTC-specific admissions queries, improving response accuracy and efficiency.
- **User-Friendly Interface**: Built with Streamlit for an intuitive and interactive experience.

---

## Getting Started

Follow the steps below to set up and run USTC-RAG on your local machine.

### Prerequisites

Ensure you have the following installed on your system:
- **Python**: Version 3.11
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/USTC-RAG.git
   cd USTC-RAG
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. Create a `.env` file in the root directory and add the following environment variables:
   ```
    LANGCHAIN_TRACING_V2=false
    NVIDIA_API_KEY=<your-api-key>
    PINECONE_API_KEY=<your-api-key>
   ```

### Running the Application

To launch the application:
   ```bash
   streamlit run App.py
   ```

The application will open in your default web browser. Use it to interact with USTC-RAG and explore its features.

