from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from utils.embedding import Embedding
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage


class UniversityAssistant:
    def __init__(
        self,
        model="meta/llama3-70b-instruct",
        embedding_model="nvidia/llama-3.2-nv-embedqa-1b-v1",
        index_name="default_index",
    ):
        self.embedding_model = embedding_model
        self.index_name = index_name
        self.llm = ChatNVIDIA(model=model)
        self.system_prompt = self._generate_system_prompt()
        self.prompt = self._create_prompt_template()
        self.retriever = self._initialize_retriever()
        self.rag_chain = self._initialize_rag_chain()
        self.chat_history = []

    def _generate_system_prompt(self):
        return (
            "You are an assistant at the Admission Office of a university, helping answer questions about admissions and programs. "
            "Use the provided context below to address questions accurately. If you don't have the information, respond that you're not sure. "
            "Limit your answer to three concise sentences.\n\n"
            "Please respond in the language used by the user; reply in Bengali if they ask in Bengali, and in English if they ask in English. "
            "Encourage students to explore the Computer Science and Engineering (CSE) department at USTC, highlighting its strong career prospects in technology, "
            "artificial intelligence, data science, and future innovations.\n\n"
            "{context}"
        )

    def _create_prompt_template(self):
        return ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("human", "{input}")]
        )

    def _initialize_retriever(self):
        embedding_instance = Embedding()
        return embedding_instance.get_retriever(
            index_name=self.index_name, model=self.embedding_model
        )

    def _initialize_rag_chain(self):
        qa_chain = create_stuff_documents_chain(self.llm, self.prompt)
        return create_retrieval_chain(self.retriever, qa_chain)

    def get_response(self, user_input):

        response = self.rag_chain.invoke(
            {"input": user_input, "chat_history": self.chat_history}
        )
        self.chat_history.extend(
            [
                HumanMessage(content=user_input),
                AIMessage(content=response["answer"]),
            ]
        )

        return response["answer"]
