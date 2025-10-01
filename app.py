import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import textwrap
import shutil
# 加载 .env 文件中的环境变量
load_dotenv()

# 定义向量数据库路径
DB_FAISS_PATH = "vector_store/"

# 自定义提问模板
# custom_prompt_template = """你是一位熟悉学术论文的研究助理，请根据以下上下文回答问题。
#
# - 如果上下文**包含相关信息**（如标题、摘要、方法、结论中的关键词或描述），请用**自然、简洁的中文**总结回答，可以适当组织语言，但**不要编造细节**。
# - 如果上下文**完全不涉及**问题主题，请回答：“根据提供的资料，我无法回答该问题”。注意！这是下下策！尽量不要这么回答！
# - 回答应像人类学者写的：**避免机械重复**，**不要用“根据上下文”开头**，**不可以使用加粗、项目符号等 Markdown 符号**，不要过度格式化。
# - 保持专业但口语化，例如：“论文提出了一种新方法……” 而不是 “该研究采用了……”。
#
# 上下文:
# {context}
#
# 问题:
# {question}
#
# 回答：
# """
custom_prompt_template = """You are a research assistant familiar with academic papers. Please answer the following question based on the context below in English.

- If the context **contains relevant information** (such as keywords or descriptions in the title, abstract, methods, or conclusions), summarize the answer in **natural, concise English**, organizing the language appropriately but **do not fabricate details**.
- If the context **does not relate** to the question at all, respond: “Based on the provided information, I cannot answer this question”. Note: this is a last resort! Try not to use it if possible.
- Answers should read like a human scholar’s writing: **avoid mechanical repetition**, **do not start with “According to the text”**, **do not use bold, bullet points, or other Markdown symbols**, and avoid excessive formatting.
- Maintain professionalism but keep it conversational, e.g., “The paper proposes a new method…” instead of “This study adopts…”.

Context:
{context}

Question:
{question}

Answer:
"""


def set_custom_prompt():
    """设置并返回一个自定义的 PromptTemplate。"""
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


def load_llm():
    """根据环境变量加载指定的 LLM。"""
    provider = os.getenv("LLM_PROVIDER", "deepseek").lower()

    if provider == "zhipuai":
        api_key = os.getenv("ZHIPUAI_API_KEY")
        # print("正在加载 ZhipuAI 模型...")
        print("Loading ZhipuAI model...")
        llm = ChatZhipuAI(
            api_key=api_key,
            model=os.getenv("ZHIPUAI_CHAT_MODEL", "glm-4.5-flash"),
            temperature=0.7,
        )
    else:  # 默认为 deepseek
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_API_BASE")
        # print("正在加载 DeepSeek 模型...")
        print("Loading DeepSeek model...")
        llm = ChatOpenAI(
            model_name=os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat"),
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.7
        )
    return llm


def get_embeddings():
    """根据环境变量获取指定的 embedding function。"""
    provider = os.getenv("LLM_PROVIDER", "deepseek").lower()

    if provider == "zhipuai":
        return ZhipuAIEmbeddings(
            api_key=os.getenv("ZHIPUAI_API_KEY"),
            model=os.getenv("ZHIPUAI_EMBEDDING_MODEL", "embedding-3")
        )
    else:
        return OpenAIEmbeddings(
            model=os.getenv("DEEPSEEK_EMBEDDING_MODEL", "text-embedding-v2"),
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_API_BASE")
        )


def retrieval_qa_chain(llm, prompt, db):
    """创建并返回一个检索问答链。"""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        # retriever=db.as_retriever(search_kwargs={'k': 2}),
        retriever=db.as_retriever(search_kwargs={'k': 6}),  # 试试 4~8
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain


def qa_bot():
    """问答机器人的主函数（带上下文调试输出）"""
    try:
        embeddings = get_embeddings()
    except Exception as e:
        # print(f"加载词嵌入模型时出错: {e}")
        print(f"Error loading embedding model: {e}")
        return

    if not os.path.exists(DB_FAISS_PATH):
        # print(f"错误：向量数据库路径 '{DB_FAISS_PATH}' 不存在。")
        # print("请先运行 'python ingest.py' 来创建数据库。")
        print(f"Error: Vector database path '{DB_FAISS_PATH}' does not exist.")
        print("Please run 'python ingest.py' to create the database first.")
        return

    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()

    if not llm:
        return

    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    # print("\n\033[94m你好！我是你的论文问答助手。输入 'exit' 来退出程序。\033[0m")
    print("\n\033[94mHello! I am your academic paper Q&A assistant. Type 'exit' to quit the program.\033[0m")
    while True:
        # query = input("\033[92m请输入你的问题: \033[0m")
        query = input("\033[92mPlease enter your question: \033[0m")
        if query.lower() == 'exit':
            break

        # print("\033[93m正在思考...\033[0m")
        print("\033[93mThinking...\033[0m")
        result = qa.invoke({'query': query})

        # === 新增：打印检索到的上下文（用于调试）===
        # print("\n\033[95m🔍 检索到的相关上下文:\033[0m")
        # print("\n\033[95m🔍 Retrieved relevant context:\033[0m")
        # for i, doc in enumerate(result["source_documents"], 1):
        #     # print(f"\n--- 片段 {i} ---")
        #     print(f"\n--- Chunk {i} ---")
        #     print(doc.page_content[:50] + "..." if len(doc.page_content) > 50 else doc.page_content)
        #     if "source" in doc.metadata:
        #         # print(f"📄 来源: {doc.metadata['source']}")
        #         print(f"📄 Source: {doc.metadata['source']}")
        print("\n" + "=" * 60)

        # 获取终端宽度（默认 80）
        terminal_width = shutil.get_terminal_size().columns

        # 对答案进行自动换行（保留原有换行符）
        wrapped_answer = textwrap.fill(
            result['result'],
            width=terminal_width - 10,  # 留点边距
            replace_whitespace=False,  # 保留原有空格
            break_long_words=False,  # 不拆单词
            break_on_hyphens=False
        )

        # print(f"\n\033[94m答案:\033[0m")
        print(f"\n\033[94mAnswer:\033[0m")
        print(f"\033[94m{wrapped_answer}\033[0m")


if __name__ == "__main__":
    qa_bot()

