import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# 加载 .env 文件中的环境变量
load_dotenv()

# 定义向量数据库路径
DB_FAISS_PATH = "vector_store/"

# 自定义提问模板
# 这个模板指导 LLM 如何利用上下文（context）来回答问题（question）
custom_prompt_template = """请基于以下提供的上下文信息来回答用户的问题。
如果根据上下文无法得出答案，请直接说“根据提供的资料，我无法回答该问题”，不要尝试编造答案。

上下文: {context}
问题: {question}

只返回有用的答案，答案应尽量简洁。
有用答案:
"""


def set_custom_prompt():
    """
    设置并返回一个自定义的 PromptTemplate。
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


def load_llm():
    """
    加载通过 API 调用的 LLM。
    这里使用 ChatOpenAI 类，它兼容所有遵循 OpenAI API 格式的接口，例如 DeepSeek。
    API密钥和基础URL从环境变量中读取，以保证安全。
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE")

    if not api_key:
        print("错误：未找到 DEEPSEEK_API_KEY。")
        print("请在项目根目录下创建一个 .env 文件，并添加 DEEPSEEK_API_KEY='your_api_key'。")
        return None

    if not base_url:
        print("警告：未找到 DEEPSEEK_API_BASE，将使用默认的 OpenAI 地址。")
        print("如果使用 DeepSeek，请在 .env 文件中添加 DEEPSEEK_API_BASE='https://api.deepseek.com/v1'。")

    llm = ChatOpenAI(
        model_name="deepseek-chat",  # 使用 DeepSeek 的模型
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0.7
    )
    return llm


def retrieval_qa_chain(llm, prompt, db):
    """
    创建并返回一个检索问答链 (RetrievalQA chain)。
    这个链整合了LLM、提问模板和向量数据库。
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain


def qa_bot():
    """
    问答机器人的主函数。
    1. 加载词嵌入模型和向量数据库。
    2. 加载大语言模型 (通过API)。
    3. 创建问答链。
    4. 启动一个循环，接收用户输入并打印模型的回答。
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})

    if not os.path.exists(DB_FAISS_PATH):
        print(f"错误：向量数据库路径 '{DB_FAISS_PATH}' 不存在。")
        print("请先运行 'python ingest.py' 来创建数据库。")
        return

    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()

    if not llm:
        return  # 如果模型加载失败，则退出

    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    # 启动交互式问答循环
    print("\n\033[94m你好！我是你的论文问答助手。输入 'exit' 来退出程序。\033[0m")
    while True:
        query = input("\033[92m请输入你的问题: \033[0m")
        if query.lower() == 'exit':
            break

        print("\033[93m正在思考...\033[0m")
        # 将问题传递给问答链并获取结果
        result = qa.invoke({'query': query})
        print("\n\033[96m答案:\033[0m", result['result'])
        # print("\n\033[95m来源文档:\033[0m", result['source_documents']) # 可选：打印来源文档以供调试


if __name__ == "__main__":
    qa_bot()

