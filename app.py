import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 加载 .env 文件中的环境变量
load_dotenv()

# 定义向量数据库路径
DB_FAISS_PATH = "vector_store/"

# 自定义提问模板
custom_prompt_template = """你是一个专业的学术助手，请严格基于以下提供的上下文信息回答问题。

上下文:
{context}

问题:
{question}

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
        if not api_key:
            print("错误: 未找到 ZHIPUAI_API_KEY。请在 .env 文件中设置。")
            return None
        print("正在加载 ZhipuAI 模型...")
        llm = ChatZhipuAI(
            api_key=api_key,
            model=os.getenv("ZHIPUAI_CHAT_MODEL", "glm-4.5-flash"),
            temperature=0.7,
        )
    else:  # 默认为 deepseek
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_API_BASE")
        if not api_key or not base_url:
            print("错误：未找到 DeepSeek API 密钥或基础 URL。请检查 .env 文件。")
            return None
        print("正在加载 DeepSeek 模型...")
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
        print(f"加载词嵌入模型时出错: {e}")
        return

    if not os.path.exists(DB_FAISS_PATH):
        print(f"错误：向量数据库路径 '{DB_FAISS_PATH}' 不存在。")
        print("请先运行 'python ingest.py' 来创建数据库。")
        return

    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()

    if not llm:
        return

    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    print("\n\033[94m你好！我是你的论文问答助手。输入 'exit' 来退出程序。\033[0m")
    while True:
        query = input("\033[92m请输入你的问题: \033[0m")
        if query.lower() == 'exit':
            break

        print("\033[93m正在思考...\033[0m")
        result = qa.invoke({'query': query})

        # === 新增：打印检索到的上下文（用于调试）===
        print("\n\033[95m🔍 检索到的相关上下文:\033[0m")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"\n--- 片段 {i} ---")
            print(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
            # 可选：打印来源文件名
            if "source" in doc.metadata:
                print(f"📄 来源: {doc.metadata['source']}")
        print("\n" + "=" * 60)

        print("\n\033[96m答案:\033[0m", result['result'])


if __name__ == "__main__":
    qa_bot()

