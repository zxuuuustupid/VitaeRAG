import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 定义数据路径和向量存储路径
DATA_PATH = "data/"
DB_FAISS_PATH = "vector_store/"


def create_vector_db():
    """
    创建向量数据库函数：
    1. 从'data/'目录加载PDF文件。
    2. 将加载的文档分割成小块。
    3. 根据环境变量选择使用 DeepSeek 或 ZhipuAI API 将文本块转换为向量。
    4. 使用FAISS（Facebook AI Similarity Search）存储这些向量。
    5. 将构建好的向量数据库保存到本地。
    """
    # 使用 DirectoryLoader 加载 'data/' 目录中的所有 PDF 文件
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"成功加载 {len(documents)} 份PDF文档。")

    # 使用 RecursiveCharacterTextSplitter 对文档进行分块
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"文档被分割成 {len(texts)} 个文本块。")

    # 根据环境变量选择使用哪个提供商的API
    provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
    print(f"正在使用提供商: {provider}")

    if provider == "zhipuai":
        zhipu_api_key = os.getenv("ZHIPUAI_API_KEY")
        if not zhipu_api_key:
            raise ValueError("错误: LLM_PROVIDER 设置为 'zhipuai', 但未找到 ZHIPUAI_API_KEY。请在 .env 文件中设置。")
        print("正在通过 ZhipuAI API 连接并加载词嵌入模型...")
        embeddings = ZhipuAIEmbeddings(
            api_key=zhipu_api_key,
            model=os.getenv("ZHIPUAI_EMBEDDING_MODEL", "embedding-2")
        )
    else:  # 默认为 deepseek
        if not all([os.getenv("DEEPSEEK_API_KEY"), os.getenv("DEEPSEEK_API_BASE")]):
            raise ValueError("错误: LLM_PROVIDER 设置为 'deepseek' (或未设置), 但环境变量缺失。请检查 .env 文件。")
        print("正在通过 DeepSeek API 连接并加载词嵌入模型...")
        embeddings = OpenAIEmbeddings(
            model=os.getenv("DEEPSEEK_EMBEDDING_MODEL", "text-embedding-v2"),
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
            chunk_size=16  # 这是向API发送请求时，每批处理的文本块数量
        )

    # 使用 FAISS 从文本块和它们的嵌入向量创建向量数据库
    print("正在创建并嵌入向量（此过程可能需要一些时间，具体取决于文档数量）...")
    db = FAISS.from_documents(texts, embeddings)

    # 将创建好的向量数据库保存到指定的本地路径
    db.save_local(DB_FAISS_PATH)
    print(f"向量数据库成功创建并保存于 '{DB_FAISS_PATH}'")


if __name__ == "__main__":
    try:
        create_vector_db()
    except ValueError as e:
        print(e)