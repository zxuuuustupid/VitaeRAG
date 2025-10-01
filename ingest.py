import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 定义数据路径和向量存储路径
DATA_PATH = "data/"
DB_FAISS_PATH = "vector_store/"

def create_vector_db():
    """
    创建向量数据库函数：
    1. 从'data/'目录加载PDF文件。
    2. 将加载的文档分割成小块。
    3. 使用预训练的词嵌入模型将文本块转换为向量。
    4. 使用FAISS（Facebook AI Similarity Search）存储这些向量。
    5. 将构建好的向量数据库保存到本地。
    """
    # 使用 DirectoryLoader 加载 'data/' 目录中的所有 PDF 文件
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"成功加载 {len(documents)} 份PDF文档。")

    # 使用 RecursiveCharacterTextSplitter 对文档进行分块
    # chunk_size 定义了每个文本块的最大字符数
    # chunk_overlap 定义了相邻文本块之间的重叠字符数，以保证语义连续性
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"文档被分割成 {len(texts)} 个文本块。")

    # 使用 HuggingFace 的预训练模型进行文本嵌入
    # all-MiniLM-L6-v2 是一个轻量且高效的模型，适合入门
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    print("正在加载词嵌入模型...")

    # 使用 FAISS 从文本块和它们的嵌入向量创建向量数据库
    print("正在创建并嵌入向量...")
    db = FAISS.from_documents(texts, embeddings)

    # 将创建好的向量数据库保存到指定的本地路径
    db.save_local(DB_FAISS_PATH)
    print(f"向量数据库成功创建并保存于 '{DB_FAISS_PATH}'")

if __name__ == "__main__":
    # 当该脚本被直接运行时，调用 create_vector_db 函数
    create_vector_db()
