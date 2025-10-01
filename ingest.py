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
    创建向量数据库函数（支持 ZhipuAI 分批嵌入）
    """
    # 加载 PDF
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"成功加载 {len(documents)} 份PDF文档。")

    # from collections import defaultdict
    # page_count = defaultdict(int)
    # for doc in documents:
    #     source = doc.metadata.get("source", "unknown")
    #     page_count[source] += 1
    #
    # for source, count in page_count.items():
    #     print(f"📄 {source}: 共 {count} 页")
    # # 调试：打印第一篇 PDF 的前 500 字
    # if documents:
    #     print("\n【调试】第一篇 PDF 内容：")
    #     print(documents[0].page_content)
    #     print("\n" + "=" * 60)

    # 分割文本
    # 分割文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"文档被分割成 {len(texts)} 个文本块。")

    # === 新增：过滤低质量/无意义的文本块 ===
    def is_low_quality(text: str) -> bool:
        """判断文本块是否为低质量内容（如封面、声明、参考文献等）"""
        text_clean = text.strip()
        if len(text_clean) < 80:  # 太短的块通常无意义
            return True

        text_lower = text_clean.lower()
        # 常见噪声关键词（中英文）
        noise_keywords = [
            "e-mail", "email", "corresponding author", "first author",
            "manuscript draft", "accepted manuscript",
            "declaration of interest", "conflict of interest",
            "editorial manager", "produxiion manager", "aries systems",
            "reference", "references", "bibliography",
            "figure", "fig.", "table", "doi:", "http", "www.",
            "cover letter", "submission", "reviewer", "editor",
            "copyright", "all rights reserved", "abstract"  # 注意：有些摘要很短，慎用
        ]

        # 如果包含多个噪声关键词，很可能是噪声页
        match_count = sum(1 for kw in noise_keywords if kw in text_lower)
        if match_count >= 2:
            return True

        # 特殊规则：如果包含邮箱但文本很短
        if "@" in text_clean and len(text_clean.split()) < 15:
            return True

        return False

    # 执行过滤
    original_count = len(texts)
    texts = [doc for doc in texts if not is_low_quality(doc.page_content)]
    print(f"过滤低质量文本块后，剩余 {len(texts)} 个有效文本块（原 {original_count} 个）。")



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
        # ===== 手动分批嵌入（ZhipuAI 限制每批 ≤64）=====
        batch_size = 64
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = embeddings.embed_documents([doc.page_content for doc in batch])
            all_embeddings.extend(batch_embeddings)
            print(f"已嵌入 {min(i + batch_size, len(texts))} / {len(texts)} 个文本块")

        # 使用 FAISS.from_embeddings 手动构建
        db = FAISS.from_embeddings(
            text_embeddings=zip([doc.page_content for doc in texts], all_embeddings),
            embedding=embeddings,
            metadatas=[doc.metadata for doc in texts]
        )

    else:  # deepseek 或其他 OpenAI 兼容 API
        if not all([os.getenv("DEEPSEEK_API_KEY"), os.getenv("DEEPSEEK_API_BASE")]):
            raise ValueError("错误: LLM_PROVIDER 设置为 'deepseek' (或未设置), 但环境变量缺失。请检查 .env 文件。")
        print("正在通过 DeepSeek API 连接并加载词嵌入模型...")
        embeddings = OpenAIEmbeddings(
            model=os.getenv("DEEPSEEK_EMBEDDING_MODEL", "text-embedding-v2"),
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
            chunk_size=16  # OpenAIEmbeddings 支持自动分批
        )
        # OpenAIEmbeddings 已内置分批，可直接使用 from_documents
        db = FAISS.from_documents(texts, embeddings)

    # 保存数据库
    db.save_local(DB_FAISS_PATH)
    print(f"向量数据库成功创建并保存于 '{DB_FAISS_PATH}'")


if __name__ == "__main__":
    try:
        create_vector_db()
    except ValueError as e:
        print(e)