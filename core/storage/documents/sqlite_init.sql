-- 创建文档存储表，包含 faiss 中文档的 id，文档文本，create_at，updated_at
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    meta TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
)