# genie-chipdb

芯片資料庫:ChromaDB 向量檢索 + 問答 Web 介面 + MCP server(供 Claude 等 AI 工具接入)。可匯入 genie-meeting 報告、PDF datasheet、純文字。

## 需求

```bash
pip install -e .        # chromadb + flask + genie-core
```

LM Studio:文字模型(問答)、vision 模型(PDF 匯入時逐頁解析)。

## 用法

### 匯入資料

```bash
genie-chipdb ingest datasheet.pdf --type pdf --description "XX 芯片手冊"
genie-chipdb ingest report.json  --type meeting        # genie-meeting 的 report.json
genie-chipdb ingest notes.txt    --type text
```

共通參數:`--data-dir ./chroma_data`(向量庫位置)、`--url`(LM Studio)。
PDF 匯入失敗的頁會列出頁碼(vision 解析失敗不會靜默跳過)。

### 問答

```bash
genie-chipdb ask 這顆芯片的工作溫度範圍是多少
```

### Web 介面

```bash
genie-chipdb serve --port 5100        # 預設只綁 127.0.0.1
```

瀏覽器開 `http://127.0.0.1:5100`:問答 + 來源引用(檔名 + 頁碼)。

### MCP server(stdio)

```bash
genie-chipdb mcp --data-dir ./chroma_data
```

Claude Desktop / Claude Code 設定範例:

```json
{"mcpServers": {"chipdb": {"command": "genie-chipdb", "args": ["mcp", "--data-dir", "/path/chroma_data"]}}}
```

工具:`chip_search`(向量檢索)、`chip_ask`(RAG 問答)、`chip_ingest`(匯入文字)。

## Embedding 選擇

預設用 ChromaDB 內建 embedding(英文 MiniLM)。**中文內容建議改 LM Studio embedding**(程式介面 `ChipDatabase(embedding="lmstudio", embedding_model="...")`,配 bge-m3 等多語模型)。**注意:換 embedding 必須刪掉 data-dir 重建重灌**,兩種向量空間不相容。

## 安全

- server 預設綁 127.0.0.1;path 型 ingest 有目錄白名單(預設家目錄內)
- 需要區網存取時再用 `--host`,並自行評估(API 無認證)
