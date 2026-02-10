# CLAUDE.md - Social Assistant

## 專案概述

ARTOGO 社群媒體自動化助手，初期聚焦 Threads 平台。Bot 自動採集展覽與導覽的討論熱點，AI 生成符合品牌人設的內容，透過 Slack 人工審核後自動發布到 Threads。

品牌定位：**藝術愛好者社群** — 像一個懂藝術的朋友，用輕鬆口吻聊展覽。

## 技術架構

- **語言**：Python 3.12+
- **套件管理**：uv
- **AI**：可換式 LLM Provider（Claude / OpenAI）
- **審核**：Slack Bot（slack-bolt）
- **發布**：Meta Threads API
- **資料庫**：SQLite（aiosqlite）
- **排程**：APScheduler
- **爬蟲**：httpx + BeautifulSoup4

## 專案結構

```
social-assistant/
├── config/
│   ├── settings.yaml       # 排程、關鍵字、限制等全域設定
│   └── persona.yaml        # 品牌人設（語氣、禁用詞、範例貼文）
├── src/
│   ├── collector/           # 內容採集
│   │   ├── base.py          # BaseCollector ABC
│   │   ├── news.py          # 藝文新聞網站爬蟲
│   │   └── threads.py       # Threads API 採集
│   ├── analyzer/
│   │   └── strategy.py      # 熱度評分 + LLM 相關性 + 風險過濾
│   ├── generator/
│   │   ├── llm_provider.py  # LLM 抽象層（Claude / OpenAI 可換）
│   │   └── content.py       # 貼文 / 留言生成 + 品牌人設檢查
│   ├── reviewer/
│   │   ├── slack_bot.py     # Slack Block Kit UI（核准/編輯/拒絕）
│   │   └── handlers.py      # Slack 事件處理（approve/reject/edit）
│   ├── publisher/
│   │   └── threads.py       # Threads 兩步驟發布（container → publish）
│   ├── models/
│   │   └── schemas.py       # Pydantic 資料模型
│   ├── db/
│   │   └── database.py      # async SQLite CRUD
│   ├── config.py            # YAML 配置載入
│   └── pipeline.py          # 全流程編排：採集 → 分析 → 生成 → 審核
├── scheduler.py             # 主程式入口（APScheduler）
├── tests/                   # 59 tests（pytest + pytest-asyncio）
└── docs/plans/              # 設計文件與實作計畫
```

## 開發指令

```bash
uv sync --all-extras     # 安裝所有依賴
uv run pytest -v         # 跑測試
uv run ruff check src/   # Lint 檢查
uv run python scheduler.py  # 啟動排程（需先配置 .env）
```

## 環境變數

必要的環境變數（見 `.env.example`）：
- `THREADS_ACCESS_TOKEN` - Threads API 存取令牌
- `THREADS_USER_ID` - Threads 用戶 ID
- `SLACK_BOT_TOKEN` - Slack Bot Token
- `SLACK_SIGNING_SECRET` - Slack App Signing Secret
- `SLACK_REVIEW_CHANNEL` - 審核頻道（預設 `#social-review`）
- `LLM_PROVIDER` - LLM 供應商（`claude` 或 `openai`）
- `ANTHROPIC_API_KEY` - Claude API Key
- `OPENAI_API_KEY` - OpenAI API Key
- `DATABASE_PATH` - SQLite 資料庫路徑

## 核心流程

```
採集（Collector）→ 分析（Analyzer）→ 生成（Generator）→ Slack 審核（Reviewer）→ 發布（Publisher）
```

1. **採集**：定時從 Threads、藝文新聞、博物館官網抓取展覽相關內容
2. **分析**：計算熱度分數 + LLM 判斷相關性 + 風險關鍵字過濾
3. **生成**：依品牌人設（persona.yaml）用 LLM 撰寫貼文或留言草稿
4. **審核**：推送到 Slack，人工一鍵核准 / 編輯 / 拒絕
5. **發布**：核准後排程發布到 Threads

## 注意事項

- 品牌人設定義在 `config/persona.yaml`，修改語氣和風格不需改程式碼
- `config/settings.yaml` 控制排程頻率、每日發文上限、風險關鍵字等
- 切換 LLM 只需改 `.env` 的 `LLM_PROVIDER`
- 所有 API 呼叫都有 mock 測試，不需真實憑證即可開發
- Commit Message 規範：`<type>: <short summary>`

始終用中文回覆。

## Dashboard 自動更新規則

每次完成以下動作時，**必須**更新全局管理面板：

1. **開始任務** → 編輯 `/Users/walt/Desktop/Ai/DASHBOARD.md`，新增到「進行中」
2. **完成任務** → 編輯 `/Users/walt/Desktop/Ai/DASHBOARD.md`，從「進行中」移到「近期完成」，加上日期
3. **產出文件** → 編輯 `/Users/walt/Desktop/Ai/DASHBOARD.md`，在「近期 Agent 活動」新增一筆（最多保留 5 筆）
4. **專案狀態變更** → 更新「專案健康度」表格中 `ARTOGO/social-assistant` 那一行
5. **更新時間戳** → 更新 `DASHBOARD.md` 頂部的「最後更新」時間
6. **追加日誌** → 在 `/Users/walt/Desktop/Ai/.ai-management/logs/YYYY-MM-DD.md` 追加一筆
7. **更新專案狀態檔** → 編輯 `/Users/walt/Desktop/Ai/.ai-management/projects/artogo-social-assistant.md`
