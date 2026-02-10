# Social Assistant

ARTOGO 的社群媒體自動化助手，自動採集藝文展覽熱點、AI 生成品牌貼文、Slack 人工審核後發布到 Threads。

## How It Works

```
採集 → 分析 → AI 生成 → Slack 審核 → 發布到 Threads
```

1. **採集** — 從 Threads、藝文新聞網站、博物館官網抓取展覽相關討論
2. **分析** — 計算話題熱度 + AI 判斷相關性 + 過濾風險內容
3. **生成** — 依品牌人設用 LLM 撰寫貼文或留言草稿
4. **審核** — 推送到 Slack，一鍵核准 / 編輯 / 拒絕
5. **發布** — 核准後自動排程發布到 Threads

## Tech Stack

- **Python 3.12+** / uv
- **LLM**: Claude / OpenAI（可切換）
- **Slack Bot**: slack-bolt（Block Kit 審核 UI）
- **Threads API**: Meta Graph API
- **Database**: SQLite（aiosqlite）
- **Scheduler**: APScheduler
- **Scraping**: httpx + BeautifulSoup4

## Quick Start

```bash
# 安裝依賴
uv sync --all-extras

# 設定環境變數
cp .env.example .env
# 編輯 .env 填入你的 API keys

# 跑測試
uv run pytest -v

# 啟動
uv run python scheduler.py
```

## Configuration

### 品牌人設 (`config/persona.yaml`)

控制 AI 生成內容的語氣和風格，不需改程式碼：

```yaml
tone: 輕鬆友善、像懂藝術的朋友
style_guidelines:
  - 用口語化的方式分享藝術觀點
  - 偶爾帶點幽默感
  - 對藝術有熱情但不說教
forbidden_words:
  - 業配
  - 工商
```

### 全域設定 (`config/settings.yaml`)

排程頻率、每日發文上限、風險關鍵字、發布時段等。

### 環境變數 (`.env`)

| 變數 | 說明 |
|------|------|
| `THREADS_ACCESS_TOKEN` | Threads API 存取令牌 |
| `THREADS_USER_ID` | Threads 用戶 ID |
| `SLACK_BOT_TOKEN` | Slack Bot Token |
| `SLACK_SIGNING_SECRET` | Slack App Signing Secret |
| `LLM_PROVIDER` | `claude` 或 `openai` |
| `ANTHROPIC_API_KEY` | Claude API Key |
| `OPENAI_API_KEY` | OpenAI API Key |

## Project Structure

```
src/
├── collector/       # 多來源內容採集（Threads API + 新聞爬蟲）
├── analyzer/        # 熱度評分 + AI 相關性分析 + 風險過濾
├── generator/       # 可換式 LLM 內容生成 + 品牌人設檢查
├── reviewer/        # Slack Block Kit 審核 UI + 事件處理
├── publisher/       # Threads 兩步驟發布
├── models/          # Pydantic 資料模型
├── db/              # async SQLite 資料層
└── pipeline.py      # 全流程編排
```

## Testing

```bash
uv run pytest -v          # 跑全部測試 (59 tests)
uv run ruff check src/    # Lint
```

## License

MIT
