# Social Assistant - 設計文件

> ARTOGO 社群自動化助手
> 建立日期：2026-02-10

---

## 概覽

Social Assistant 是 ARTOGO 的社群媒體自動化工具，初期聚焦 Threads 平台。Bot 自動從多個來源抓取展覽與導覽的討論熱點，經 AI 分析篩選後生成符合品牌人設的貼文與留言，透過 Slack 人工審核後自動發布。

### 品牌定位

**藝術愛好者社群** — 像一個懂藝術的朋友，用輕鬆口吻聊展覽、分享心得、參與討論，拉近與大眾的距離。

### 核心流程

```
採集 → 分析 → 生成 → 審核（Slack）→ 排程發布 → 成效追蹤
```

### 技術棧

- **語言：** Python
- **AI：** 可換式 LLM Provider（Claude / OpenAI / 可擴充）
- **審核：** Slack Bot（Slack Bolt for Python）
- **發布：** Meta Threads API
- **資料庫：** SQLite
- **排程：** APScheduler
- **套件管理：** uv

---

## 模組一：內容採集引擎（Collector）

### 職責

定時從多個來源抓取展覽 / 導覽相關的討論和資訊。

### 資料來源（依優先順序）

1. **Threads** — 搜尋藝文關鍵字（展覽、美術館、導覽、藝術家名等），追蹤熱門藝文帳號的新貼文
2. **藝文新聞網站** — ARTouch（典藏）、ARTPRESS、非池中等台灣主要藝文媒體
3. **官方資訊** — 文化部展覽資訊、各大美術館 / 博物館官網近期展覽公告
4. **ARTOGO 平台** — 自有平台的熱門展覽、新上架展覽、用戶評論趨勢

### 技術實作

- **Threads API**（Meta Graph API）讀取公開貼文與搜尋
- **httpx + BeautifulSoup** 爬取藝文新聞網站
- **RSS / API** 接文化部或博物館的公開資訊
- 所有採集結果統一存入 **SQLite** 資料庫
- **APScheduler** 做定時排程（每小時 / 每日可配置）

### 資料結構

每筆採集到的內容存為一個 `ContentItem`：

| 欄位 | 類型 | 說明 |
|------|------|------|
| id | str (UUID) | 唯一識別碼 |
| source | str | 來源平台 (threads / news / official / artogo) |
| url | str | 原始 URL |
| title | str | 標題 |
| summary | str | 摘要 |
| author | str | 作者 |
| published_at | datetime | 發布時間 |
| likes | int | 讚數 |
| replies | int | 回覆數 |
| tags | list[str] | 關鍵字標籤 |
| collected_at | datetime | 採集時間 |

---

## 模組二：內容分析與策略引擎（Analyzer）

### 職責

從採集到的內容中，評估哪些話題值得參與，以及用什麼方式參與。

### 分析流程

1. **熱度評分** — 根據互動數據（讚數、回覆數、轉發數）和時間衰減算出話題熱度
2. **相關性評分** — 用 LLM 判斷內容與 ARTOGO 品牌 / 業務的相關程度（0-10）
3. **參與決策** — 根據綜合分數決定行動類型：
   - **發布原創貼文** — 話題夠熱、ARTOGO 有獨特觀點可分享
   - **回覆留言** — 討論串活躍且 ARTOGO 能提供價值或自然互動
   - **跳過** — 相關性低或風險高的話題

### 風險過濾

- 自動排除政治敏感、爭議性、負面攻擊性的討論
- 避免過度商業推廣感的留言
- 設定每日發文 / 留言上限，避免洗版

### 輸出

每個建議行動（`ActionSuggestion`）包含：

| 欄位 | 類型 | 說明 |
|------|------|------|
| content_item_id | str | 對應的 ContentItem |
| action_type | str | post / reply / skip |
| relevance_score | float | 相關性分數 (0-10) |
| heat_score | float | 熱度分數 (0-10) |
| suggested_angle | str | 建議切入角度 |
| priority | int | 優先順序 |
| risk_flags | list[str] | 風險標記 |

---

## 模組三：內容生成引擎（Generator）

### 職責

依據品牌人設和策略建議，用 AI 生成貼文或留言草稿。

### 品牌人設配置（Persona）

存為 `config/persona.yaml`，定義 ARTOGO 在 Threads 上的人格：

```yaml
name: ARTOGO
tone: 輕鬆友善、像懂藝術的朋友
style:
  - 用口語化的方式分享藝術觀點
  - 偶爾帶點幽默感
  - 對藝術有熱情但不說教
  - 鼓勵大家親自去看展
forbidden_words:
  - 業配
  - 工商
  # ... 其他禁用詞
example_posts:
  - "最近去看了 XXX 展，那個光影的呈現真的會讓人停下腳步..."
  - "有人也覺得這次的策展動線設計得特別好嗎？"
```

### 生成類型

1. **原創貼文** — 根據熱門話題撰寫觀點分享、展覽推薦、藝文小知識
2. **回覆留言** — 根據目標討論串的上下文，生成自然的互動回應
3. **資訊型回覆** — 當有人詢問展覽資訊時，提供精確的展期、票價、地點等

### LLM Provider 抽象層

```python
class LLMProvider(Protocol):
    async def generate(self, prompt: str, context: dict, persona: dict) -> str: ...

class ClaudeProvider(LLMProvider): ...
class OpenAIProvider(LLMProvider): ...
```

- 統一介面：`generate(prompt, context, persona) -> str`
- 透過環境變數或配置檔切換 Provider

### 品質控制

- 生成後自動檢查：字數限制（Threads 500 字）、是否包含禁用詞、語氣是否符合人設
- 不通過則自動重新生成（最多 3 次）

---

## 模組四：審核系統（Reviewer）

### 職責

將 AI 生成的內容推送到 Slack，讓管理者審核後再發布。

### Slack 整合

- 建立 **Slack App**，使用 Slack Bolt for Python
- 專用頻道：`#social-review`
- 每筆待審內容以 **Block Kit 訊息** 呈現：
  - 話題摘要 + 原始來源連結
  - AI 生成的貼文/留言內容（完整預覽）
  - 行動類型標籤（發文 / 留言）
  - 三個按鈕：✅ 核准 | ✏️ 編輯 | ❌ 拒絕

### 審核流程

1. **核准** → 內容進入發布佇列，按排程發出
2. **編輯** → 彈出 Slack Modal 對話框，可修改內容後再核准
3. **拒絕** → 標記為拒絕，記錄拒絕原因（可選填），供 AI 學習改進

### 排程發布

- 核准後不會立即發布，而是進入佇列
- 可設定發文時段（例如上午 10 點、下午 3 點、晚上 8 點）
- 避免短時間內密集發文

---

## 模組五：發布引擎（Publisher）

### 職責

將審核通過的內容發布到 Threads。

### Threads API 整合

- 使用 **Meta Threads API**（基於 Instagram Graph API）
- 支援兩種操作：
  - **發布貼文** — 建立新的 Thread 貼文
  - **回覆留言** — 在指定貼文底下留言

### 發布策略

- 佇列式發布，依照設定的時段排程
- 每次發布後記錄：發布時間、貼文 ID、原始內容
- 發布失敗自動重試（最多 3 次），失敗通知到 Slack

### 成效追蹤

- 定時回查已發布內容的互動數據（讚數、回覆數）
- 存入資料庫，供後續分析哪類內容表現好
- 每週在 Slack 推送一份簡單的成效摘要

---

## 專案結構

```
ARTOGO/social-assistant/
├── README.md
├── pyproject.toml          # 專案配置（使用 uv 管理）
├── .env.example            # 環境變數範本
├── config/
│   ├── settings.yaml       # 全域設定（排程、上限等）
│   └── persona.yaml        # 品牌人設配置
├── src/
│   ├── collector/          # 內容採集引擎
│   │   ├── threads.py
│   │   ├── news.py
│   │   └── official.py
│   ├── analyzer/           # 內容分析引擎
│   │   └── strategy.py
│   ├── generator/          # 內容生成引擎
│   │   ├── llm_provider.py
│   │   └── content.py
│   ├── reviewer/           # Slack 審核系統
│   │   └── slack_bot.py
│   ├── publisher/          # Threads 發布引擎
│   │   └── threads.py
│   ├── models/             # 資料模型
│   │   └── schemas.py
│   └── db/                 # 資料庫
│       └── database.py
├── scheduler.py            # 排程主程式
└── tests/
```

### 運行方式

- `scheduler.py` 為主進程，定時觸發：採集 → 分析 → 生成 → 推送審核
- Slack Bot 常駐監聽審核回應
- 核准後排程發布

### 環境變數

```
# Threads API
THREADS_ACCESS_TOKEN=
THREADS_USER_ID=

# Slack
SLACK_BOT_TOKEN=
SLACK_SIGNING_SECRET=
SLACK_REVIEW_CHANNEL=

# LLM
LLM_PROVIDER=claude  # claude / openai
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# Database
DATABASE_URL=sqlite:///data/social_assistant.db
```
