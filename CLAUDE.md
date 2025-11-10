# CLAUDE.md

> 本文件用于指导 **Claude Code** 在本仓库中的工作方式（目标、边界、流程、命令、权限）。请按需替换尖括号占位符，例如 `<REPO_NAME>`、`<OWNER>`、`<SERVICE_HOST>` 等。

---

## 1) 项目速览（Project Overview）

* **仓库**：`<OWNER>/<REPO_NAME>`
* **简介**：<一句话说明项目用途，例如：二手 iPhone 价格抓取、清洗、分析与可视化>
* **技术栈**：Django + Django REST + Channels、Celery、PostgreSQL、Redis、Docker Compose、Metabase、n8n、（前端：<ECharts/HTMX/…>）
* **核心目录**：

  * `AppleStockChecker/`（Django 应用）
  * `tasks.py` / `AppleStockChecker/tasks/`（Celery 任务）
  * `docker-compose.yml` / `compose/*.yml`
  * `scripts/`（安全白名单脚本；**Claude 只能调用此处脚本**）
  * `docs/`（文档与流程）
  * `fixtures/`（小样本数据）

---

## 2) 架构与服务（Architecture）

* **Web**：Django（ASGI，经 `daphne` 提供 Web & WebSocket），端口 `<WEB_PORT>`
* **Worker**：Celery（队列：`default`/`high`），并发 `<CELERY_CONCURRENCY>`
* **Broker/Cache**：Redis `<REDIS_HOST>:<REDIS_PORT>`
* **DB**：PostgreSQL `<DB_HOST>:<DB_PORT>`，库名 `<DB_NAME>`
* **可视化**：Metabase `<METABASE_URL>`
* **自动化**：n8n `<N8N_URL>`（Webhooks、Slack 通知等）
* **静态资源**：<S3/本地路径/…>
* **网络**：所有服务通过 Docker Compose 网络 `<COMPOSE_NETWORK>` 互联

> 参见：`docs/ARCHITECTURE_SUMMARY.md`（如不存在，可由 Claude 生成）。

---

## 3) 开发与运行（Dev/Run Commands）

> **Claude 只能通过 `scripts/` 调用命令。**

* 本地开发：

  * `scripts/dev_up.sh`：`docker compose up -d web redis db`
  * `scripts/dev_down.sh`：`docker compose down`
  * `scripts/migrate.sh`：`python manage.py makemigrations && python manage.py migrate`
  * `scripts/run_tests.sh`：运行单测（pytest）
  * `scripts/lint.sh`：ruff/black/mypy
  * `scripts/flower.sh`：启动 Flower（查看 Celery）
* 数据操作（**需小心**）：

  * `scripts/pg_dump.sh`：只读备份（生产环境禁止写操作）
  * `scripts/pg_restore.sh`：仅限本地/测试环境
* 辅助：

  * `scripts/slack_notify.sh "<msg>"`
  * `scripts/check_health.sh`

> 若命令缺失，请先由 Claude 提交脚本到 `scripts/`，并解释风险与回滚方案。

---

## 4) 分支与 PR 流程（Branching & PR Policy）

* **主线**：`main`（稳定、可部署）
* **工作分支**：

  * 命名：`feature/<short_slug>`、`fix/<short_slug>`、`chore/<short_slug>`
  * 示例：`feature/psta-overallbar-cohortbar`、`fix/celery-pg-slots`
* **更新主线**：必须 `git pull --ff-only`（或在 PyCharm 勾选 *Fast-forward only*）
* **Claude 改动**：

  1. **必须**先切到工作分支；
  2. 只在该分支提交并推送；
  3. 打开 PR → 触发 CI（lint + test）→ 人工复核 → 合并；
  4. 合并策略推荐 **Squash and merge**；或 **Rebase and merge** 维持线性历史。
* **禁止**：向 `main` 直接 push。

**PR 描述模板**：

```
### 变更内容
- <列出代码改动要点>

### 影响范围
- <影响哪些模块/服务>

### 验证步骤
- `scripts/dev_up.sh`
- `scripts/run_tests.sh`
- <手测要点>

### 回滚方案
- `git revert <merge-commit>`
- 配置回退：<ENV/Flag>
```

---

## 5) Claude 的权限与凭据（GitHub PAT）

> **优先使用 Fine-grained PAT（细粒度 Token）**，只授权到本仓库。

* **Fine-grained**（推荐）

  * **Repository access**：Only selected → `<REPO_NAME>`
  * **Permissions**：

    * Repository → **Contents: Read & write**
    * Repository → **Pull requests: Read & write**
    * Repository → Metadata: Read-only（可选）
    * （如需改工作流）Repository → **Actions: Read & write**
* **Classic**（如必须）：只勾选 `repo`；如需改 GitHub Actions 再勾 `workflow`。

**本机/服务器配置**：

```bash
# 临时：
export GITHUB_TOKEN="<YOUR_FINE_GRAINED_PAT>"
# 登录 gh 以便生态复用：
gh auth login --with-token <<< "$GITHUB_TOKEN"
# 验证：
gh auth status
```

**保护分支（GitHub Settings → Branches）**：

* Require pull request
* Require status checks (CI 必须通过)
* Restrict who can push to matching branches（禁止直推）

---

## 6) 允许 / 禁止操作（Allowlist / Blocklist）

* ✅ **允许**（仅通过 `scripts/`）：

  * 代码重构、修复 Bug、补充/调整单测与 lint 配置
  * 只读备份与只读健康检查
  * 生成/更新文档（`docs/*.md`）
* ⛔ **禁止**：

  * 改 `.env` 真实敏感值 / 提交任何密钥
  * 直连生产数据库进行写操作
  * 删除数据、修改迁移导致数据丢失的危险变更
  * 直接 push 到 `main`

> 如确需“危险操作”，必须：提出方案 → 风险评估 → 人工同意 → 只在受控环境执行并可回滚。

---

## 7) 迁移与数据（Migrations & Data Policy）

* **新增/变更模型**：生成 `migrations`；在 PR 中解释兼容性与回滚方式。
* **后向兼容优先**：避免破坏现有接口/数据路径。
* **数据回填/清洗脚本**：放入 `scripts/data_*` 并提供 dry-run；说明运行环境（仅测试/预发）。

---

## 8) CI / 质量门禁（CI & Quality Gates）

* GitHub Actions：`.github/workflows/ci.yml`

  * 步骤：`lint` → `tests` → `build`（按需）
  * PR 必须通过 CI 才可合并
* 本地校验：

  * `scripts/lint.sh`
  * `scripts/run_tests.sh`

---

## 9) 环境变量与密钥（Secrets）

* `.env` 与任何真实密钥 **不进入版本控制**；提供 `.env.sample`
* 生产密钥只存放于 `<Secret Manager / GitHub Actions Secrets / Docker Secret>`
* Claude **不得**创建/提交任何包含敏感信息的文件

---

## 10) 目标驱动用法（Goal Examples）

> 在 **当前分支** 执行（由人或 Claude）：

* **生成架构总结**：

  * 目标：阅读 `CLAUDE.md` + `docker-compose.yml` + settings，产出 `docs/ARCHITECTURE_SUMMARY.md`。
* **修 Celery 连接槽不足**：

  * 目标：分析 `OperationalError: remaining connection slots are reserved for superuser`，给出并提交参数调优（不改生产 ENV），补充文档与测试，开 PR。
* **新增统计指标**：

  * 目标：在 `psta_process_minute_bucket` 增加 `<指标名>` 计算，调试开关、单测覆盖、文档更新，开 PR。
* **Docker/Compose 标准化**：

  * 目标：拆分 dev/prod compose，增加 healthcheck 与 `make dev|prod`，更新 `docs/DEPLOY.md`，开 PR。

> 每个目标都需：变更清单 + 验证步骤 + 回滚方案。

---

## 11) 审查检查表（Review Checklist）

* [ ] CI 全绿（lint/tests/build）
* [ ] 不触碰 `.env` 与真实密钥
* [ ] 迁移有回滚方案且经小样本验证
* [ ] 只在工作分支改动；PR 说明清晰
* [ ] 影响面评估（Web/Celery/DB/前端/可视化/自动化）
* [ ] 文档更新（如涉及配置/接口）

---

## 12) 回滚策略（Rollback）

* 代码：`git revert <merge-commit>`
* 配置：通过 Feature Flag / ENV 回退到上一个稳定值
* 数据：如涉及迁移，提供 `migrate <prev>` 或对称迁移脚本；必要时读取备份（仅在测试/预发验证后）

---

## 13) 与外部系统的接口（Integrations）

* **Slack**：`scripts/slack_notify.sh` 使用 `<SLACK_WEBHOOK_URL>`（存于 Secrets）
* **n8n**：通过 HTTP Webhook 或远程命令触发 `claude code --goal "<…>"`
* **Metabase**：只读连接 `<METABASE_DB_CONN>`；禁止从本仓库直接写入 Metabase 设置表

---

## 14) 可替换占位符（一览）

* `<OWNER>`：GitHub 账户或组织名
* `<REPO_NAME>`：仓库名
* `<WEB_PORT>` `<REDIS_HOST>` `<DB_HOST>` `<DB_NAME>` `<N8N_URL>` `<METABASE_URL>` …
* `<COMPOSE_NETWORK>`：Compose 网络名
* `<SLACK_WEBHOOK_URL>`：Slack 入站 Webhook（存在 Secrets 中）
* `<指标名>`：你要新增的统计指标名

---

## 15) 变更历史（Changelog for CLAUDE.md）

* `v0.1` 初版模板（<YYYY-MM-DD>）
* `v0.2` 更新权限与分支保护说明（<YYYY-MM-DD>）

> 若本文件与现实流程不一致，以本文件为准；必要时先提 PR 更新本文件，再让 Claude 依此执行。
