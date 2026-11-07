# Python-Health-Insight-Agent 要件定義(Phase 0)

食事・運動・睡眠ログを題材にした「データ分析AIエージェント」。エージェントがデータを調べ、
根拠(使ったデータ・集計・モデル評価)付きで答える。ポートフォリオとして、
AI Agent設計 / RAG / MCP / SQL / 自作モデルとの併用 / 評価 / コスト管理を示し、
将来は既存アプリ(Python-Smart-Health-Intelligence-Platform、以下「既存アプリ」)へ統合できる構成にする。

> 状態: Phase 0 確定。第7章の未決事項は全て推奨案で決定済み(既存アプリは変更しない)。

## 1. 役割分担(最重要の設計原則)

| 担当 | やること | やらないこと |
|---|---|---|
| Gemini | 質問の解釈、ツール選択、結果の説明 | 数値の計算・判定 |
| 通常ツール | 取得・集計・相関・SQL・HealthScore等、事実のデータを返す | 解釈 |
| 自作モデル(Phase 9) | 翌日睡眠時間の予測、要因の重要度 | 事実の主張(「参考値」として返す) |

数値・判定はLLMに考えさせず、ツール/モデルの結果を根拠に説明させる。

## 2. ユースケースと想定質問

1. **原因探索**: 「先週、睡眠が悪かった日の共通点は?」「就寝直前の食事は睡眠に影響している?」
2. **相関・比較**: 「運動した日としていない日で睡眠時間はどれくらい違う?」
3. **予測(Phase 9)**: 「今夜遅めに食べたら、明日の睡眠はどうなりそう?」
4. **要約**: 「今週の健康スコアと、改善のおすすめは?」(既存HealthScore/Recommendation)
5. **振り返り検索(RAG)**: 「疲れが溜まっていた時期に、自分は何をメモしていた?」
6. **自由集計(Phase 7)**: 「平日と休日で就寝時刻の平均はどう違う?」(ガード付きSQL)
7. **「ない」と答える**: 仕込みのない関係(例: 朝食カロリーと睡眠)を聞かれたら、関係は見られないと答える。

回答に必ず含める: **使ったデータ / 件数 / わかったこと / わからないこと**。
データ不足・因果の断定は避ける。医療診断・治療助言はしない(注意書きを固定表示)。

## 3. やらないこと(Non-goals)

- 医療診断・疾病予測・治療や服薬の助言
- 実データの公開(公開物・デモは合成データのみ)
- 既存アプリDBへの書き込み(読み取り専用ユーザーのみ)
- 本番用の認証(当面は固定デモ`user_id`。統合時はREST API + Google OAuth を想定し、ADRに論点として残す)
- 合成データで学習したモデル精度の成果主張(「パイプライン動作実証」としてのみ扱う)
- Gemini以外のLLM対応(Protocol抽象は作るが実装はFake+Geminiのみ)
- 体重予測(既存アプリの領域。本プロジェクトのMLは睡眠のみ)

## 4. 既存アプリとの差分報告(読んだ範囲)

読んだもの: `database/schema.py`、`domain/models/{sleep_log,workout_log,daily_journal}.py`、
`docs/requirements.md`(14.1 ER図・9.3.1・10.4)、`ai/{retry_policy,protocols,gemini_provider}.py`、`config.py`、
`services/recommendation_service.py`、`analysis/health_score.py`、`features/daily_features.py`、
`ml/{dataset,evaluation,cross_validation,baseline,model_registry}.py`、`pyproject.toml`、`ci.yml`。
未読: `docs/architecture.md`、Alembic各リビジョン、`app/`(Streamlit)。

### 4.1 スキーマ(合わせるもの)

- 主キーは全てUUID。ユーザー所有テーブルは`user_id`を持つ(`ingredients`のみ共通)。日時は`timestamptz`。
- `sleep_logs(bedtime, wake_time, quality)` — **睡眠時間は保存しない**(`duration_minutes`は都度算出)。
- `meal_logs(consumed_at, meal_type, calories, protein_g, fat_g, carbohydrate_g, memo)`
- `workout_logs(performed_at, calories_burned?, memo)` → `workout_exercises(order_index, exercise_name, category, duration_minutes?)` → `workout_sets(order_index, reps, weight_kg?)`
- `health_logs(recorded_at, condition_score, morning_condition, evening_condition, mental_score, fatigue_score, stress_score, symptoms text[], memo)`
- `daily_journals(journal_date, ..., reflection_memo, improvement_point, ...)` — `(user_id, journal_date)`で一意。RAG対象はこの振り返り系テキスト(+`health_logs.memo`)。
- 日付の突合は **JST** で行う(睡眠は`wake_time`の日付に属する)。`local_time.py`相当が必要。

### 4.2 指示とのズレ・要判断(★は未決事項に載せた)

1. ★**入眠潜時の列が無い**: 仕込み例「夜の高強度運動 → 入眠が遅い」は、`sleep_logs`に入眠時刻/潜時が無く、`bedtime`=床に就いた時刻しか持てないため、そのままでは表現不能。
2. ★**運動強度の列が無い**: `workout_logs`に強度が無い。強度は `calories_burned`・有酸素の分数・筋トレのボリューム(重さ×回数)から派生させる必要がある。
3. ★**RetryPolicyのTimeout**: 指示は「既存と同じ値、1回20秒」だが、既存は **60秒**(`retry_policy.py`、requirements 10.4.1)。他(3回/1秒×2/±20%/429・503のみ)は一致。
4. ★**HealthScore/Recommendationの取り込み方**: 既存の`RecommendationService`は食事・睡眠・運動に加え **`weight_logs`も必須**(HealthScoreが体重を含む)で、既存パッケージ全体(Streamlit/authlib等を含む)に依存している。→ 第7章の案参照。
5. **既存`ml/`は体重専用**: `WeightChangeSample`/`MeanBaseline`が体重に固定。`forward_chaining_splits`・`evaluate`(閾値10%/20%)・`model_registry`(JSON保存)の考え方は流用できるが、睡眠用に**一般化(汎用Sample型)が必要**。評価の構造は同じ。
6. **欠損の扱い**: 既存`dataset.py`は`workout_duration_minutes`を欠損チェック対象外にして0扱い(既知の不具合の修正済み設計)。睡眠モデルでも同じ方針(運動なし=0、欠損ではない)。ただし既存は **筋トレ(STRENGTH)を運動時間に計上しない**ため、強度特徴量は別途設計する(上記2)。
7. **既存の`model_registry`保存先**はリポジトリ直下`ml_artifacts/`。本プロジェクトも同様にし、`.gitignore`に入れる。
8. **既存は認証がStreamlit+Google OAuth(allowlist)でREST APIが無い**。統合時はFastAPI層の新設が論点(ADR化)。
9. 既存のGeminiモデル既定値は`gemini-3.6-flash`(コメントで「要確認」)。本プロジェクトは環境変数で切替え、Phase 2で公式ドキュメントを確認して決める。
10. 既存CIはPostgres 16サービス+`alembic upgrade head`+ruff/format/mypy(`src`のみ)/pytest。同形で作り、Ruff/mypyは本プロジェクトのCIにも入れる。

## 5. 技術スタック確認結果

| 項目 | 結果 |
|---|---|
| `uv python install 3.15.1` | **不要・導入済み**(`cpython-3.15.1`がuv管理下に既存。Homebrewには3.15.2もある) |
| 依存(3.15.1実測) | fastapi / uvicorn / sqlalchemy 2.1 / alembic / psycopg[binary] / pydantic / google-genai / lightgbm 4.7 / numpy 2.5 / sqlglot / mcp 2.3 / pytest / ruff / mypy は**wheelあり・import可・LightGBM学習も動作確認済み** |
| 要注意: mlflow | **pyyamlの3.15用wheelが無く、sdistからビルド**されて導入・import成功(mlflow 3.18)。wheel-onlyの環境では解決失敗する。CI(ubuntu)ではビルドできる見込みだが未検証 |
| 要注意: google-genai | importで`SyntaxWarning`(無害)。pytestで警告をエラー化する設定は避ける |
| 既存アプリ | Python 3.13(`>=3.13`)。3.13/3.14は全依存がwheelで解決可(実測) |
| Node | v24.22.1 / npm 11.19.1(React+Vite可) |
| PostgreSQL | docker daemon稼働中、psql/postgres(14)もあり。開発・テストDBはdocker composeのPostgres 16(CIと揃える)を想定 |

→ 3.15.1で進められるが、mlflowのsdistビルド依存は「黙って進めない」対象のため、第7章で確認する。

## 6. 設計方針(要約)

- 層: Presentation(API/CLI/MCP/tools) → Service → Domain ← Repository → Infrastructure。外部境界はPydantic、内部はdataclass。
- 全データアクセスは`user_id`境界。他ユーザーのデータは取得不能(Repositoryの全メソッドが`user_id`必須。分離テストを書く)。
- `LLMProvider` / `VectorStore` / `Predictor` / ツール呼び出し方式(直接/MCP)は全てProtocol+設定で差し替え。
- `AI_PROVIDER=fake|gemini`、テストは実API禁止。Geminiに送るデータは最小限(メール等の個人識別情報は送らない)。
- 1質問ごとに トークン数・ツール呼び出し回数・所要時間をログ。ツール最大回数は設定値。
- 合成データ(90日、既知の因果パターン入り)が評価の「正解」。実DBは読み取り専用接続のみ。
- テストDBは`.env.test`で分離。既存`test_migrations`のようにテーブルを消すテストは実DBに向けさせない(接続先ガードを入れる)。

## 7. 決定事項(全て推奨案を採用。既存アプリは読み取りのみで変更しない)

決定: 1=案A / 2=派生`intensity` / 3=20秒 / 4=案A(移植+Protocol) / 5=3.15.1(CI失敗時のみ3.13) / 6=`health_insight_agent` / 7=docker compose Postgres 16 + 任意の読み取り専用接続。

以下は検討時の選択肢(経緯の記録)。

1. **睡眠の仕込みパターンの表現**(4.2-1)
   - 案A(推奨): パターンを「就寝時刻が遅れる」「睡眠時間が短くなる」「`quality`が悪化」で表現する(スキーマ変更なし。「入眠が遅い」は`bedtime`の遅れで代替)。
   - 案B: 本プロジェクトの`sleep_logs`に`sleep_onset_at`等を追加(既存と乖離し、統合時にmigrationが必要)。
2. **運動強度の定義**(4.2-2): 案A(推奨): 派生指標`intensity`を定義(有酸素=分数×係数、筋トレ=ボリューム、`calories_burned`があれば優先)し、`high`は閾値で判定。合成データの仕込みもこの定義に合わせる。
3. **RetryPolicyのTimeout**: 20秒(ご指示)/ 60秒(既存の実値)のどちらにしますか。推奨は、エージェントは1質問で複数回呼ぶため **20秒**で、ADRに「既存は60秒、エージェントは短縮」と記録。
4. **既存HealthScore/Recommendationの取り込み**(4.2-4):
   - 案A(推奨): `HealthInsightPort`(Protocol)を定義し、既存の`analysis/`・`features/`・`recommendation/`の必要部分を**移植(コピー+出典明記)**。`weight_logs`テーブルも合成データ・スキーマに含める。統合時は実装を既存サービスのアダプタに差し替えるだけ。
   - 案B: 既存リポジトリをgit依存として追加(Streamlit等が丸ごと入る。重い)。
5. **Pythonバージョン**: 3.15.1で進める(mlflowのpyyaml sdistビルドを許容)/ 3.13に揃える(既存と同じ・全てwheel)。推奨は、**3.15.1で進め、CIでビルド失敗した場合のみ3.13へ後退**。
6. **パッケージ名**: `health_insight_agent`(推奨)でよいか。
7. **DB接続**: 開発はdocker composeのPostgres 16、既存DB(Neon)は読み取り専用ユーザーの`READONLY_DATABASE_URL`で任意接続、で良いか。

## 8. ディレクトリ構成案(確認後に作成)

```text
Python-Health-Insight-Agent/
├── pyproject.toml / uv.lock / .python-version / alembic.ini
├── .env.example / .env.test.example / .gitignore / docker-compose.yml
├── .github/workflows/ci.yml          # ruff, ruff format, mypy, alembic, pytest, (frontend lint/build)
├── docs/
│   ├── requirements.md               # 本書
│   └── adr/                          # 設計判断(Phase 8〜)
├── src/health_insight_agent/
│   ├── config.py                     # 設定・DI組み立て(AI_PROVIDER等)
│   ├── local_time.py                 # JST日付処理(既存に合わせる)
│   ├── domain/                       # enums, models/(dataclass: Sleep/Meal/Workout/Health/Journal/Weight)
│   ├── database/                     # schema.py(ORM), connection.py(Neon再試行), migrations/
│   ├── repositories/                 # protocols.py, unit_of_work.py, sqlalchemy/
│   ├── services/                     # sleep/meal/workout/journal, analysis(compare/correlate), health_score
│   ├── ai/                           # protocols(LLMProvider), fake_provider, gemini_provider,
│   │                                 #   retry_policy, errors, embeddings
│   ├── rag/                          # VectorStore Protocol, numpy実装, indexer(pgvector差替え口)
│   ├── tools/                        # ツール定義(Pydanticスキーマ)+registry、sql_guard(Phase 7)
│   ├── agent/                        # ループ(最大回数・エラー処理)、prompts、usage_log、CLI
│   ├── mcp_server/                   # MCPサーバー、直接/MCP切替(Phase 6)
│   ├── ml/                           # dataset, features, evaluation, cross_validation,
│   │                                 #   model_registry, predictor(Protocol), train(Phase 9)
│   ├── synthetic/                    # 合成データ生成+仕込みパターン定義
│   └── api/                          # FastAPI(routes, schemas)
├── evaluation/                       # questions.yaml(正解付き20問), run_eval.py, results/(生成物はgit管理外)
├── frontend/                         # React + Vite + TypeScript(チャットUI、根拠、グラフ)
├── ml_artifacts/ , mlruns/           # 生成物(.gitignore)
└── tests/                            # unit/ integration/(実DBはテストDBのみ)
```

既存との対応: `domain / repositories / services / ai / ml / database / config.py / local_time.py` は同名。
新規は `tools / agent / rag / mcp_server / synthetic / api / evaluation / frontend`。

## 9. フェーズ計画

ご指示のPhase 1〜9のとおり。各フェーズ末で停止し、報告→「OK」後に`Phase N: 内容`でコミット。
