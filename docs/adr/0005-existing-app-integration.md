# 0005: 既存アプリとの統合方針(認証の論点を含む)

- 状態: 採用(認証は論点として残す)

## 背景
本プロジェクトは既存の健康アプリ(`Python-Smart-Health-Intelligence-Platform`)へ後で統合する前提。
既存アプリは変更しない。

## 決定
- **スキーマ・命名・RetryPolicy・`AI_PROVIDER` 切替を既存に揃える**。ORMは既存と同形(対象は読むテーブルのみ)。
- **HealthScore/Recommendation は移植**(出典明記)し、`HealthInsightPort`(Protocol)の背後に置く。
  既存パッケージをgit依存で入れる案は、Streamlit等が丸ごと入るため採らなかった。統合時は既存の
  `RecommendationService` を包むアダプタに差し替える。
- 既存DBを読む場合は読み取り専用ユーザーのみ(`READONLY_DATABASE_URL`)。書き込みは構造上できない。
- 既存の睡眠は「保存しない・都度算出」、日付はJST、睡眠は起床日に属する、という既存の約束を踏襲する。
- 差分: 入眠の遅れ・運動強度の列が無い → スキーマを変えず、就寝時刻の遅れと派生強度で表現した
  (統合時のマイグレーション不要)。

## 論点(未決): 統合時の認証
既存アプリはStreamlit + Google OAuth(許可リスト)でREST APIを持たない。統合には、FastAPI層を足し、
OAuthのセッション/トークンから `user_id` を取る必要がある。本プロジェクトでは `api/app.py` の
`current_user_id` 依存を差し替え点として用意しており、ツール・サービスは影響を受けない。
トークンの検証方式(セッションCookie / Bearer JWT)と、Streamlitとの共存方法は統合時に決める。

## 結果
- 本リポジトリ単体で完結して動き、統合の差し替え箇所が少数に限られる。
