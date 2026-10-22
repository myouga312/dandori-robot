# 段取りロボット（dandori-robot）

単一タスク用のVLAに、**追加学習なしで**、日本語のあいまいな指示を**複数の手順に分けて続けて**こなさせるシステム。
Physical AI 応用編1 最終課題。

**デモ動画**：https://youtu.be/MfEJrppipX4 （2分43秒。1:29〜2:07は2倍速）

```
日本語の指示 ─▶ Planner (Claude)      … スキル一覧の中から手順の列を選ぶ（JSON Schemaで出力を制約）。実行できない依頼は断る
                  │
                  ▼
           Executor (SmolVLA)         … 1手順ずつ閉ループで実行。リセットせずシーンを引き継ぐ
                  │  手順の間：関節空間のホーム復帰
                  │  100ステップごと
                  ▼
           Verifier (Claude, 画像)    … 実行前後の画像から達成を判定 → 停止／再試行／再計画
```

- シミュレータ：[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)のLIBERO-Goal（1つの台所シーンに10のゴールタスク）
- VLA：[`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero)（公開の重みをそのまま使用。50手を予測し10手を実行）
- LLM/VLM：Claude（`claude-opus-5-5`）。Claude Code の非対話モード（`claude -p`）か Claude API で呼ぶ

## 結果

各シナリオ10エピソード。成否はシミュレータの正解判定で数え、最後の時点で全目標が成立しているかを見る。

| 構成 | 全目標の達成 |
|---|---|
| 手順を単純につなぐ | 1/50 |
| ＋関節空間のホーム復帰 | 14/50 |
| ＋Verifier（停止・再試行・再計画）＝システム全体 | **23/50** |

- 2手順目以降の成功率：ホーム復帰なし 6/70（9%）→ あり 43/70（61%）
- Plannerの計画の正しさ：50/50。Verifierの判定の正解率：92.7%（各手順の最終判定191件）
- 単一スキル：77/100

詳細は [`docs/report.md`](docs/report.md) と [`results/`](results/) を参照。

## 工夫した点

1. **リセットせずにゴールだけを差し替える**：LIBERO-Goalの10タスクは同じシーンを共有している（`eval_skills.py` で起動時に検査）。
   物理状態はそのままに、BDDLのゴール条件だけを入れ替えるので、手順ごとにシミュレータの正解判定を使える。
2. **関節空間のホーム復帰**（`ChainSession.go_home`）：前の手順を終えた姿勢のままでは、次の手順が学習時の分布から外れる。
   手を開いて真上に逃がしたあと、コントローラを一時的に `JOINT_POSITION` に切り替え、7つの関節を初期値に戻す。
   手先の位置と向きを戻すだけ（OSC）では、肘の構えが戻らないことがあった。
3. **止めた後の猶予ステップ**：達成した瞬間に止めると、置いた物が倒れる。止めると決めた後も、25ステップだけVLAを動かす。
4. **Plannerの出力を、VLAが学習した指示文だけに絞る**：JSON Schemaの `enum` で制約する。
5. **APIを使わない呼び出し**：Claude Code の非対話モードを最小構成で起動する（`--system-prompt`、`--setting-sources ""`、`--tools ""`、空のMCP設定）。
   これで、1回の消費を約55,000トークンから約2,800トークンまで削った。
   GPUのVMからは、GCSを介した中継（`relay`）で呼ぶので、VMに認証情報を置かずに済む。

## 構成

```
src/dandori/
  skills.py        スキル一覧（LIBERO-Goalの指示文と日本語ラベル）
  sim.py           ChainSession：リセットなしの連続実行、ゴールの差し替え、ホーム復帰
  executor.py      SmolVLAで1手順を実行（LeRobotの前処理とポリシー読み込みを利用）
  planner.py       日本語の指示 → 手順の列
  verifier.py      実行前後の画像 → 達成の判定
  orchestrator.py  計画 → 実行 → 判定 → 再試行／再計画のループ
  llm.py           LLMの呼び出し（claude-cli / codex-cli / anthropic-api / vertex / relay）
  relay.py         GCSを介した中継（VM側）
  render.py        デモ動画の字幕（指示・計画の状態・判定の理由）
scripts/
  eval_skills.py       単一スキルの評価
  eval_chains.py       連続実行の評価（--mode oracle|vlm、--given-plan、--no-home）
  relay_worker.py      中継ワーカー（Claude Code にログインしている端末で動かす）
  compare_verifiers.py 保存した判定データで、別モデルのVerifierと比べる
  plot_results.py / make_final_video.py / rerender.py   図と動画
  vm/                  GCE VM の作成・同期・ジョブ実行
```

## 再現手順

```bash
# 0. 手元（Claude Code にログイン済みの端末）
uv venv && uv pip install -e ".[sim]"

# 1. GCE (L4) VM を作成。startup-script が環境構築（LeRobot/LIBERO/EGL）とスモーク評価まで自走する
cp scripts/vm/env.local.sh.example scripts/vm/env.local.sh   # PROJECT / GCS を記入
bash scripts/vm/vm.sh create
bash scripts/vm/sync.sh                                      # コードを同期（.env は送らない）

# 2. 中継ワーカーを手元で起動（評価中は動かしておく）
python scripts/relay_worker.py "$GCS/relay"

# 3. VM上で評価
MUJOCO_GL=egl python scripts/eval_skills.py --episodes 10
MUJOCO_GL=egl python scripts/eval_chains.py --mode oracle --given-plan --episodes 10          # Executorの上限
MUJOCO_GL=egl LLM_BACKEND=relay RELAY_GCS="$GCS/relay" python scripts/eval_chains.py --mode vlm --episodes 10
```

Claude API を使う場合は、`.env` に `ANTHROPIC_API_KEY` を置き、`LLM_BACKEND=anthropic-api` にする（`.env.example` を参照）。`.env` はgitの管理対象外。

### 環境構築の注意

- GCPのDeep Learning VMイメージには、計算用のNVIDIAドライバしか入っていない。そのため、MuJoCoの画面なし描画（EGL）が `failed to create dri2 screen` で失敗する。
  ドライバと**同じ版**の `libnvidia-gl-580-server` を入れると直る（`scripts/vm/startup_setup.sh`）。
- 一般ユーザーで描画するには、`video` / `render` グループに入れる必要がある。

## ライセンスと出典

- LIBERO：MIT License
- LeRobot：Apache-2.0（`lerobot.envs.libero`、プロセッサ、ポリシーの読み込みを利用）
- SmolVLA（`HuggingFaceVLA/smolvla_libero`）：公開の重みをそのまま利用
