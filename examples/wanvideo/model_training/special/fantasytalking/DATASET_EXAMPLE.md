# FantasyTalking 数据集目录示例（text + audio + video）

建议目录结构：

```text
fantasytalking_dataset/
├─ metadata.jsonl
├─ videos/
│  ├─ 000001.mp4
│  ├─ 000002.mp4
│  └─ ...
└─ audios/
   ├─ 000001.wav
   ├─ 000002.wav
   └─ ...
```

## `metadata.jsonl` 格式

每行一个 JSON 样本，最少需要 3 个字段：

- `prompt`: 文本描述（训练中的正向文本）
- `video`: 相对 `dataset_base_path` 的视频路径
- `input_audio`: 相对 `dataset_base_path` 的音频路径

示例：

```jsonl
{"prompt":"a woman is talking to camera", "video":"videos/000001.mp4", "input_audio":"audios/000001.wav"}
{"prompt":"a man speaks with calm expression", "video":"videos/000002.mp4", "input_audio":"audios/000002.wav"}
```

## 注意事项

- 音频建议为 **16kHz 单声道**（脚本里会按 16k 读取）。
- 视频会按训练参数做抽帧与裁剪（`--num_frames --height --width`）。
- `--data_file_keys` 需要包含：`video,input_audio`。
