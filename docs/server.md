# Local transcription server

[Русский](server.ru.md)

The optional server lets OpenAI-compatible transcription clients use GigaAM
Multilingual MLX locally. Audio stays on the Mac unless you explicitly expose
the server to another device.

## Start the server

Install `ffmpeg`, the package, and its server dependencies:

```bash
brew install ffmpeg
uv tool install 'gigaam-multilingual-mlx[server]'
gigaam-stt serve
```

The default configuration loads INT8 g64 and listens on
`http://127.0.0.1:8000`.

## Send a request

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=whisper-1 \
  -F file=@meeting.m4a
```

For the OpenAI Python client:

```bash
uv add openai
```

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
with Path("meeting.m4a").open("rb") as audio:
    result = client.audio.transcriptions.create(model="whisper-1", file=audio)
print(result.text)
```

The SDK requires a non-empty `api_key`, but `local` is not an OpenAI key and is
not checked while the server is bound to loopback without authentication.

## Supported API

Endpoints:

- `POST /v1/audio/transcriptions`
- `GET /v1/models`
- `GET /v1/models/{model_id}`
- `GET /healthz`
- `GET /readyz`

| Supported | Not supported in 0.2 |
|---|---|
| `json`, `text`, `verbose_json`, `srt`, `vtt` | realtime, SSE, WebSocket |
| word and segment timestamps | translation and diarization |
| language hints: `ru`, `kk`, `ky`, `uz`, `en` | logprobs and prompt conditioning |

The language field validates metadata only; it does not change CTC decoding or
perform language detection.

## Model IDs

Each process loads one model. Use either its canonical ID or the compatibility
alias `whisper-1`. The alias is provided for client compatibility; it does not
turn GigaAM into Whisper. To see the IDs served by the current process:

```bash
curl http://127.0.0.1:8000/v1/models
```

Select another variant when starting the server:

```bash
gigaam-stt serve --variant fp16
gigaam-stt serve --variant int6
gigaam-stt serve --variant int4
```

## Network access

The default loopback address accepts connections only from the same Mac. To
serve another device, set a bearer token and bind to the local network:

```bash
export GIGAAM_STT_API_KEY='replace-with-a-long-random-value'
gigaam-stt serve --host 0.0.0.0
```

Clients must then send `Authorization: Bearer ...`. You can store the token in
a file and pass `--api-key-file` instead. The server refuses a non-loopback bind
without authentication unless `--allow-unauthenticated` is explicitly set.

## Configuration

```bash
gigaam-stt serve --help
```

Useful options include `--port`, `--variant`, `--model`, `--offline`,
`--cache-dir`, `--max-upload-mb`, `--max-queue`, and `--temp-dir`. The default
upload limit is 1 GB. Inference is serialized to keep MLX/Metal access safe;
excess requests receive `429` instead of growing the queue indefinitely.

## Troubleshooting

| Response or message | What to do |
|---|---|
| Server dependencies are not installed | Run `uv tool install 'gigaam-multilingual-mlx[server]'` |
| `ffmpeg is required` | Run `brew install ffmpeg` |
| `401 invalid_api_key` | Send the bearer token configured on the server |
| `404 model_not_found` | Choose an ID returned by `/v1/models` |
| `413 file_too_large` | Use a smaller file or raise `--max-upload-mb` |
| `422 invalid_audio` | Check that `ffmpeg` can decode the file |
| `429 queue_full` | Wait and retry the request |

Unknown or unsupported OpenAI parameters return a structured error rather than
being ignored.
