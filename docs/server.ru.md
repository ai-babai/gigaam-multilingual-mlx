# Локальный сервер транскрипции

[English](server.md)

Опциональный сервер позволяет использовать GigaAM Multilingual MLX в
приложениях с поддержкой OpenAI Audio Transcriptions API. Аудио остается на
Mac, пока вы сами не откроете сервер для других устройств.

## Запуск

Установите `ffmpeg`, пакет и зависимости сервера:

```bash
brew install ffmpeg
uv tool install 'gigaam-multilingual-mlx[server]'
gigaam-stt serve
```

По умолчанию загружается INT8 g64, а сервер доступен по адресу
`http://127.0.0.1:8000`.

## Первый запрос

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=whisper-1 \
  -F file=@meeting.m4a
```

Для Python-клиента OpenAI:

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

SDK требует непустое значение `api_key`, но строка `local` не является ключом
OpenAI. При локальном запуске без авторизации сервер ее не проверяет.

## Поддерживаемый API

Эндпоинты:

- `POST /v1/audio/transcriptions`
- `GET /v1/models`
- `GET /v1/models/{model_id}`
- `GET /healthz`
- `GET /readyz`

| Поддерживается | Не входит в версию 0.2 |
|---|---|
| `json`, `text`, `verbose_json`, `srt`, `vtt` | realtime, SSE, WebSocket |
| таймкоды слов и сегментов | перевод и диаризация |
| языковые подсказки: `ru`, `kk`, `ky`, `uz`, `en` | logprobs и prompt conditioning |

Поле языка только проверяется и возвращается в метаданных. Оно не меняет
CTC-декодирование и не включает автоматическое определение языка.

## Названия моделей

Один процесс загружает одну модель. В запросе можно указать ее каноническое имя
или совместимый псевдоним `whisper-1`. Псевдоним нужен только клиентам: Whisper
при этом не запускается. Текущие имена возвращает команда:

```bash
curl http://127.0.0.1:8000/v1/models
```

Другой вариант выбирается при запуске:

```bash
gigaam-stt serve --variant fp16
gigaam-stt serve --variant int6
gigaam-stt serve --variant int4
```

## Доступ из локальной сети

По умолчанию подключиться к серверу можно только с этого Mac. Для доступа с
другого устройства задайте токен и откройте сервер в локальной сети:

```bash
export GIGAAM_STT_API_KEY='replace-with-a-long-random-value'
gigaam-stt serve --host 0.0.0.0
```

Клиент должен передавать заголовок `Authorization: Bearer ...`. Вместо
переменной окружения можно хранить токен в файле и использовать
`--api-key-file`. Без авторизации сервер откажется слушать внешний адрес, если
только явно не указан небезопасный флаг `--allow-unauthenticated`.

## Настройка

```bash
gigaam-stt serve --help
```

Основные параметры: `--port`, `--variant`, `--model`, `--offline`,
`--cache-dir`, `--max-upload-mb`, `--max-queue` и `--temp-dir`. Размер файла по
умолчанию ограничен 1 GB. Запросы к MLX выполняются последовательно; при
переполнении очереди сервер отвечает `429`.

## Решение проблем

| Сообщение или код | Что сделать |
|---|---|
| Не установлены зависимости сервера | Выполнить `uv tool install 'gigaam-multilingual-mlx[server]'` |
| `ffmpeg is required` | Выполнить `brew install ffmpeg` |
| `401 invalid_api_key` | Передать токен, настроенный на сервере |
| `404 model_not_found` | Выбрать имя из `/v1/models` |
| `413 file_too_large` | Уменьшить файл или увеличить `--max-upload-mb` |
| `422 invalid_audio` | Проверить, что `ffmpeg` открывает файл |
| `429 queue_full` | Подождать и повторить запрос |

Неизвестные и неподдерживаемые параметры OpenAI возвращают явную ошибку, а не
игнорируются.
