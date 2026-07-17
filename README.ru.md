# GigaAM-Multilingual MLX

[English](README.md) · **Русский**

Быстрое локальное распознавание речи в текст и транскрибация аудио на Mac
с Apple Silicon — для русского, казахского, кыргызского (киргизского) и узбекского
языков. GigaAM MLX локально расшифровывает встречи, интервью, лекции, подкасты,
голосовые сообщения и аудиодорожки видео в TXT, JSON, SRT или VTT. Это
независимый нативный порт
[GigaAM](https://github.com/salute-developers/GigaAM) на
[MLX](https://github.com/ml-explore/mlx), основанный на официальной модели
[GigaAM-Multilingual](https://huggingface.co/ai-sage/GigaAM-Multilingual).

[PyPI](https://pypi.org/project/gigaam-multilingual-mlx/) ·
[Модели на Hugging Face](https://huggingface.co/collections/ai-babai/gigaam-multilingual-mlx-6a57ef36e44d1a0c4eb49276) ·
[Последний релиз](https://github.com/ai-babai/gigaam-multilingual-mlx/releases/latest) ·
[Полный бенчмарк](docs/benchmark-multilingual-v1.md)

## Качество распознавания по языкам

![Сравнение GigaAM MLX, Whisper и Parakeet](https://raw.githubusercontent.com/ai-babai/gigaam-multilingual-mlx/main/docs/benchmark-multilingual-v1.png?v=0.1.5)

На графике сравниваются WER, время расшифровки пятиминутного WAV, пиковая память
процесса и размер модели. Залитые плашки обозначают лидеров столбцов, контурные —
вторые места, `◇` — Pareto frontier, `★` — рекомендуемый вариант. Во всех
колонках меньше — лучше.

## Быстрый старт

Нужны Mac на Apple Silicon, macOS 14+, Python 3.12 или 3.13,
[`uv`](https://docs.astral.sh/uv/getting-started/installation/) и
[`ffmpeg`](https://ffmpeg.org/).

```bash
brew install uv ffmpeg
uv tool install gigaam-multilingual-mlx
gigaam-stt meeting.m4a --output transcript.txt
```

Для Python-проекта под управлением uv:

```bash
uv add gigaam-multilingual-mlx
```

Также поддерживаются `pip install gigaam-multilingual-mlx` и полная команда
`gigaam-multilingual-mlx transcribe ...`.

## Локальный сервер транскрипции

GigaAM MLX можно подключить к приложениям, которые поддерживают OpenAI
Transcriptions API:

```bash
uv tool install 'gigaam-multilingual-mlx[server]'
gigaam-stt serve
```

После запуска API доступен по адресу `http://127.0.0.1:8000/v1`:

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=whisper-1 \
  -F file=@meeting.m4a
```

Этот же адрес можно передать Python-клиенту OpenAI (`uv add openai`):

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
with Path("meeting.m4a").open("rb") as audio:
    result = client.audio.transcriptions.create(model="whisper-1", file=audio)
print(result.text)
```

`whisper-1` — имя для совместимости: внутри по-прежнему работает GigaAM
Multilingual MLX. По умолчанию сервер доступен только на этом Mac. Форматы,
названия моделей, сетевой доступ и решение проблем описаны в
[руководстве по серверу](https://github.com/ai-babai/gigaam-multilingual-mlx/blob/main/docs/server.ru.md).

## Зачем этот порт?

- Нативный MLX-инференс без PyTorch, ONNX Runtime, Core ML и облачного API.
- Высокое измеренное качество на четырёх основных языках GigaAM, особенно на
  казахском, кыргызском и узбекском.
- Модель по умолчанию весит 699 MB и использовала 0.877 GB пиковой памяти
  процесса в опубликованном пятиминутном тесте.
- Локальная расшифровка WAV, FLAC, MP3, M4A и видео в TXT, JSON, SRT или VTT.

На тестовом пятиминутном русском аудио INT8 оказался в 3.30× быстрее Whisper v3
Turbo, в 7.02× быстрее Whisper large-v2 и в 8.94× быстрее Whisper large-v3.
На английском Whisper и Parakeet показали меньший WER. Это результаты одного
компьютера и публичного корпуса, а не универсальный рейтинг ASR. Доверительные
интервалы, команды, хеши и ограничения находятся в
[полном отчёте](docs/benchmark-multilingual-v1.md).

## Варианты модели

| Вариант | Когда выбирать | Размер модели | Peak RAM | Hugging Face |
|---|---|---:|---:|---|
| **INT8 g64** | **баланс по умолчанию** | **699 MB** | **0.88 GB** | [`ai-babai/...-int8-g64`](https://huggingface.co/ai-babai/gigaam-multilingual-mlx-int8-g64) |
| FP16 | максимальная измеренная скорость, эталонный порт | 1.17 GB | 1.35 GB | [`ai-babai/gigaam-multilingual-mlx`](https://huggingface.co/ai-babai/gigaam-multilingual-mlx) |
| INT6 g64 | компактнее, качество около INT8 | 573 MB | 0.76 GB | [`ai-babai/...-int6-g64`](https://huggingface.co/ai-babai/gigaam-multilingual-mlx-int6-g64) |
| INT4 g64 | минимум памяти и места | 447 MB | 0.63 GB | [`ai-babai/...-int4-g64`](https://huggingface.co/ai-babai/gigaam-multilingual-mlx-int4-g64) |

Выбрать вариант можно без знания имени репозитория:

```bash
gigaam-stt speech.wav --variant fp16
gigaam-stt speech.wav --variant int6 --format json
gigaam-stt models
```

При первом запуске библиотека скачивает неизменяемую ревизию весов `v0.1.0`,
затем использует стандартный кеш Hugging Face.

## Основные команды

```bash
# Субтитры
gigaam-stt interview.mp3 --format srt --output interview.srt

# Создаст transcripts/recording.vtt
gigaam-stt recording.mov --format vtt --output-dir transcripts/

# Текст, таймкоды, ревизия модели и метрики
gigaam-stt sample.flac --format json --output sample.json

# Работа без сети с уже скачанной моделью
gigaam-stt audio.wav --offline

# Другой каталог кеша Hugging Face
gigaam-stt audio.wav --cache-dir /Volumes/ML/huggingface
```

Длинные записи обрабатываются детерминированными перекрывающимися фрагментами.
Словесные таймкоды приблизительно восстанавливаются из CTC-эмиссий: они подходят
для навигации и субтитров, но не заменяют forced alignment.

## Python API

```python
import mlx.core as mx
from gigaam_multilingual_mlx import load_model

model = load_model(variant="int8")
audio = mx.zeros((1, 16_000), dtype=mx.float32)
log_probs, lengths = model(audio)
text = model.greedy_decode(log_probs, lengths)[0]["text"]
```

Можно загрузить и локальный переносимый каталог модели:
`load_model("/path/to/model")`.

## Совместимость и ограничения

- Поддерживаются Apple Silicon, нативный ARM Python и macOS 14+.
- M1–M5 должны быть совместимы на уровне runtime; опубликованные цифры получены
  на 14-дюймовом MacBook Pro с Apple M4 Pro и 48 GB памяти.
- Intel Mac, Linux, Windows и iOS не поддерживаются.
- Используется greedy CTC decoding. В релиз не входят diarization, обучение,
  поток с микрофона и realtime HTTP/WebSocket streaming.
- Качество может снижаться на шуме, дальней речи, перекрывающихся голосах,
  code-switching и доменах, не похожих на публичные тестовые данные.

## Воспроизводимость и разработка

В репозитории находятся зафиксированные публичные FLEURS-манифесты, компактные
результаты и команды для GigaAM, MLX Whisper и MLX Parakeet. Модели, датасеты,
кеши, приватное аудио и большие сырые результаты в Git не входят.

Инструменты конвертации и оценки используют дополнительные зависимости:

```bash
python -m pip install 'gigaam-multilingual-mlx[convert,quality]'
python -m gigaam_multilingual_mlx.dev_cli --help
```

Методика описана в
[`benchmarks/multilingual-v1/README.md`](benchmarks/multilingual-v1/README.md),
результаты — в
[`docs/benchmark-multilingual-v1.md`](docs/benchmark-multilingual-v1.md).

### Ключевые цифры текстом

Эта компактная таблица дублирует часть изображения в доступном текстовом виде.
Core macro WER усредняет русский, казахский, кыргызский и узбекский; меньше —
лучше.

| Вариант GigaAM MLX | Core macro WER | 5-min WAV | Peak RAM | Размер модели |
|---|---:|---:|---:|---:|
| FP16 | 5.066% | **1.952s** | 1.350 GB | 1.171 GB |
| **INT8 g64 (по умолчанию)** | **5.070%** | **2.036s** | **0.877 GB** | **0.699 GB** |
| INT6 g64 | 5.069% | 2.195s | 0.755 GB | 0.573 GB |
| INT4 g64 | 5.219% | 2.563s | **0.626 GB** | **0.447 GB** |

## Лицензия и атрибуция

MLX-порт опубликован под лицензией MIT. Происхождение upstream-кода, модели и
датасетов записано в [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). При
использовании укажите оригинальную работу GigaAM-Multilingual и этот проект через
[`CITATION.cff`](CITATION.cff).

Это не официальный релиз авторов GigaAM. Инструкции по сообщениям об уязвимостях
и участию находятся в [`SECURITY.md`](SECURITY.md) и
[`CONTRIBUTING.md`](CONTRIBUTING.md).
