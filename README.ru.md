# GigaAM-Multilingual MLX

[English](README.md) · **Русский**

Офлайн-распознавание речи на русском, казахском, кыргызском и узбекском языках
для Apple Silicon. Это независимый нативный порт
[GigaAM](https://github.com/salute-developers/GigaAM) на
[MLX](https://github.com/ml-explore/mlx), основанный на официальной модели
[GigaAM-Multilingual](https://huggingface.co/ai-sage/GigaAM-Multilingual).
Проект не является официальным релизом авторов GigaAM.

[Пакет PyPI](https://pypi.org/project/gigaam-multilingual-mlx/) ·
[коллекция моделей](https://huggingface.co/collections/ai-babai/gigaam-multilingual-mlx-6a57ef36e44d1a0c4eb49276) ·
[последний релиз](https://github.com/ai-babai/gigaam-multilingual-mlx/releases/latest)

`gigaam-multilingual-mlx` выполняет локальное многоязычное распознавание речи без
PyTorch, ONNX Runtime, Core ML и облачных ASR-сервисов. Веса загружаются отдельно
с Hugging Face и повторно используются из стандартного кеша Hub.

## Установка и распознавание за минуту

Требования: Mac с Apple Silicon, macOS 14 или новее, Python 3.12 или 3.13,
[`uv`](https://docs.astral.sh/uv/getting-started/installation/) и
[`ffmpeg`](https://ffmpeg.org/).

```bash
brew install uv ffmpeg
uv tool install gigaam-multilingual-mlx
gigaam-stt meeting.m4a --output transcript.txt
```

Для Python API в проекте под управлением uv выполните
`uv add gigaam-multilingual-mlx`. Обычная установка
`python -m pip install gigaam-multilingual-mlx` тоже поддерживается.
Каноническая команда `gigaam-multilingual-mlx transcribe ...` и полная форма
`gigaam-stt transcribe ...` сохранены для совместимости.

При первом запуске загружается модель INT8 с неизменяемого тега релиза `v0.1.0`.
Следующие запуски используют кешированные файлы.

## Зачем GigaAM STT?

- Высокое измеренное качество для четырёх основных языков GigaAM, особенно для
  казахского, кыргызского и узбекского.
- Нативный MLX-инференс на Apple Silicon без PyTorch, ONNX Runtime, Core ML и
  облачного сервиса.
- Модель по умолчанию размером 699 MB и заметно меньшая память процесса, чем у
  протестированных MLX-чекпойнтов Whisper.
- Полностью локальная работа после первой загрузки, простой CLI
  `gigaam-stt AUDIO` и форматы TXT, JSON, SRT и VTT.

### Выбор модели: GigaAM MLX, MLX Whisper и MLX Parakeet

Corpus WER измерен на зафиксированных test-выборках FLEURS. Ресурсные колонки
используют один пятиминутный русский WAV на 14-дюймовом MacBook Pro с Apple M4 Pro
и 48 GB unified memory. `5-min WAV` — медиана пяти запусков после загрузки модели;
Peak RAM — peak RSS всего процесса; model size — размер файла весов на диске.

`✓` лидер столбца · `◇` Pareto frontier · `★` рекомендуемый вариант. Меньше —
лучше. Английский — appendix. Parakeet официально не поддерживает казахский,
кыргызский и узбекский, поэтому там указано `N/A`.

#### ◇ Pareto frontier — начните отсюда

Фронтир рассчитан среди MLX-кандидатов по равновзвешенному macro WER русского,
казахского, кыргызского и узбекского, времени `5-min WAV`, Peak RAM и размеру
модели. Original GigaAM — reference baseline PyTorch/MPS; английский исключён как
appendix; модели без одного из основных языков в расчёт не входят.

| MLX-вариант на фронтире | Core macro WER | 5-min WAV | Peak RAM | Model size | Для чего выбрать |
|---|---:|---:|---:|---:|---|
| ◇ GigaAM MLX FP16 | 5.066% | **✓ 1.952s** | 1.350 GB | 1.171 GB | максимальная скорость и близость к исходному порту |
| **◇ ★ GigaAM MLX INT8** | **5.070%** | **2.036s** | **0.877 GB** | **0.699 GB** | **рекомендуемый баланс качества, скорости и размера** |
| ◇ GigaAM MLX INT6 | 5.069% | 2.195s | 0.755 GB | 0.573 GB | меньший footprint при качестве около INT8 |
| ◇ GigaAM MLX INT4 | 5.219% | 2.563s | **✓ 0.626 GB** | **✓ 0.447 GB** | минимальные размер модели и Peak RAM |

#### Все сравниваемые модели

| Модель / вариант | RU WER | KZ WER | KY WER | UZ WER | EN WER | 5-min WAV | Peak RAM | Model size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original GigaAM PyTorch/MPS | **✓ 2.995%** | **✓ 4.325%** | **✓ 5.553%** | **✓ 7.310%** | 9.717% | 2.789s | 5.059 GB | 2.342 GB |
| ◇ GigaAM MLX FP16 | **✓ 2.995%** | 4.342% | 5.560% | 7.367% | 9.726% | **✓ 1.952s** | 1.350 GB | 1.171 GB |
| **◇ ★ GigaAM MLX INT8** | 3.013% | 4.351% | 5.582% | 7.334% | 9.734% | 2.036s | 0.877 GB | 0.699 GB |
| ◇ GigaAM MLX INT6 | 3.013% | 4.385% | 5.568% | **✓ 7.310%** | 9.848% | 2.195s | 0.755 GB | 0.573 GB |
| ◇ GigaAM MLX INT4 | 3.234% | 4.377% | 5.768% | 7.497% | 9.883% | 2.563s | **✓ 0.626 GB** | **✓ 0.447 GB** |
| MLX Whisper large-v2 | 3.855% | 39.468% | 92.582% | 95.551% | 4.194% | 14.287s | 3.733 GB | 3.083 GB |
| MLX Whisper large-v3 | 3.123% | 32.778% | 86.680% | 87.981% | **✓ 4.098%** | 18.214s | 3.765 GB | 3.084 GB |
| MLX Whisper large-v3-turbo | 3.549% | 20.468% | 84.078% | 108.931% | 4.579% | 6.722s | 1.898 GB | 1.614 GB |
| MLX Parakeet TDT 0.6B v3 | 4.961% | N/A | N/A | N/A | 4.928% | 3.843s | 1.085 GB | 2.508 GB |

Для казахского, кыргызского и узбекского paired bootstrap 95% CI подтверждает
меньший WER INT8 относительно всех трёх Whisper. По русскому INT8 достоверно лучше
large-v2 и v3 Turbo, но разница с large-v3 не подтверждена. Русский WER INT8 также
на 39.3% ниже Parakeet (paired-bootstrap 95% CI: 32.7–45.6%). На английском
Whisper и Parakeet лучше GigaAM; кроме того, Whisper поддерживает значительно
больше языков и перевод речи на английский. Parakeet официально не поддерживает
казахский, кыргызский и узбекский, поэтому указано `N/A`, а не нерелевантный WER.

На этом входе INT8 оказался в 3.30× быстрее Whisper v3 Turbo, в 7.02× быстрее
large-v2 и в 8.94× быстрее large-v3 при снижении peak RSS на 54–77%. Относительно
Parakeet он быстрее в 1.89×, использует на 19.2% меньше peak RSS, а файл весов
меньше на 72.1%. Полные данные по всем языкам, FP16/INT6/INT4, CER, p95,
confidence intervals, хешам и ограничениям находятся в
[многоязычном отчёте](https://github.com/ai-babai/gigaam-multilingual-mlx/blob/main/docs/benchmark-multilingual-v1.md).

## Варианты модели

> **Рекомендуемый вариант: INT8 g64.** Он сохраняет качество, близкое к FP16, и
> даёт лучший измеренный баланс размера загрузки, памяти процесса и скорости.

| Вариант | Назначение | Размер модели | Репозиторий Hugging Face |
|---|---|---:|---|
| **INT8 g64** | **рекомендуемый вариант, наиболее безопасная квантизация** | **699 MB** | **[`ai-babai/gigaam-multilingual-mlx-int8-g64`](https://huggingface.co/ai-babai/gigaam-multilingual-mlx-int8-g64)** |
| FP16 | эталонный MLX-артефакт | 1.17 GB | [`ai-babai/gigaam-multilingual-mlx`](https://huggingface.co/ai-babai/gigaam-multilingual-mlx) |
| INT6 g64 | компактный сбалансированный | 573 MB | [`ai-babai/gigaam-multilingual-mlx-int6-g64`](https://huggingface.co/ai-babai/gigaam-multilingual-mlx-int6-g64) |
| INT4 g64 | минимальный, включается явно | 447 MB | [`ai-babai/gigaam-multilingual-mlx-int4-g64`](https://huggingface.co/ai-babai/gigaam-multilingual-mlx-int4-g64) |

Все варианты объединены в
[коллекцию GigaAM Multilingual MLX](https://huggingface.co/collections/ai-babai/gigaam-multilingual-mlx-6a57ef36e44d1a0c4eb49276).
Техническое upstream-имя ветки `large_ctc` сохраняется только в metadata и не
является частью публичного названия проекта.

Выбрать вариант можно без знания имени репозитория:

```bash
gigaam-stt speech.wav --variant fp16
gigaam-stt speech.wav --variant int6 --format json
gigaam-stt models
```

## Входные и выходные форматы

`ffmpeg` декодирует WAV, FLAC, MP3, M4A и аудиодорожки распространённых
видеоконтейнеров. Перед инференсом звук преобразуется в mono PCM 16 kHz.

```bash
# Записать результат по указанному пути
gigaam-stt interview.mp3 --format srt --output interview.srt

# Создать OUTPUT_DIR/INPUT_STEM.vtt
gigaam-stt recording.mov --format vtt --output-dir transcripts/

# JSON с текстом, таймкодами слов, revision модели и метриками
gigaam-stt sample.flac --format json --output sample.json
```

Поддерживаются `txt`, `json`, `srt` и `vtt`. CTC-таймкоды слов оцениваются по
кадрам emissions. Они монотонны и пригодны для навигации и субтитров, но не
являются forced alignment.

Длинные записи обрабатываются детерминированными перекрывающимися фрагментами.
Граф инференса и тензоры повторно используются между фрагментами.

## Кеш, локальные модели и офлайн-режим

По умолчанию Hugging Face хранит snapshots в `~/.cache/huggingface/hub`. Для
переноса кеша используйте стандартные переменные Hub, например `HF_HOME`, либо
передайте каталог явно:

```bash
gigaam-stt audio.wav --cache-dir /Volumes/ML/huggingface
gigaam-stt audio.wav --offline
gigaam-stt audio.wav --model /path/to/portable/model
gigaam-stt audio.wav \
  --model ai-babai/gigaam-multilingual-mlx-int8-g64 \
  --revision v0.1.0
```

`--offline` не обращается к Hub и сообщает понятную ошибку, если snapshot не
закеширован. Для удаления только загруженных моделей используйте кеш-команды
Hugging Face CLI или удалите соответствующие каталоги `models--ai-babai--gigaam-*`,
не затрагивая другие файлы Hub.

## Python API

Загрузка из локального каталога и вариантов Hub использует одну функцию:

```python
import mlx.core as mx
from gigaam_multilingual_mlx import load_model

model = load_model(variant="int8")  # загружает/использует кеш ai-babai/...@v0.1.0
audio = mx.zeros((1, 16_000), dtype=mx.float32)
log_probs, lengths = model(audio)
text = model.greedy_decode(log_probs, lengths)[0]["text"]

local_model = load_model("/path/to/portable/model")
```

## Матрица поддержки

| Область | Статус |
|---|---|
| Apple Silicon, поддерживаемый MLX; macOS 14+; нативный ARM Python | поддерживается |
| Поколения M1–M5 | ожидается совместимость; отдельно не тестировались |
| Apple M4 Pro, 48 GB, macOS 15.7.7 | машина релизного benchmark |
| Intel Mac, Linux, Windows, iOS | не поддерживаются в `v0.1.0` |

На неподдерживаемой платформе и при отсутствии `ffmpeg` пакет выводит понятную
инструкцию по исправлению.

## Валидация и benchmark

Принятые артефакты модели используют source revision
`3905cd51c3ed4e88c8edf33f3302969ba480a327`. Строгая FP32-конверсия достигла 100%
совпадения greedy-токенов на frozen parity set. Каждый production-артефакт строго
перезагружается и проверяется по записанному SHA-256.

Quality suite релиза `v0.1.0` использует зафиксированные поднаборы FLEURS `ru_ru`,
Russian LibriSpeech и SOVA RuDevices с единой нормализацией и одинаковым greedy
decoding. В отчёте Original PyTorch/MPS, FP16, INT8, INT6 и INT4 показаны рядом:
WER/CER, load/cold/warm, RTF, peak resident memory, MLX memory и swap. Приватное
пользовательское аудио не используется.

### Краткие результаты релизного теста

Apple M4 Pro (48 GB), macOS 15.7.7. Колонки WER: FLEURS / Russian LibriSpeech /
SOVA. Runtime — warm median на одном публичном пяти­минутном входе; скачивание не
учитывается. Меньше — лучше, кроме speedup.

| Реализация | Веса | WER: FLEURS / RuLibri / SOVA | Load | 5 мин, warm | Скорость к Original | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| Original PyTorch/MPS | 2.342 GB | 6.271% / 5.911% / 12.448% | 6.066s | 3.023s | 1.00× | 5.019 GB |
| MLX FP16 | 1.171 GB | 6.271% / 5.929% / 12.431% | 0.947s | 2.089s | 1.45× | 1.348 GB |
| **MLX INT8 g64 (по умолчанию)** | **0.699 GB** | **6.292% / 5.911% / 12.481%** | **0.578s** | **2.332s** | **1.30×** | **0.879 GB** |
| MLX INT6 g64 | 0.573 GB | 6.249% / 6.022% / 12.465% | 0.478s | 3.142s | 0.96× | 0.755 GB |
| MLX INT4 g64 | 0.447 GB | 6.358% / 6.003% / 12.581% | 0.378s | 2.750s | 1.10× | 0.628 GB |

INT8 выбран по умолчанию за баланс качества, размера и памяти. FP16 оказался
самым быстрым вариантом на репрезентативном пяти­минутном входе: квантизация не
обязана ускорять каждую длительность. Peak RSS — resident memory всего процесса,
а не только модель или дополнительная GPU-память.

Воспроизводимая публичная методика описана в
[`benchmarks/README.md`](benchmarks/README.md), результаты релиза — в
[`docs/benchmark-v0.1.0.md`](docs/benchmark-v0.1.0.md).

## Ограничения

- В `v0.1.0` доступен greedy CTC; beam search не добавлен.
- Таймкоды — приблизительные CTC emission times.
- Обучение, fine-tuning, diarization, microphone streaming и локальный HTTP-сервер
  не входят в этот релиз.
- Наилучшее качество ожидается на русском, казахском, кыргызском и узбекском;
  upstream сообщает умеренное качество английского. Авторитетный scope указан в
  исходной model card.
- Ограничения и bias исходных обучающих и оценочных данных сохраняются.

## Инструменты разработчика

Конверсия, квантизация, parity, оценка качества и benchmark вынесены из
production CLI и используют optional dependencies:

```bash
python -m pip install 'gigaam-multilingual-mlx[convert,quality]'
python -m gigaam_multilingual_mlx.dev_cli --help
```

Веса, датасеты, сгенерированное аудио и raw-результаты benchmark должны храниться
вне Git-репозитория.

## Лицензия, атрибуция и цитирование

Код репозитория распространяется по MIT. Происхождение кода и модели GigaAM, а
также уведомления о датасетах записаны в
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). При использовании модели
цитируйте оригинальную работу GigaAM-Multilingual по upstream model card; metadata
этого проекта находится в [`CITATION.cff`](CITATION.cff).

Правила сообщения об уязвимостях и внесения изменений описаны в
[`SECURITY.md`](SECURITY.md) и [`CONTRIBUTING.md`](CONTRIBUTING.md).
