# Ferma — Farm LLM + CV Service

FastAPI сервис для фермера: чат‑помощник на Gemini и компьютерное зрение для анализа кадров с фермы.

Проект включает:
- LLM‑эндпойнт `/ask-question` (Gemini API)
- CV‑эндпойнт `/analyze` (YOLO, `models/prod1.pt`)
- Скрипт подготовки/обучения YOLO
- Наборы данных и утилиты для тестов

---

## Архитектура

- `main.py` — FastAPI приложение и эндпойнты
- `ai_services/llm.py` — клиент Gemini API
- `ai_services/computer_vision.py` — YOLO инференс
- `schemas.py` — Pydantic модели и парсинг `add_info`
- `train/finetune_yolo26m_pmfeed.py` — подготовка датасета и обучение
- `tests/` — скрипты для ручной проверки

---

## Требования

- Python 3.12
- CUDA (опционально, для ускорения)
- Зависимости из `requirements.txt`

Установка:
```bash
pip install -r requirements.txt
```

---

## Конфигурация LLM

Создай `.env` в корне проекта:
```
GEMINI_API_KEY=YOUR_KEY
GEMINI_MODEL=gemini-3-pro-preview
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
GEMINI_TIMEOUT_S=30
```

---

## Запуск сервера

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

---

## Эндпойнты

### 1) `GET /ask-question`

Query параметры:
- `user_text` (строка, обязательный)
- `history` (JSON‑строка списка словарей, опционально)

Пример:
```bash
curl -G "http://127.0.0.1:8000/ask-question" \
  --data-urlencode "user_text=Как лечить мастит у коровы?" \
  --data-urlencode 'history=[{"Вопрос":"Ответ"}]'
```

Ответ:
```json
{ "llm_answer": "..." }
```

---

### 2) `GET /analyze` (или `POST /analyze`)

Формат: `multipart/form-data`
- `image` — файл (JPEG/PNG), до 5 MB
- `add_info` — JSON‑строка **списка словарей** (например `[{"a":1}]`)

Пример:
```bash
curl -X GET "http://127.0.0.1:8000/analyze" \
  -F "image=@/path/to/frame.jpg" \
  -F 'add_info=[{"camera":"barn-1"}]'
```

Ответ:
```json
{
  "cows_num": 10,
  "ill_cow": [],
  "hunter": [[x1, x2, y1, y2], ...],
  "thief": [[x1, x2, y1, y2], ...],
  "pregnant": [[x1, x2, y1, y2], ...],
  "info": {}
}
```

Маппинг классов (YOLO):
- `cow` → `cows_num`
- `cow` → `pregnant` (список боксов коров)
- `wolf` → `hunter` (список боксов)
- `person` → `thief` (список боксов)

**Важно:** координаты в формате `[x1, x2, y1, y2]`.

---

## Конкурентный доступ

Одновременно обрабатывается только один запрос на эндпойнт.
Если запрос уже в обработке, следующий получит:
```
HTTP 429 Too Many Requests
```

---

## Обучение модели

Скрипт: `train/finetune_yolo26m_pmfeed.py`

Пример запуска на `data/main_dataset` (20 эпох):
```bash
python train/finetune_yolo26m_pmfeed.py \
  --model-path models/cow_best_yolo.pt \
  --images-dir data/main_dataset/images \
  --labels-dir data/main_dataset/labels \
  --output-dir train/datasets/main_dataset_yolo \
  --epochs 20 \
  --classes-path data/annotated_labels/classes.txt
```

Скрипт поддерживает `.png`, `.jpg`, `.jpeg`.

---

## Тестовые скрипты

- `tests/analyze_video_test.py` — отправка кадров из видео в `/analyze`
- `tests/run_yolo_video.py` — запуск YOLO по видео и сохранение кадров

---

## Модель и данные

- Веса инференса: `models/prod1.pt`
- Датасет на Hugging Face: `I77/ferma-camera-yolo`
- Пример обучения: `runs/detect/train/runs/*/weights/best.pt`

---

## Замечания

- При первом вызове `/analyze` модель может грузиться долго.
- Файлы >5 MB будут отклонены.
- Для стабильной работы real‑time лучше отправлять кадры с учётом скорости инференса.
