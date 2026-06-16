# Распознавание объектов городской инфраструктуры в видеопотоках

---

## Аннотация

Репозиторий содержит программную реализацию, разработанную в рамках выпускной квалификационной работы, посвящённой автоматическому обнаружению объектов городской инфраструктуры в визуальных данных. Система предоставляет инструменты для выполнения инференса на изображениях и видеопотоках, визуализации результатов в браузерном интерфейсе, а также структурированного представления детекций объектов с учётом показателей уверенности, ориентированной геометрии и метаданных трекинга.

Реализованное решение основано на нейронной сети, экспортированной в формат ONNX и исполняемой посредством ONNX Runtime. Серверная часть разработана с использованием FastAPI, клиентская — в виде облегчённого веб-интерфейса для практического взаимодействия с системой.

Помимо функциональности инференса, репозиторий включает ноутбук `train.ipynb`, содержащий полный воспроизводимый конвейер обучения модели.

---

## Научная актуальность

Анализ городских видеопотоков представляет собой важное направление современных исследований в области компьютерного зрения. Городская среда генерирует значительные объёмы визуальной информации посредством камер наблюдения, систем мониторинга транспорта и инфраструктуры «умного города».

Методы автоматического обнаружения объектов на основе нейронных сетей позволяют выявлять значимые элементы городской инфраструктуры в изображениях и видеопотоках. Разработка воспроизводимой программной системы для обнаружения объектов городской инфраструктуры в видеопотоках представляет актуальную научную, инженерную и прикладную задачу.

---

## Цель и задачи работы

**Цель** — разработка воспроизводимой программной системы для обнаружения объектов городской инфраструктуры в изображениях и видеопотоках, включающей обучение модели, выполнение инференса и интерактивную визуализацию результатов.

**Задачи:**

1. Анализ применимости методов нейросетевого обнаружения объектов к городским визуальным данным.
2. Проектирование архитектуры программной системы для инференса на изображениях и видеозаписях.
3. Реализация серверного сервиса для выполнения инференса на основе ONNX-модели.
4. Разработка веб-интерфейса для загрузки, обработки и визуализации результатов.
5. Реализация межкадрового сопровождения объектов и пространственного анализа на основе полигонов зон контроля.

---

## Структура репозитория

```text
Urban-Infra-Detection/
├── README.md
├── .dockerignore
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── train.ipynb
├── backend/
│   ├── config.py
│   ├── dependencies.py
│   ├── main.py
│   ├── routes_health.py
│   ├── routes_inference.py
│   ├── routes_video.py
│   ├── schemas.py
│   ├── serializers.py
│   └── video_schemas.py
├── inference/
│   ├── __init__.py
│   ├── contracts.py
│   ├── engine.py
│   ├── model_manager.py
│   ├── postprocess.py
│   └── tracker.py
├── frontend/
│   ├── api.js
│   └── renderers.js
├── gui/
│   ├── app.js
│   ├── index.html
│   └── style.css
└── models/
    └── model.onnx
```

---

## Настройка порогов обнаружения

Качество детекций напрямую зависит от `CONFIDENCE_THRESHOLD`. Значение подбирается под конкретную сцену.

| Порог | Поведение |
|---|---|
| `0.001` | Максимум детекций, значимая доля ложных срабатываний |
| `0.1` | Умеренное количество, по-прежнему часть ложных |
| `0.25` | Рекомендуемое начальное значение |
| `0.5` | Только уверенные детекции |
| `0.7+` | Единичные высококачественные детекции |

`IOU_THRESHOLD` управляет подавлением дублирующихся рамок (NMS). Значение `0.45` подходит для большинства случаев. При плотных сценах (много объектов рядом) снизьте до `0.3`.

---

## Стек технологий

| Компонент | Технологии |
|---|---|
| Серверная часть | Python 3.11, FastAPI, Uvicorn, ONNX Runtime, OpenCV, NumPy, Pydantic |
| Клиентская часть | HTML5, CSS3, JavaScript |
| Контейнеризация | Docker, Docker Compose |
| Обучение модели | `train.ipynb` (Jupyter / Colab) |

---

## Конфигурация окружения

Приложение использует переменные окружения, хранящиеся в файле `.env`.

Шаблон конфигурации:

```bash
cp .env.example .env
```

### Основные параметры конфигурации

| Переменная | Описание | Значение по умолчанию |
|---|---|---|
| `MODEL_PATH` | Путь к файлу ONNX-модели | `./models/model.onnx` |
| `MODEL_INPUT_SIZE` | Размер входного тензора (px) | `1024` |
| `MODEL_LABELS` | Метки классов через запятую | *...* |
| `CONFIDENCE_THRESHOLD` | Порог уверенности детекции | `0.25` |
| `IOU_THRESHOLD` | Порог IoU для подавления немаксимумов | `0.45` |
| `ONNX_PROVIDERS` | Провайдеры выполнения ONNX Runtime | `CPUExecutionProvider` |
| `TRACKER_MAX_DISAPPEARED` | Допуск трекера к исчезновению объекта | `30` |
| `TRACKER_MAX_DISTANCE` | Максимальное расстояние сопоставления треков | `100.0` |
| `TRACKER_POLYGON` | Полигон зоны контроля в нормализованных координатах [0..1] | `0.2,0.2;0.8,0.2;0.8,0.8;0.2,0.8` |
| `DEFAULT_FPS` | Частота кадров по умолчанию | `30.0` |
| `MAX_IMAGE_SIZE_BYTES` | Максимальный размер изображения | `10485760` (10 МБ) |
| `MAX_VIDEO_SIZE_BYTES` | Максимальный размер видеозаписи | `209715200` (200 МБ) |
| `CORS_ALLOW_ORIGINS` | Разрешённые источники CORS | `http://localhost:3000,...` |

---

## Системные требования

### Для локального запуска

- Python 3.11
- `pip`
- Операционная система с поддержкой Python и OpenCV (Linux, macOS, Windows)

### Для запуска в Docker

- Docker Engine ≥ 24
- Docker Compose ≥ 2

### Для запуска в Colab

- Учётная запись
- При необходимости публичного доступа — учётная запись ngrok и токен авторизации

---

# Часть I. Локальный запуск

## Linux / macOS

### 1. Клонирование репозитория

```bash
git clone https://github.com/advri/Urban-Infra-Detection.git
cd Urban-Infra-Detection
```

### 2. Создание и активация виртуального окружения

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Установка зависимостей

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Создание файла конфигурации

```bash
cp .env.example .env
```

При необходимости отредактируйте `.env`, указав актуальный путь к модели и параметры инференса.

### 5. Запуск сервера

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 6. Открытие веб-интерфейса

После успешного запуска откройте в браузере:

```
http://127.0.0.1:8000/
```

Браузер автоматически перейдёт на страницу графического интерфейса обработки видео:

```
http://127.0.0.1:8000/gui/index.html
```

Документация API (Swagger UI) доступна по адресу:

```
http://127.0.0.1:8000/docs
```

---

## Windows (PowerShell)

### 1. Клонирование репозитория

```powershell
git clone https://github.com/advri/Urban-Infra-Detection.git
cd Urban-Infra-Detection
```

### 2. Создание и активация виртуального окружения

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Установка зависимостей

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Создание файла конфигурации

```powershell
Copy-Item .env.example .env
```

### 5. Запуск сервера

```powershell
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 6. Открытие веб-интерфейса

После успешного запуска откройте в браузере:

```
http://127.0.0.1:8000/
```

---

# Часть II. Запуск в Docker

## Вариант A. Docker Compose (рекомендуется)

Docker Compose автоматически монтирует директорию с моделью и управляет переменными окружения.

### 1. Подготовка конфигурации

```bash
cp .env.example .env
```

### 2. Запуск контейнера

```bash
docker compose up --build
```

Для запуска в фоновом режиме:

```bash
docker compose up --build -d
```

### 3. Открытие веб-интерфейса

```
http://127.0.0.1:8000/
```

### 4. Остановка

```bash
docker compose down
```

---

## Вариант B. Docker без Compose

### 1. Сборка образа

```bash
docker build -t urban-infra-detection .
```

### 2. Запуск контейнера (Linux / macOS)

```bash
docker run --rm -it \
  -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/models:/app/models:ro" \
  -v uid-artifacts:/app/artifacts \
  urban-infra-detection
```

### 3. Запуск контейнера (Windows PowerShell)

```powershell
docker run --rm -it `
  -p 8000:8000 `
  --env-file .env `
  -v "${PWD}\models:/app/models:ro" `
  -v uid-artifacts:/app/artifacts `
  urban-infra-detection
```

### 4. Открытие веб-интерфейса

```
http://127.0.0.1:8000/
```

---

# Часть III. Запуск в Colab

Colab позволяет запустить серверное приложение в облачной среде без локальной установки. Для организации публичного доступа к интерфейсу используется сервис ngrok.

## Ячейка 1. Установка системных зависимостей и клонирование репозитория

```python
# Установка системных библиотек OpenCV и ffmpeg
# ffmpeg требуется для корректного воспроизведения аннотированных видео в браузере
import subprocess
subprocess.run(["apt-get", "update", "-y"], check=True)
subprocess.run(["apt-get", "install", "-y", "libgl1", "libglib2.0-0", "ffmpeg", "libx264-dev"], check=True)

# Клонирование репозитория
REPO_URL = "https://github.com/advri/Urban-Infra-Detection.git"
REPO_DIR = "Urban-Infra-Detection"

import os
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

os.chdir(REPO_DIR)
print("Рабочая директория:", os.getcwd())
```

## Ячейка 2. Проверка наличия модели

```python
import os
print("Модель существует:", os.path.exists("models/model.onnx"))
```

## Ячейка 3. Установка зависимостей Python

```python
import subprocess
subprocess.run(["pip", "install", "--upgrade", "pip"],         check=True)
subprocess.run(["pip", "install", "-r", "requirements.txt"],  check=True)
subprocess.run(["pip", "install", "pyngrok"],                 check=True)
print("Зависимости установлены. Проверка ключевых пакетов:")
for pkg in ["fastapi", "pydantic", "cv2", "PIL", "onnxruntime"]:
    try:
        m = __import__(pkg)
        print(f"  {pkg}: {getattr(m, '__version__', 'OK')}")
    except ImportError:
        print(f"  {pkg}: НЕ УСТАНОВЛЕН")
print("Зависимости установлены.")
```

## Ячейка 4. Создание файла конфигурации

```python
import shutil, os

if not os.path.exists(".env"):
    shutil.copy(".env.example", ".env")
    print("Файл .env создан на основе .env.example")
else:
    print("Файл .env уже существует")
```

При необходимости скорректируйте параметры непосредственно:

```python
# Опционально: ручная правка отдельных параметров
env_overrides = {
    "MODEL_PATH": "./models/model.onnx",
    "CONFIDENCE_THRESHOLD": "0.25",
    "ONNX_PROVIDERS": "CPUExecutionProvider",
}

with open(".env", "r") as f:
    lines = f.readlines()

with open(".env", "w") as f:
    for line in lines:
        key = line.split("=")[0].strip()
        if key in env_overrides:
            f.write(f"{key}={env_overrides[key]}\n")
        else:
            f.write(line)

print("Параметры конфигурации обновлены.")
```

## Ячейка 4б. Проверка чтения конфигурации

Убедитесь, что переменные из `.env` читаются корректно:

```python
from dotenv import dotenv_values
cfg = dotenv_values(".env")
print("MODEL_LABELS:", cfg.get("MODEL_LABELS", "(не задан)"))
print("MODEL_PREDECODED:", cfg.get("MODEL_PREDECODED"))
print("CONFIDENCE_THRESHOLD:", cfg.get("CONFIDENCE_THRESHOLD"))
# Если MODEL_LABELS пустой — заполните .env согласно инструкции в разделе «Конфигурация окружения»
```

## Ячейка 5. Запуск сервера в фоновом режиме

```python
import subprocess, time, os

# Запуск uvicorn в фоне
server = subprocess.Popen(
    ["python", "-m", "uvicorn", "backend.main:app",
     "--host", "0.0.0.0", "--port", "8000"],
    stdout=open("server.log", "w"),
    stderr=subprocess.STDOUT,
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
)

print(f"Сервер запущен (PID {server.pid}). Ожидание инициализации…")
time.sleep(8)

# Проверка журнала запуска
with open("server.log") as f:
    print(f.read()[-2000:])
```

## Ячейка 5а. Альтернатива ngrok — cloudflared (без таймаута)

```python
import subprocess, time, re

# Установка cloudflared
subprocess.run(["wget", "-q", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "-O", "/usr/local/bin/cloudflared"], check=True)
subprocess.run(["chmod", "+x", "/usr/local/bin/cloudflared"], check=True)

# Запуск туннеля
tunnel_proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://localhost:8000"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)

time.sleep(5)
public_url = None
for _ in range(20):
    line = tunnel_proc.stdout.readline()
    match = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", line)
    if match:
        public_url = match.group(0)
        break

interface_url = f"{public_url}/gui/index.html"
print("Веб-интерфейс:", interface_url)

from IPython.display import display, HTML, IFrame
display(HTML(f'<a href="{interface_url}" target="_blank" style="padding:12px 24px;background:#1a4b7a;color:#fff;border-radius:8px;text-decoration:none;">Открыть интерфейс</a>'))
display(IFrame(src=interface_url, width="100%", height=700))
```

## Ячейка 5б. Диагностика среды (выполнить при ошибках)

Если сервер запустился, но API возвращает ошибки — выполните эту ячейку:

```python
import requests, json

# Проверка через диагностический эндпоинт
r = requests.get("http://localhost:8000/api/health/debug")
print(json.dumps(r.json(), indent=2, ensure_ascii=False))
```

Вывод покажет версии всех пакетов, доступность шрифтов для рендеринга,
наличие ffmpeg и статус инициализации runtime.

При ошибках в server.log найдите строки `ERROR` — все исключения
логируются.

## Ячейка 6. Публикация интерфейса через ngrok и открытие в браузере

```python
from pyngrok import ngrok
from IPython.display import display, IFrame, HTML
import time

# Укажите ваш токен авторизации ngrok.
# Получить токен бесплатно: https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTH_TOKEN = "ВАШ_ТОКЕН_NGROK"

ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Создание публичного туннеля
tunnel = ngrok.connect(8000)
public_url = tunnel.public_url
interface_url = f"{public_url}/gui/index.html"

print("=" * 60)
print("Сервер запущен и доступен по адресу:")
print(f"  {public_url}/")
print()
print("Веб-интерфейс обработки видео:")
print(f"  {interface_url}")
print()
print("Документация API:")
print(f"  {public_url}/docs")
print("=" * 60)

# Встраивание интерфейса непосредственно в ячейку Colab
display(HTML(f"""
<div style="margin: 16px 0;">
  <a href="{interface_url}" target="_blank"
     style="display:inline-block; padding:12px 24px; background:#1f4e79;
            color:#fff; border-radius:8px; text-decoration:none; font-size:15px;
            font-family:sans-serif;">
    Открыть интерфейс обработки видео
  </a>
</div>
"""))

display(IFrame(src=interface_url, width="100%", height=700))
```

### Порядок работы с интерфейсом

После открытия интерфейса доступны следующие операции:

1. В блоке **«Параметры обработки»** выбрать режим: **Изображение** или **Видеозапись**.
2. Нажать **«Файл»** и загрузить исходный файл.
3. При обработке видеозаписи при необходимости задать шаг дискретизации, предел кадров и включить трекинг объектов.
4. Нажать **«Запустить анализ»** для получения структурированных результатов детекции (таблица, сводка, JSON).
5. Нажать **«Получить аннотированный результат»** для загрузки визуализированного изображения или видеозаписи с нанесёнными рамками обнаружений.

## Ячейка 7. Остановка сервера (по завершении работы)

```python
from pyngrok import ngrok as _ngrok

# Закрытие туннеля ngrok (если использовался)
try:
    _ngrok.disconnect(tunnel.public_url)
    _ngrok.kill()
except Exception:
    pass

# Закрытие туннеля cloudflared (если использовался)
try:
    tunnel_proc.terminate()
except Exception:
    pass

# Остановка процесса uvicorn
server.terminate()
server.wait()
print("Сервер остановлен.")
```

---

# Часть IV. Обучение модели: локальный запуск `train.ipynb`

Репозиторий включает ноутбук `train.ipynb`, содержащий полный воспроизводимый цикл обучения модели.

## Рекомендуемый порядок запуска

### 1. Клонирование репозитория

```bash
git clone https://github.com/advri/Urban-Infra-Detection.git
cd Urban-Infra-Detection
```

### 2. Создание отдельного виртуального окружения для обучения

```bash
python3.11 -m venv .venv-train
source .venv-train/bin/activate   # Linux / macOS
# .venv-train\Scripts\Activate.ps1  # Windows PowerShell
```

### 3. Установка Jupyter

```bash
python -m pip install --upgrade pip
pip install notebook
```

### 4. Запуск Jupyter Notebook

```bash
jupyter notebook
```

Откройте `train.ipynb` и последовательно выполните ячейки.

---

# Часть V. Обучение модели: запуск `train.ipynb` в Colab

Ноутбук `train.ipynb` расположен в корне репозитория и может быть выполнен в Colab.

## Вариант A. Открытие ноутбука из GitHub

1. Откройте страницу репозитория на GitHub.
2. Выберите файл `train.ipynb`.
3. Нажмите кнопку «Open in Colab».
4. Последовательно выполните ячейки ноутбука.

## Вариант B. Клонирование репозитория в Colab

```python
REPO_URL = "https://github.com/advri/Urban-Infra-Detection.git"
REPO_DIR = "Urban-Infra-Detection"

import subprocess, os
subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)
print("Структура директории:")
print(os.listdir("."))
```

Затем откройте и выполните `train.ipynb`.

---

# Воспроизводимость

Репозиторий обеспечивает два уровня воспроизводимости:

**Воспроизводимость инференса.** Файл `models/model.onnx` позволяет немедленно запустить инференс без повторного обучения модели.

**Воспроизводимость обучения.** Ноутбук `train.ipynb` содержит полный задокументированный конвейер обучения, допускающий независимое воспроизведение экспериментальных результатов.

Данная структура соответствует требованиям, предусматривающим одновременно научную прозрачность и практическую верифицируемость.

---

# Использование GPU

Конфигурация по умолчанию ориентирована на выполнение инференса на центральном процессоре (CPU).

Для перехода на GPU необходимо:

1. Заменить пакет `onnxruntime` на `onnxruntime-gpu` в `requirements.txt`.
2. Использовать среду с поддержкой CUDA.
3. Задать в `.env`:
   ```env
   ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
   ```
4. При использовании Docker — применить базовый образ с поддержкой CUDA.

---

# Устранение неисправностей

### Веб-интерфейс недоступен после запуска сервера

- Убедитесь, что сервер запустился без ошибок импорта (проверьте журнал `server.log`).
- Порт `8000` должен использоваться согласованно во всех командах.
- В Colab — убедитесь, что туннель ngrok создан и публичный URL указан корректно.
- Перейдите непосредственно по адресу `http://127.0.0.1:8000/gui/index.html`.

### Ошибки, связанные с OpenCV (libGL, libglib2.0)

Установите системные зависимости:

```bash
# Linux / Colab
apt-get install -y libgl1 libglib2.0-0 ffmpeg
```

Данные пакеты включены в Dockerfile и в Colab-ячейки настоящего руководства.

### Ошибка 503 при обращении к API (Runtime not initialized)

Сервер запустился, однако модель не была загружена. Проверьте:

- наличие файла `models/model.onnx`;
- корректность переменной `MODEL_PATH` в `.env`;
- отсутствие ошибок в журнале сервера, связанных с инициализацией ONNX Runtime.

### Видеозапись не обрабатывается (ошибка 413)

Размер загружаемой видеозаписи превышает допустимый предел. Увеличьте значение переменной `MAX_VIDEO_SIZE_BYTES` в `.env`:

```env
MAX_VIDEO_SIZE_BYTES=524288000
```

### Ошибки при выполнении ноутбука `train.ipynb`

Ноутбук может требовать установки дополнительных зависимостей, характерных для экспериментального окружения. Установите необходимые пакеты согласно импортам, определённым внутри ноутбука.

---

## Использование репозитория

Данный репозиторий является формальной программной реализацией. Он демонстрирует:

- проектирование системы обнаружения объектов городской инфраструктуры;
- реализацию серверного API для выполнения инференса;
- интеграцию браузерного интерфейса визуализации результатов;
- развёртывание ONNX-модели для практического применения;
- наличие воспроизводимого конвейера обучения;
- связь между экспериментальными исследованиями и развёртываемым программным обеспечением.

Репозиторий может быть использован как для демонстрации, так и для независимой верификации результатов реализации.
