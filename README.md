# Modeling-the-structure-of-polycrystalline-materials

# Требования к установленным программам:
Python 3.9+ с pip
Node.js 16+
Visual Studio Code (опционально)
Anaconda Jupyter (опционально)
# Зависимости:
fastapi==0.115.12
uvicorn==0.34.2
python-multipart==0.0.20
pillow==11.2.1
numpy==1.26.4
torch==2.7.0
torchvision==0.22.0
pandas==2.2.3
orix==0.13.0
# Шаги по установки и запуску проекта:
1.	Создать и активировать виртуальное окружение venv: 
python -m venv venv
.\venv\Scripts\activate
2.	Установить зависимости для backend:
pip install -r backend/requirements.txt
3.	установить зависимости для frontend:
cd frontend
npm install
4.	Запустить backend-сервер из директории проекта:
cd backend
uvicorn app.main:app –reload
5.	Запустить frontend:
cd frontend
npm run dev
6.	Собрать проект: 
cd frontend
npm run build
# Приложение будет доступно:
Backend: http://localhost:8000
Frontend: http://localhost:5173
# Конфигурация:
1.	Для настройки API URL необходимо отредактировать файл frontend/.env: VITE_API_BASE_URL=/api
2.	API endpoints редактируются в файле main.py
# Решение проблем при установке:
1. При установке некоторых библиотек (например, Orix) может возникнуть ошибка компиляции C++, для её решения требуется установить компилятор MSVC, где необходимо указать готовую конфигурацию Desktop development with C++: https://visualstudio.microsoft.com/ru/visual-cpp-build-tools/
2. Убедитесь, что установлен PyTorch с поддержкой cuda, например следующей командой:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
3. Для быстрого запуска тестирования на операционной системе Windows в исходном коде проекта имеется файл run.bat, при нажатии на который автоматически загружаются backend и frontend.
