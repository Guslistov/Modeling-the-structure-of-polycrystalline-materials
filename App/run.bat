@echo off
title EBSD-Reconstruction Launcher
color 0a
echo Starting EBSD-Reconstruction application...

:: Активация виртуального окружения (для текущего окна, необязательно)
call .\venv\Scripts\activate

:: Запуск бэкенда в новом окне с активацией окружения
echo Starting backend server...
start "Backend" cmd /k ".\venv\Scripts\python -m uvicorn backend.app.main:app --reload"

:: Запуск фронтенда в новом окне
echo Starting frontend server...
cd frontend
start "Frontend" cmd /k "npm run dev"

echo Application started successfully!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
pause
