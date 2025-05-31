@echo off
title EBSD-Reconstruction Launcher
color 0a
echo Starting EBSD-Reconstruction application...

:: Активация виртуального окружения
echo Activating Python virtual environment...
call .\venv\Scripts\activate

:: Запуск бэкенда в новом окне
echo Starting backend server...
start "Backend" cmd /k "cd %cd% && uvicorn backend.app.main:app --reload"

:: Запуск фронтенда в новом окне
echo Starting frontend server...
cd frontend
start "Frontend" cmd /k "npm run dev"

echo Application started successfully!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
pause