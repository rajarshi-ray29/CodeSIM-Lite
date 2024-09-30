@echo off
set retryDelay=10
set command=python .\src\main.py --strategy SCoder --model ChatGPT
@REM set command=python .\src\main.py --strategy SelfPlanning --model GPT4T

:: Activate the Conda environment
call conda activate scoder

:run
echo Running command: %command%
%command%

if %ERRORLEVEL% neq 0 (
    echo An error occurred. Retrying in %retryDelay% seconds...
    timeout /t %retryDelay% /nobreak
    goto run
)

echo Command executed successfully!
pause
