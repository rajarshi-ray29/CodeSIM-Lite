@echo off
set retryDelay=60
set command=python .\src\main.py --strategy SCoder --model LLaMa8B

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
