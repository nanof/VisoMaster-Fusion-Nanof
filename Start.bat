@echo off
REM Check if .venv directory exists
IF EXIST ".venv" (
    echo Found .venv, activating virtual uv based environment...
    call ".venv\Scripts\activate"
) ELSE (
    echo .venv not found, activating conda environment "visomaster"...
    call conda activate visomaster
)

REM Run main.py
echo Running VisoMaster...
python main.py
SET EXIT_CODE=%ERRORLEVEL%

REM Keep the console open after a crash so users can read the error output.
REM Exit code 0 = clean exit (user closed the window normally).
IF %EXIT_CODE% NEQ 0 (
    echo.
    echo [ERROR] VisoMaster exited with code %EXIT_CODE%.
    echo         Review the output above for details, then press any key to close.
    pause >nul
)
