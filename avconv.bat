@echo off
chcp 65001 >nul
REM AV1 → H.264 (NVENC). Requiere: python, ffmpeg, ffprobe en PATH.
REM Uso: avconv.bat [opciones]
REM Ejemplos:
REM   avconv.bat
REM   avconv.bat --dry-run
REM   avconv.bat -d "C:\Videos" -y
REM   avconv.bat -j 2 --size-match

REM No cambiar directorio: el escaneo usa la carpeta actual (o -d).

where python >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Python no encontrado en el PATH.
    echo         Instala Python o anade python.exe al PATH del sistema.
    pause >nul
    exit /b 1
)

python "%~dp0avconv.py" %*
SET EXIT_CODE=%ERRORLEVEL%

IF %EXIT_CODE% NEQ 0 (
    echo.
    echo [ERROR] avconv salio con codigo %EXIT_CODE%.
    echo         Revisa la salida anterior. Pulsa una tecla para cerrar.
    pause >nul
)

exit /b %EXIT_CODE%
