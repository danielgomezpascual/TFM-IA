@echo off
chcp 65001 >nul
setlocal

rem ============================================================
rem  Reorganiza los datasets del TFM bajo 00.Corpus
rem
rem    00.TUSZ\v2.0.6                                 -> 00.Corpus\TUSZ\v2.0.6
rem    chbmit-1.0.0\chb-mit-scalp-eeg-database-1.0.0  -> 00.Corpus\CHB-MIT\1.0.0
rem    chbmit-1.0.0.zip                               -> 00.Corpus\CHB-MIT\chbmit-1.0.0.zip
rem
rem  Todo está en el mismo disco, así que mover es instantáneo
rem  (solo cambia la ruta, no se copian datos).
rem  Antes de lanzarlo: cierra Jupyter, VS Code y PyCharm, para que
rem  ningún EDF esté abierto.
rem ============================================================

set "RAIZ=%~dp0"
set "CORPUS=%RAIZ%00.Corpus"
set "TUSZ_ORIGEN=%RAIZ%00.TUSZ\v2.0.6"
set "CHB_ORIGEN=%RAIZ%chbmit-1.0.0\chb-mit-scalp-eeg-database-1.0.0"
set "CHB_ZIP=%RAIZ%chbmit-1.0.0.zip"

echo Carpeta del proyecto: %RAIZ%
echo.

rem 1) Comprobaciones: que existan los orígenes y que no existan los destinos
if not exist "%TUSZ_ORIGEN%" (
    echo [ERROR] No encuentro %TUSZ_ORIGEN%
    goto :fin
)
if not exist "%CHB_ORIGEN%" (
    echo [ERROR] No encuentro %CHB_ORIGEN%
    goto :fin
)
if exist "%CORPUS%\TUSZ\v2.0.6" (
    echo [ERROR] Ya existe %CORPUS%\TUSZ\v2.0.6. No toco nada.
    goto :fin
)
if exist "%CORPUS%\CHB-MIT\1.0.0" (
    echo [ERROR] Ya existe %CORPUS%\CHB-MIT\1.0.0. No toco nada.
    goto :fin
)

rem 2) Crear la estructura nueva
mkdir "%CORPUS%\TUSZ" 2>nul
mkdir "%CORPUS%\CHB-MIT" 2>nul

rem 3) Mover TUSZ
echo Moviendo TUSZ...
move "%TUSZ_ORIGEN%" "%CORPUS%\TUSZ\v2.0.6" >nul
if errorlevel 1 (
    echo [ERROR] No se pudo mover TUSZ. Puede que algún fichero esté abierto.
    goto :fin
)
echo   OK: 00.Corpus\TUSZ\v2.0.6

rem 4) Mover CHB-MIT
echo Moviendo CHB-MIT...
move "%CHB_ORIGEN%" "%CORPUS%\CHB-MIT\1.0.0" >nul
if errorlevel 1 (
    echo [ERROR] No se pudo mover CHB-MIT. TUSZ ya está en su sitio nuevo.
    goto :fin
)
echo   OK: 00.Corpus\CHB-MIT\1.0.0

rem 5) Mover el ZIP de CHB-MIT (si sigue ahí)
if exist "%CHB_ZIP%" (
    move "%CHB_ZIP%" "%CORPUS%\CHB-MIT\chbmit-1.0.0.zip" >nul
    if errorlevel 1 (
        echo [AVISO] No se pudo mover el ZIP; se queda donde estaba.
    ) else (
        echo   OK: 00.Corpus\CHB-MIT\chbmit-1.0.0.zip
    )
)

rem 6) Borrar las carpetas viejas solo si se han quedado vacías
rmdir "%RAIZ%00.TUSZ" 2>nul
rmdir "%RAIZ%chbmit-1.0.0" 2>nul
if exist "%RAIZ%00.TUSZ" echo [AVISO] 00.TUSZ no está vacía; revísala a mano.
if exist "%RAIZ%chbmit-1.0.0" echo [AVISO] chbmit-1.0.0 no está vacía; revísala a mano.

echo.
echo Reorganización terminada.

:fin
echo.
pause
