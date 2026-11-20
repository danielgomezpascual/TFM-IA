@echo off
setlocal

rem ============================================================
rem  Reorganizacion por corpus, pasos A y B
rem
rem  A) Notebooks:  ALMA\fase_0X\*          -> ALMA\fase_0X\tusz\
rem     (notebooks, img, README y guias: todo lo que hay en la fase)
rem  B) Salidas:    ALMA\outputs\fase_0X\*  -> ALMA\outputs\fase_0X\tusz\
rem     (outputs\comun no se toca)
rem
rem  Mismo disco: mover es instantaneo (no se copian los 36 GB).
rem  Antes: cerrar Jupyter, VS Code y PyCharm.
rem  Despues: NO ejecutar notebooks hasta terminar el paso C
rem  (herramientas\actualizar_referencias.py).
rem  Deja un registro en herramientas\mover_tusz_por_corpus.log
rem ============================================================

cd /d "%~dp0.."
set "LOG=%~dp0mover_tusz_por_corpus.log"
echo Inicio %date% %time% > "%LOG%"
echo Carpeta ALMA: %cd%
echo.

rem 1) Comprobacion: que no se haya hecho ya
for %%N in (01 02 03) do (
    if exist "fase_%%N\tusz\" goto :ya_hecho
    if exist "outputs\fase_%%N\tusz\" goto :ya_hecho
)

rem 2) Mover cada fase
for %%N in (01 02 03) do call :mover "fase_%%N"
for %%N in (01 02 03) do call :mover "outputs\fase_%%N"

echo.
echo Terminado. Revisa que no haya lineas [ERROR] arriba.
echo Siguiente paso: herramientas\actualizar_referencias.py (paso C).
echo Fin >> "%LOG%"
goto :fin


rem ---- Mueve todo lo que hay en %1 a %1\tusz ----
:mover
if not exist "%~1\" (
    echo [AVISO] No existe %~1, se salta.
    exit /b
)
mkdir "%~1\tusz" >> "%LOG%" 2>&1
for %%X in ("%~1\*") do move "%%X" "%~1\tusz\" >> "%LOG%" 2>&1
for /d %%D in ("%~1\*") do if /i not "%%~nxD"=="tusz" move "%%D" "%~1\tusz\" >> "%LOG%" 2>&1
set "RESTO="
for %%X in ("%~1\*") do set "RESTO=1"
for /d %%D in ("%~1\*") do if /i not "%%~nxD"=="tusz" set "RESTO=1"
if defined RESTO (
    echo [ERROR] En %~1 queda algo sin mover. Mira el .log
) else (
    echo   OK: %~1\tusz
)
exit /b


:ya_hecho
echo [ERROR] Ya existe alguna carpeta tusz. Parece que ya se movio; no toco nada.
goto :fin

:fin
echo.
pause
