@echo off
rem Borra los modulos antiguos de tfm_comun que se han separado en carpetas (2026-11-12).
rem Su codigo ya esta en los ficheros nuevos; ningun notebook los importa.
cd /d "%~dp0tfm_comun"
for %%f in (red\datos.py red\entrada.py red\aumento.py red\secuencias.py red\entrenamiento.py red\preentreno.py red\registro.py red\dcrnn.py red\corpus.py prediccion\metricas.py prediccion\atajos.py prediccion\contexto.py) do if exist %%f (del "%%f" & echo borrado %%f)
for /d /r %%d in (__pycache__) do if exist "%%d" rd /s /q "%%d"
echo Listo.
pause
