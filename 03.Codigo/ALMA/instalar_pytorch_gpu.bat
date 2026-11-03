@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo Cambiando PyTorch por la version con GPU (CUDA) en .venv...
echo Descarga de unos 2,5 GB: puede tardar varios minutos. No cierres esta ventana.
".venv\Scripts\python.exe" -c "import sys; sys.path.insert(0, '.'); from tfm_comun.entorno.requisitos import reinstalar_torch_gpu, torch_usa_gpu, hay_gpu_nvidia, cuda_del_driver; print('Tarjeta NVIDIA:', hay_gpu_nvidia(), '| CUDA que admite el driver:', cuda_del_driver()); reinstalar_torch_gpu(); print('RESULTADO:', torch_usa_gpu())" > instalar_pytorch_gpu.log 2>&1
type instalar_pytorch_gpu.log
echo.
pause
