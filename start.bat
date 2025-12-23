@echo off
chcp 65001 > nul
echo ==============================================
echo           Z时代客群分析系统一键启动
echo ==============================================
echo.

:: 1. 安装/升级依赖
echo 📌 正在安装项目依赖...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo 🔴 依赖安装失败！请手动执行pip install -r requirements.txt
    pause
    exit /b 1
)
echo 🟢 依赖安装完成！
echo.

:: 2. 启动Flask服务
echo 📌 正在启动Z时代客群分析系统...
echo 📌 服务地址：http://localhost:5000
echo 📌 停止服务请按 Ctrl+C
echo.
python app.py

:: 3. 异常兜底
if errorlevel 1 (
    echo.
    echo 🔴 服务启动失败！请检查app.py代码或密钥配置
    pause
)
pause