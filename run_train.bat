@echo off
cd /d C:\Users\shadb\Downloads\dataset
:loop
python -u train_all.py >> train_log.txt 2>&1
if exist results\ALL_DONE.flag goto done
echo [watchdog] python exited, relaunching in 15s >> train_log.txt
timeout /t 15 /nobreak >nul
goto loop
:done
echo [watchdog] ALL DONE >> train_log.txt
