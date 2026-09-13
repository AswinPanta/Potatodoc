@echo off
cd /d C:\Users\shadb\Downloads\dataset
:loop
python -u train_v2.py >> train_v2_log.txt 2>&1
if exist results\V2_DONE.flag goto done
echo [watchdog-v2] relaunching in 15s >> train_v2_log.txt
timeout /t 15 /nobreak >nul
goto loop
:done
echo [watchdog-v2] ALL DONE >> train_v2_log.txt
