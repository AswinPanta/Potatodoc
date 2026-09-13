@echo off
cd /d C:\Users\shadb\Downloads\dataset
:waitdl
if exist results\EXTRACT_DONE.flag goto train
echo [watchdog-v3] waiting for download+extract... >> train_v3_log.txt
timeout /t 120 /nobreak >nul
goto waitdl
:train
python -u train_v3.py >> train_v3_log.txt 2>&1
if exist results\V3_DONE.flag goto done
echo [watchdog-v3] relaunching training in 15s >> train_v3_log.txt
timeout /t 15 /nobreak >nul
goto train
:done
echo [watchdog-v3] V3 ALL DONE >> train_v3_log.txt
