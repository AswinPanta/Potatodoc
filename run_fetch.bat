@echo off
cd /d C:\Users\shadb\Downloads\dataset
:loop
python -u fetch_ipd37.py >> fetch_ipd37_log.txt 2>&1
if exist results\EXTRACT_DONE.flag goto done
echo [watchdog-dl] relaunching in 15s >> fetch_ipd37_log.txt
timeout /t 15 /nobreak >nul
goto loop
:done
echo [watchdog-dl] ALL DONE >> fetch_ipd37_log.txt
