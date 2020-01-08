@ECHO OFF
SET ANA_PATH="C:\Users\rmappin\temp_anaconda\Scripts"
SET MAT_PATH="C:\Program Files\MATLAB\R2019a\bin"
SET DIRECTORY="C:\Users\rmappin\OneDrive - University College London\PhD\PhD_Prog\010_CNN_SISR\scripts\pre-post-proc"
SET M_FILE_PATH="C:\Users\rmappin\PhD_Data\Super_Res_Raw_Data\Toshiba"
SET M_SAVE_PATH="Z:\Super_Res_Data\Toshiba\mat"
SET P_FILE_PATH="Z:\Super_Res_Data\Toshiba\mat"
SET P_SAVE_PATH="Z:\Super_Res_Data\Toshiba"

CD %DIRECTORY%
CALL %MAT_PATH%\matlab.exe -wait -nosplash -nodisplay -noFigureWindows -r "simLoResGen %M_FILE_PATH% %M_SAVE_PATH%; quit"

CD %ANA_PATH%
CALL activate.bat base

CD %DIRECTORY%
CALL python MAT_NPY_Conv.py -f %P_FILE_PATH% -s %P_SAVE_PATH%

pause