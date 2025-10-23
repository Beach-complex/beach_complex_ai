@ECHO OFF

SETLOCAL

SET DIRNAME=%~dp0
IF NOT "%DIRNAME%"=="" SET DIRNAME=%DIRNAME:~0,-1%
SET APP_BASE_NAME=%~n0
SET APP_HOME=%DIRNAME%

SET DEFAULT_JVM_OPTS=

SET CLASSPATH=%APP_HOME%\gradle\wrapper\gradle-wrapper.jar

IF DEFINED JAVA_HOME GOTO findJavaFromJavaHome

SET JAVA_EXE=java.exe
%JAVA_EXE% -version >NUL 2>&1
IF "%ERRORLEVEL%" == "0" GOTO execute

ECHO. 1>&2
ECHO ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH. 1>&2
ECHO. 1>&2
GOTO fail

:findJavaFromJavaHome
SET JAVA_HOME=%JAVA_HOME:"=%
SET JAVA_EXE=%JAVA_HOME%\bin\java.exe

IF EXIST "%JAVA_EXE%" GOTO execute

ECHO. 1>&2
ECHO ERROR: JAVA_HOME is set to an invalid directory: %JAVA_HOME% 1>&2
ECHO. 1>&2
GOTO fail

:execute
SET COMMAND_LINE_ARGS=
IF NOT "%@%"=="" SET COMMAND_LINE_ARGS=%*

"%JAVA_EXE%" %DEFAULT_JVM_OPTS% -classpath "%CLASSPATH%" org.gradle.wrapper.GradleWrapperMain %COMMAND_LINE_ARGS%
GOTO end

:fail
EXIT /B 1

:end
ENDLOCAL
