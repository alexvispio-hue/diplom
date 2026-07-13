#define MyAppName "Handwriting OCR"
#define MyAppVersion "0.4.0-dev"
#define MyAppPublisher "Handwriting OCR"
#define ProjectRoot ".."

[Setup]
AppId={{4E95A72A-E8C7-47D4-8FD8-8AD6E5D8474B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\HandwritingOCR
DefaultGroupName={#MyAppName}
OutputDir=..\dist
OutputBaseFilename=handwriting-ocr-{#MyAppVersion}-setup
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
UninstallDisplayIcon={app}\START_APP.bat
ChangesEnvironment=no

[Types]
Name: "lite"; Description: "Lite (TrOCR)"
Name: "full"; Description: "Full (TrOCR + Qwen runtime)"
Name: "developer"; Description: "Developer (training and tests)"

[Components]
Name: "core"; Description: "Application"; Types: lite full developer; Flags: fixed
Name: "profile_lite"; Description: "Lite runtime"; Types: lite; Flags: exclusive
Name: "profile_full"; Description: "Full runtime"; Types: full; Flags: exclusive
Name: "profile_developer"; Description: "Developer runtime"; Types: developer; Flags: exclusive

[Files]
Source: "{#ProjectRoot}\app\*"; DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#ProjectRoot}\ui\*"; DestDir: "{app}\ui"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#ProjectRoot}\.streamlit\*"; DestDir: "{app}\.streamlit"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#ProjectRoot}\scripts\install_windows.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#ProjectRoot}\scripts\system_diagnostics.py"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#ProjectRoot}\scripts\manage_models.py"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#ProjectRoot}\scripts\audit_acceptance.py"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#ProjectRoot}\scripts\preflight_qwen_training.py"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#ProjectRoot}\scripts\launch_qwen_training.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#ProjectRoot}\scripts\run_hwr200_full_training.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#ProjectRoot}\scripts\run_baseline_suite.ps1"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#ProjectRoot}\scripts\run_ocr_benchmark.py"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#ProjectRoot}\scripts\evaluate_ocr_predictions.py"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#ProjectRoot}\scripts\build_gold_benchmark.py"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#ProjectRoot}\scripts\build_dataset_manifest.py"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#ProjectRoot}\scripts\train_qwen_vl_lora.py"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#ProjectRoot}\scripts\validate_hwr200_full_dataset.py"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#ProjectRoot}\scripts\prepare_hwr200_full_qwen.py"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "{#ProjectRoot}\requirements-base.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\requirements-trocr.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\requirements-qwen.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\requirements-ml.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\requirements-dev.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\START_APP.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ProjectRoot}\CHANGELOG.md"; DestDir: "{app}"; Flags: ignoreversion

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked

[Icons]
Name: "{group}\Handwriting OCR"; Filename: "{app}\START_APP.bat"; WorkingDir: "{app}"
Name: "{autodesktop}\Handwriting OCR"; Filename: "{app}\START_APP.bat"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\install_windows.ps1"" -Profile Lite -NoShortcut"; StatusMsg: "Installing Lite runtime..."; Components: profile_lite; Flags: runhidden waituntilterminated
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\install_windows.ps1"" -Profile Full -NoShortcut"; StatusMsg: "Installing Full runtime..."; Components: profile_full; Flags: runhidden waituntilterminated
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\scripts\install_windows.ps1"" -Profile Developer -NoShortcut"; StatusMsg: "Installing Developer runtime..."; Components: profile_developer; Flags: runhidden waituntilterminated
Filename: "{app}\START_APP.bat"; Description: "Start Handwriting OCR"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv-app"

[Code]
function InitializeUninstall(): Boolean;
begin
  Result := True;
  MsgBox('User documents and downloaded models in LocalAppData\HandwritingOCR are kept.', mbInformation, MB_OK);
end;
