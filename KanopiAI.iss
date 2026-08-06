; Inno Setup installer for the PyInstaller onedir build of KanopiAI.
; Build order:
;   pyinstaller --clean --noconfirm build\KanopiAI.spec
;   ISCC.exe KanopiAI.iss

#define MyAppName "KanopiAI"
#define MyAppVersion "1.2.4"
#define MyAppPublisher "KanopiAI"
#define MyAppExeName "KanopiAI.exe"
#define PyInstallerDist "dist\KanopiAI"

[Setup]
AppId={{B8A98E72-2C7A-4D64-9BE0-6B2C6A3AE2E4}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=installer
OutputBaseFilename=KanopiAI-Setup-{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=admin
UninstallDisplayIcon={app}\{#MyAppExeName}
SetupIconFile=logo\logo.ico

[Files]
Source: "{#PyInstallerDist}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\output"
