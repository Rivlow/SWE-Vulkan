; Inno Setup script of SWE Vulkan. Built by packaging\make_installer.bat from the staging folder build\release\dist
; (CMake target `dist`). Per-user install (no administrator rights), the application writes its settings,
; captures and imported cases next to its executable.
#define AppName "SWE Vulkan"
#define AppVersion "0.2.0"
#define AppExe "SWE-Vulkan.exe"
#ifndef DistDir
  #define DistDir "..\build\release\dist"
#endif
#ifndef OutDir
  #define OutDir "..\build\release"
#endif
#ifndef Suffix
  #define Suffix ""
#endif

[Setup]
AppId={{7E1A3C52-5B1F-4C0E-9C3D-A1B2C3D4E5F6}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Luca Santoro
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#OutDir}
OutputBaseFilename=SWE-Vulkan-{#AppVersion}{#Suffix}-setup
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
ShowLanguageDialog=no

[Languages]
Name: "fr"; MessagesFile: "compiler:Languages\French.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"
Name: "{group}\Désinstaller {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; files written by the application next to its executable
Type: files; Name: "{app}\swe_settings.json"
Type: filesandordirs; Name: "{app}\captures"

[Messages]
fr.WelcomeLabel2=Ce programme va installer {#AppName} sur votre ordinateur.%n%nSWE Vulkan simule des crues en 3D sur un relief réel (import de la carte) ou analytique, avec nappe souterraine, pluie et infiltration.%n%nPrérequis : Windows 10/11 64 bits et une carte graphique avec des pilotes Vulkan récents (NVIDIA, AMD ou Intel).
