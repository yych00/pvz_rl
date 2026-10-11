param(
    [string]$PythonExe = "",
    [ValidateSet("cpu", "cu121")][string]$TorchIndex = "cu121",
    [string]$NativeArchive = "",
    [string]$ReleaseBaseUrl = "",
    [switch]$BuildNative
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent
Push-Location $repoRoot
try {
    if (-not (Test-Path .venv\Scripts\python.exe)) {
        if ($PythonExe) { & $PythonExe -m venv .venv }
        else { & py -3.12 -m venv .venv }
        if ($LASTEXITCODE -ne 0) { throw "Install 64-bit Python 3.12, or pass -PythonExe for Python 3.10-3.12." }
    }
    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip installation failed" }
    $torchVersion = if ($TorchIndex -eq "cu121") { "2.5.1+cu121" } else { "2.5.1+cpu" }
    & $venvPython -m pip install "torch==$torchVersion" --index-url "https://download.pytorch.org/whl/$TorchIndex" --extra-index-url https://pypi.org/simple
    if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed" }
    & $venvPython -m pip install -e ".[train]"
    if ($LASTEXITCODE -ne 0) { throw "Package installation failed" }
    if (-not (Test-Path .env)) { Copy-Item .env.example .env }
    if ($BuildNative) {
        & scripts\build_native.bat x64
        if ($LASTEXITCODE -ne 0) { throw "Native build failed" }
    } elseif ($NativeArchive -or -not (Test-Path bin\pvz_gym_native.dll)) {
        if (-not $NativeArchive) {
            $version = (& $venvPython -c "from importlib.metadata import version; print(version('pvz-rl'))").Trim()
            if (-not $ReleaseBaseUrl) {
                $remote = & git config --get remote.origin.url
                if ($LASTEXITCODE -eq 0 -and $remote -match 'github\.com[:/](.+?)(?:\.git)?$') {
                    $githubRepository = $Matches[1]
                    $downloadDirectory = Join-Path $repoRoot "build\downloads"
                    & $venvPython -m scripts.download_native --repository $githubRepository --destination $downloadDirectory
                    if ($LASTEXITCODE -ne 0) { throw "Release download failed. Use -NativeArchive or -BuildNative if the Release is unavailable." }
                    $NativeArchive = Join-Path $downloadDirectory "pvz-rl-native-$version-windows-x64.zip"
                }
            }
            if (-not $NativeArchive) {
                if (-not $ReleaseBaseUrl) { throw "No published Release is configured. Pass -NativeArchive, -ReleaseBaseUrl, or -BuildNative." }
                New-Item -ItemType Directory -Force build\downloads | Out-Null
                $NativeArchive = Join-Path $repoRoot "build\downloads\pvz-rl-native-$version-windows-x64.zip"
                Invoke-WebRequest "$ReleaseBaseUrl/$(Split-Path $NativeArchive -Leaf)" -OutFile $NativeArchive
                Invoke-WebRequest "$ReleaseBaseUrl/$(Split-Path $NativeArchive -Leaf).sha256" -OutFile "$NativeArchive.sha256"
            }
        }
        if (-not (Test-Path "$NativeArchive.sha256")) { throw "Missing SHA256 sidecar: $NativeArchive.sha256" }
        $expected = ((Get-Content "$NativeArchive.sha256" -Raw).Trim() -split '\s+')[0]
        if ((Get-FileHash -LiteralPath $NativeArchive -Algorithm SHA256).Hash -ne $expected) { throw "Native archive checksum mismatch" }
        & $venvPython -m scripts.install_native $NativeArchive
        if ($LASTEXITCODE -ne 0) { throw "Native package installation failed" }
    }
    Write-Host "Installed. Supply the complete game directory (game/README.md), then run:"
    Write-Host ".venv\Scripts\python.exe -m scripts.doctor --skip-save"
    Write-Host ".venv\Scripts\python.exe -m scripts.prepare_training_save"
    Write-Host ".venv\Scripts\python.exe -m scripts.doctor"
} finally { Pop-Location }
