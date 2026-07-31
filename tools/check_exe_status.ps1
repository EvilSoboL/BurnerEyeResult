[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

try {
    $projectRoot = [System.IO.Path]::GetFullPath(
        (Join-Path -Path $PSScriptRoot -ChildPath "..")
    )
    $exePath = Join-Path -Path $projectRoot -ChildPath "dist\BurnerEyeReport.exe"

    $buildInputs = @(
        (Join-Path $projectRoot "main.py"),
        (Join-Path $projectRoot "requirements.txt"),
        (Join-Path $projectRoot "requirements-build.txt"),
        (Join-Path $projectRoot "BurnerEyeReport.spec"),
        (Join-Path $projectRoot "build_exe.bat")
    )
    $buildInputs += Get-ChildItem `
        -LiteralPath (Join-Path $projectRoot "burner_eye_report") `
        -Filter "*.py" `
        -File `
        -Recurse |
        Select-Object -ExpandProperty FullName

    $missingInputs = @($buildInputs | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
    if ($missingInputs.Count -gt 0) {
        Write-Host "[ERROR] Build input is missing:"
        $missingInputs | ForEach-Object { Write-Host "  $_" }
        exit 2
    }

    if (-not (Test-Path -LiteralPath $exePath -PathType Leaf)) {
        Write-Host "[STATUS] EXE is missing."
        exit 10
    }

    $exeTime = (Get-Item -LiteralPath $exePath).LastWriteTimeUtc
    $newerInputs = @(
        $buildInputs |
            Get-Item |
            Where-Object { $_.LastWriteTimeUtc -gt $exeTime } |
            Sort-Object LastWriteTimeUtc
    )

    if ($newerInputs.Count -gt 0) {
        Write-Host "[STATUS] EXE is outdated. Newer build inputs:"
        $newerInputs | ForEach-Object {
            $relativePath = $_.FullName.Substring($projectRoot.Length).TrimStart("\", "/")
            Write-Host "  $relativePath"
        }
        exit 11
    }

    Write-Host "[STATUS] EXE is up to date."
    exit 0
}
catch {
    Write-Host "[ERROR] Could not determine EXE status: $($_.Exception.Message)"
    exit 2
}
