Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $root "outputs\build"
$vsDevCmd = "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat"

New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

if (-not (Test-Path $vsDevCmd)) {
    throw "VsDevCmd.bat not found at $vsDevCmd"
}

$sources = @(
    (Join-Path $root "cpp\direction_encoding.cpp"),
    (Join-Path $root "cpp\StateValueRegistry.cpp"),
    (Join-Path $root "cpp\RewardConfigStore.cpp"),
    (Join-Path $root "cpp\utils\direction_pattern_lookup.cpp"),
    (Join-Path $root "cpp\precompute\DirectionDeltaTable.cpp")
)

function Invoke-ClBuild {
    param(
        [Parameter(Mandatory = $true)]
        [string]$OutputExe,
        [Parameter(Mandatory = $true)]
        [string[]]$ExtraSources
    )

    $allSources = @($sources + $ExtraSources)
    $quotedSources = ($allSources | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $includeDir = Join-Path $root 'cpp'
    $outputPath = Join-Path $buildDir $OutputExe
    $command = "cl /nologo /std:c++17 /EHsc /utf-8 /I `"$includeDir`" $quotedSources /link /OUT:`"$outputPath`""

    Push-Location $buildDir
    try {
        cmd /c ('call "' + $vsDevCmd + '" -arch=x64 && ' + $command)
    } finally {
        Pop-Location
    }
}

Invoke-ClBuild -OutputExe "direction_pattern_test.exe" -ExtraSources @(
    (Join-Path $root "cpp\test\direction_pattern_test_main.cpp")
)

Invoke-ClBuild -OutputExe "build_direction_delta_table.exe" -ExtraSources @(
    (Join-Path $root "cpp\precompute\build_direction_delta_table.cpp")
)
