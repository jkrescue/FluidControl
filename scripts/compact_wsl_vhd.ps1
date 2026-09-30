param(
    [Parameter(Mandatory = $true)]
    [string]$VhdPath
)

$ErrorActionPreference = "Stop"
$scriptPath = Join-Path $env:TEMP "compact-wsl-vhd-diskpart.txt"
@(
    "select vdisk file=`"$VhdPath`""
    "compact vdisk"
) | Set-Content -LiteralPath $scriptPath -Encoding ASCII

$wslService = Get-Service -Name "WSLService" -ErrorAction SilentlyContinue
$restartService = $null -ne $wslService -and $wslService.Status -eq "Running"
try {
    if ($restartService) {
        Stop-Service -Name "WSLService" -Force
        Start-Sleep -Seconds 3
    }
    & diskpart.exe /s $scriptPath
    if ($LASTEXITCODE -ne 0) {
        throw "diskpart failed with exit code $LASTEXITCODE"
    }
}
finally {
    if ($restartService) {
        Start-Service -Name "WSLService"
    }
    Remove-Item -LiteralPath $scriptPath -Force -ErrorAction SilentlyContinue
}
