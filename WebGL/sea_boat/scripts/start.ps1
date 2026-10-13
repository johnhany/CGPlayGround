# 启动 sea_boat 本地静态服务器（检测端口占用并清理旧进程）
$Port = 38517
$Dir = Split-Path -Parent $PSScriptRoot

Write-Host "[1/3] 检查端口 $Port 占用"
$conns = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
if ($conns) {
  $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
  foreach ($p in $pids) {
    Write-Host "      端口被占用，结束旧进程 PID $p"
    Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
  }
  Start-Sleep -Seconds 1
} else {
  Write-Host "      端口空闲"
}

Write-Host "[2/3] 以 $Dir 为根启动静态服务器（禁缓存头，避免开发期模块缓存）"
Write-Host "[3/3] 访问 http://127.0.0.1:$Port"

Set-Location $Dir
python "$Dir\scripts\serve.py" $Port
