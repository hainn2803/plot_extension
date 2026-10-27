$ErrorActionPreference = 'Stop'
$workspace = (Get-Location).Path
$root = Join-Path $workspace 'eval/mcqa_apc_128_20261026_v1'
$python = Join-Path $workspace 'diagnostics/ap_thermal_20261021/evidence/python_candidate_20261022/payload/python.exe'
$cpu = Get-Content -LiteralPath (Join-Path $root 'engineering/cpu_tests.json') -Raw | ConvertFrom-Json
if (-not $cpu.passed -or $cpu.cuda_initialized -or $cpu.original_test_count -ne 14 -or $cpu.additional_test_count -ne 7) { throw 'All21 CPU checks required' }
$ledger = Get-Content -LiteralPath (Join-Path $root 'engineering/gpu_budget.json') -Raw | ConvertFrom-Json
if ($ledger.active -or $ledger.seconds -ne 8769.316136500012 -or $ledger.total_limit_seconds -ne 43200) { throw 'Inherited closed ledger mismatch' }
foreach ($name in @('pause.request','terminal.json','latest_process.json','gpu_completion.json','completion.json')) {
    if (Test-Path -LiteralPath (Join-Path $root $name)) { throw "Existing $name requires independent investigation" }
}
$old = Get-Content -LiteralPath (Join-Path $workspace 'eval/mcqa_at_rank_sweep_20261026_v1/latest_process.json') -Raw | ConvertFrom-Json
$actual = Get-CimInstance Win32_Process -Filter "ProcessId=$($old.pid)"
if ($null -ne $actual) {
    $created = [DateTimeOffset]($actual.CreationDate.ToUniversalTime())
    if ([Math]::Abs($created.ToUnixTimeMilliseconds()/1000 - $old.created) -lt .01) { throw 'Actual prior worker still alive' }
}
$live = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*-m experiments.mcqa_apc_128_20261026 run*' })
if ($live.Count -gt 0) { throw 'AP/C worker already exists' }
$stamp = [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss')
$stdout = Join-Path $root "apc_$stamp.stdout.log"
$stderr = Join-Path $root "apc_$stamp.stderr.log"
$arguments = @('-u','-B','-m','experiments.mcqa_apc_128_20261026','run')
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $workspace -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Start-Sleep -Seconds 2
$actual = Get-CimInstance Win32_Process -Filter "ProcessId=$($process.Id)"
if ($null -eq $actual) { throw "Worker exited immediately; inspect $stderr" }
$value = [ordered]@{
    pid = $process.Id
    created_utc = $actual.CreationDate.ToUniversalTime().ToString('o')
    command_line = $actual.CommandLine
    stdout = $stdout
    stderr = $stderr
    identity_sha256 = (Get-FileHash -LiteralPath (Join-Path $root 'identity.json') -Algorithm SHA256).Hash.ToLowerInvariant()
    launch_script_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    inherited_seconds = $ledger.seconds
    total_limit_seconds = 43200
    AT_accuracy_qualification_enabled = $false
}
$json = $value | ConvertTo-Json -Depth 6
[IO.File]::WriteAllText((Join-Path $root "launch_$stamp.json"), $json, [Text.UTF8Encoding]::new($false))
$json
