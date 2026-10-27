$ErrorActionPreference = 'Stop'
$workspace = (Get-Location).Path
$root = Join-Path $workspace 'eval/mcqa_apc_128_20261026_v1'
$python = Join-Path $workspace 'diagnostics/ap_thermal_20261021/evidence/python_candidate_20261022/payload/python.exe'
$proofPath = Join-Path $root 'engineering/authorized_resumes/1793037991795489900/authorization.json'
$proof = Get-Content -LiteralPath $proofPath -Raw | ConvertFrom-Json
if (-not $proof.passed -or $proof.cuda_initialized -or $proof.verified_orders -ne 8100) { throw 'Complete CPU resume proof required' }
$checks = @{
    'diagnostics/apc_128_20261026/resume.py' = $proof.helper_sha256
    'diagnostics/apc_128_20261026/resume_tests.py' = $proof.test_helper_sha256
    'eval/mcqa_apc_128_20261026_v1/identity.json' = $proof.identity_sha256
    'eval/mcqa_apc_128_20261026_v1/profile.json' = $proof.profile_sha256
    'eval/mcqa_apc_128_20261026_v1/execution.json' = $proof.execution_sha256
    'diagnostics/ap_thermal_20261021/evidence/python_candidate_20261022/payload/python.exe' = '0c81a500d76bd4a9af739d528acd57bde001dc9e666effb3494bbaec30a71004'
}
foreach ($path in $checks.Keys) {
    if ((Get-FileHash -LiteralPath (Join-Path $workspace $path) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $checks[$path]) { throw "Identity changed: $path" }
}
$ledger = Get-Content -LiteralPath (Join-Path $root 'engineering/gpu_budget.json') -Raw | ConvertFrom-Json
if ($ledger.active -or $ledger.seconds -ne 12988.62537730002 -or $ledger.profile_seconds -ne 853.1099996000121 -or $ledger.total_limit_seconds -ne 43200) { throw 'Exact prior closed ledger required' }
$pause = Get-Content -LiteralPath (Join-Path $root 'pause.request') -Raw | ConvertFrom-Json
if ($pause.reason -ne $proof.pause.reason -or $pause.time -ne $proof.pause.time) { throw 'Exact authorized user pause required' }
$old = $proof.prior_process
$actual = Get-CimInstance Win32_Process -Filter "ProcessId=$($old.pid)"
if ($null -ne $actual) {
    $created = [DateTimeOffset]($actual.CreationDate.ToUniversalTime())
    if ([Math]::Abs($created.ToUnixTimeMilliseconds()/1000 - $old.created) -lt .01) { throw 'Actual prior worker still alive' }
}
$live = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match '-m (experiments.mcqa_apc_128_20261026|diagnostics.apc_128_20261026.resume) run' })
if ($live.Count -gt 0) { throw 'AP/C worker already exists' }
foreach ($name in @('gpu_completion.json','completion.json','final_access.json')) {
    if (Test-Path -LiteralPath (Join-Path $root $name)) { throw "Unexpected $name at user pause" }
}
$stamp = [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss')
$stdout = Join-Path $root "resume_$stamp.stdout.log"
$stderr = Join-Path $root "resume_$stamp.stderr.log"
$arguments = @('-u','-B','-m','diagnostics.apc_128_20261026.resume','run')
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
    helper_sha256 = $proof.helper_sha256
    test_helper_sha256 = $proof.test_helper_sha256
    identity_sha256 = $proof.identity_sha256
    CPU_proof = $proofPath
    launch_script_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    inherited_seconds = $ledger.seconds
    total_limit_seconds = 43200
    resume_fit = 'AP_gradual_S2'
    resume_step = 100
    AT_accuracy_qualification_enabled = $false
}
$json = $value | ConvertTo-Json -Depth 6
[IO.File]::WriteAllText((Join-Path $root "launch_resume_$stamp.json"), $json, [Text.UTF8Encoding]::new($false))
$json
