$ErrorActionPreference = 'Stop'
$workspace = (Get-Location).Path
$folder = Join-Path $workspace 'eval/mcqa_apc_128_20261026_v1/supplemental/full_mediator_donor_v1'
$python = Join-Path $workspace 'diagnostics/ap_thermal_20261021/evidence/python_candidate_20261022/payload/python.exe'
$plan = Get-Content -LiteralPath (Join-Path $folder 'plan.json') -Raw | ConvertFrom-Json
$tests = Get-Content -LiteralPath (Join-Path $folder 'CPU_tests.json') -Raw | ConvertFrom-Json
if (-not $tests.passed -or $tests.cuda_initialized -or $tests.test_count -ne 13) { throw 'All13 CPU regressions required' }
if ($plan.new_training_updates -ne 0 -or $plan.new_handles -ne 0 -or $plan.new_readouts -ne 0 -or $plan.new_contexts -ne 0) { throw 'Evaluation-only scope required' }
if ($plan.shared_total_limit_seconds -ne 46800 -or $plan.allocation_seconds_added -ne 3600 -or $plan.reserve_seconds -ne 300) { throw 'Exact authorized13h cumulative budget required' }
if ((Get-FileHash -LiteralPath $python -Algorithm SHA256).Hash.ToLowerInvariant() -ne '0c81a500d76bd4a9af739d528acd57bde001dc9e666effb3494bbaec30a71004') { throw 'Signed interpreter changed' }
foreach ($set in @($plan.source_files,$plan.frozen_inputs)) {
    foreach ($property in $set.PSObject.Properties) {
        if ((Get-FileHash -LiteralPath $property.Name -Algorithm SHA256).Hash.ToLowerInvariant() -ne $property.Value) { throw "Sealed input changed: $($property.Name)" }
    }
}
$existing = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match '-m diagnostics\.apc_128_20261026\.full_transfer (wait|gpu)' })
if ($existing.Count -gt 0) { throw 'An addendum controller or GPU worker already exists' }
foreach ($name in @('queue.json','terminal.json','authorization.json','gpu_completion.json','completion.json')) {
    if (Test-Path -LiteralPath (Join-Path $folder $name)) { throw "Existing $name requires inspection; no blind restart" }
}
$source = $plan.source_worker
$actualSource = Get-CimInstance Win32_Process -Filter "ProcessId=$($source.pid)"
if ($null -ne $actualSource) {
    $created = [DateTimeOffset]($actualSource.CreationDate.ToUniversalTime())
    if ([Math]::Abs($created.ToUnixTimeMilliseconds()/1000 - $source.created) -lt .01 -and $actualSource.CommandLine -notmatch '-m diagnostics\.apc_128_20261026\.resume_last_fit run') { throw 'Original worker command changed' }
}
$stdout = Join-Path $folder 'controller.stdout.log'
$stderr = Join-Path $folder 'controller.stderr.log'
$process = Start-Process -FilePath $python -ArgumentList @('-u','-B','-m','diagnostics.apc_128_20261026.full_transfer','wait') -WorkingDirectory $workspace -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Start-Sleep -Seconds 2
$actual = Get-CimInstance Win32_Process -Filter "ProcessId=$($process.Id)"
if ($null -eq $actual) { throw "Controller exited immediately; inspect $stderr" }
$receipt = [ordered]@{pid=$process.Id;created_utc=$actual.CreationDate.ToUniversalTime().ToString('o');command_line=$actual.CommandLine;plan_sha256=(Get-FileHash -LiteralPath (Join-Path $folder 'plan.json') -Algorithm SHA256).Hash.ToLowerInvariant();launch_script_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant();source_worker=$source;waiting_only_no_GPU_until_original_exit=$true;stdout=$stdout;stderr=$stderr;additional_authorized_seconds=3600;cumulative_total_limit_seconds=46800}
$json = $receipt | ConvertTo-Json -Depth 7
[IO.File]::WriteAllText((Join-Path $folder 'controller_launch.json'),$json,[Text.UTF8Encoding]::new($false))
$json
