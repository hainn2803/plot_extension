param([Parameter(Mandatory=$true)][string]$Receipt)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath '.').Path
$python = (Resolve-Path -LiteralPath 'diagnostics/ap_thermal_20261021/evidence/python_candidate_20261022/payload/python.exe').Path
if ((Get-FileHash -LiteralPath $python -Algorithm SHA256).Hash.ToLowerInvariant() -ne '0c81a500d76bd4a9af739d528acd57bde001dc9e666effb3494bbaec30a71004') {
    throw 'Signed interpreter changed'
}
$receiptPath = (Resolve-Path -LiteralPath $Receipt).Path
$proof = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
if (-not $proof.passed -or -not $proof.actual_source_and_old_controller_exited) {
    throw 'Successful CPU-only report repair proof required'
}
$output = Join-Path $workspace 'eval/mcqa_apc_128_20261026_v1/supplemental/full_mediator_donor_v1'
$stdout = Join-Path $output 'controller.stdout.log'
$stderr = Join-Path $output 'controller.stderr.log'
if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) {
    throw 'Controller logs must first be archived, not overwritten'
}
$arguments = '-u -B -m diagnostics.apc_128_20261026.report_seed_repair watch --receipt "' + $receiptPath + '"'
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $workspace -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$launch = [ordered]@{
    pid = $process.Id
    created_utc = $process.StartTime.ToUniversalTime().ToString('o')
    executable = $python
    arguments = $arguments
    receipt = $receiptPath
    receipt_sha256 = (Get-FileHash -LiteralPath $receiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
    evaluation_only = $true
    no_training = $true
    cumulative_total_limit_seconds = 50400
    time_utc = [DateTime]::UtcNow.ToString('o')
}
$launch | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $output 'controller_launch.json') -Encoding utf8
$launch | ConvertTo-Json -Depth 4
