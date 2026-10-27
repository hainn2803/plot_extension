param([Parameter(Mandatory=$true)][string]$ProofPath)
$ErrorActionPreference='Stop'
$workspace=(Get-Location).Path
$python=Join-Path $workspace 'diagnostics/ap_thermal_20261021/evidence/python_candidate_20261022/payload/python.exe'
$root=Join-Path $workspace 'eval/mcqa_apc_128_20261026_v1'
$extra=Join-Path $root 'supplemental/full_mediator_donor_v1'
$proofPath=(Resolve-Path -LiteralPath $ProofPath).Path
$proof=Get-Content -LiteralPath $proofPath -Raw | ConvertFrom-Json
$authorizationFolder=Split-Path -Parent $proofPath
if (-not $proof.passed -or $proof.cuda_initialized -or $proof.new_training_updates -ne 0 -or $proof.verified_fits -ne 15 -or $proof.verified_orders -ne 15000 -or $proof.independently_scored_saved_outputs -ne 175) { throw 'Full CPU evaluation-resume proof required' }
if ($proof.CPU_tests.evaluation_resume_tests -ne 6 -or -not $proof.CPU_tests.passed -or $proof.CPU_tests.full_transfer_tests.test_count -ne 13 -or -not $proof.CPU_tests.full_transfer_tests.passed) { throw 'All19 CPU regressions required' }
if (-not $proof.complete_remaining_gate.allowed) { throw 'Full original+supplement+validation+25percent gate required' }
foreach($property in $proof.source_files.PSObject.Properties){
    if((Get-FileHash -LiteralPath $property.Name -Algorithm SHA256).Hash.ToLowerInvariant() -ne $property.Value){throw "Resume source changed: $($property.Name)"}
}
if((Get-FileHash -LiteralPath $python -Algorithm SHA256).Hash.ToLowerInvariant() -ne '0c81a500d76bd4a9af739d528acd57bde001dc9e666effb3494bbaec30a71004'){throw 'Signed interpreter changed'}
$ledger=Get-Content -LiteralPath (Join-Path $root 'engineering/gpu_budget.json') -Raw | ConvertFrom-Json
if($ledger.active -or $ledger.seconds -ne 31952.714111800044 -or $ledger.profile_seconds -ne 853.1099996000121 -or $ledger.total_limit_seconds -ne 43200){throw 'Exact cumulative closed source ledger required'}
foreach($old in @($proof.prior_process,$proof.prior_queue)){
    $actual=Get-CimInstance Win32_Process -Filter "ProcessId=$($old.pid)"
    if($null -ne $actual){
        $created=[DateTimeOffset]($actual.CreationDate.ToUniversalTime())
        if([Math]::Abs($created.ToUnixTimeMilliseconds()/1000-$old.created) -lt .01){throw 'Actual stopped worker/controller has not exited'}
    }
}
$live=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -match '-m diagnostics\.apc_128_20261026\.(resume_evaluation (run|watch)|full_transfer (wait|gpu))'})
if($live.Count -gt 0){throw 'An evaluation worker or controller is already alive'}
$stamp=[DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss')
$stdout=Join-Path $root "evaluation_resume_$stamp.stdout.log"
$stderr=Join-Path $root "evaluation_resume_$stamp.stderr.log"
$arguments=@('-u','-B','-m','diagnostics.apc_128_20261026.resume_evaluation','run','--authorization',"`"$proofPath`"")
$worker=Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $workspace -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$resumed=Join-Path $authorizationFolder 'resumed_process.json'
$ready=$false
for($i=0;$i -lt 120;$i++){
    Start-Sleep -Seconds 2
    $actual=Get-CimInstance Win32_Process -Filter "ProcessId=$($worker.Id)"
    if($null -eq $actual){throw "Evaluation worker exited before stamping; inspect $stderr"}
    if(Test-Path -LiteralPath $resumed){$ready=$true;break}
}
if(-not $ready){throw 'Worker CPU preflight still pending; inspect without restarting the live worker'}
$actual=Get-CimInstance Win32_Process -Filter "ProcessId=$($worker.Id)"
$process=Get-Content -LiteralPath $resumed -Raw | ConvertFrom-Json
$created=[DateTimeOffset]($actual.CreationDate.ToUniversalTime())
if($process.pid -ne $worker.Id -or [Math]::Abs($created.ToUnixTimeMilliseconds()/1000-$process.created) -ge .01 -or $actual.CommandLine -notmatch '-m diagnostics\.apc_128_20261026\.resume_evaluation run --authorization'){throw 'New worker command/creation identity mismatch'}
$watchOut=Join-Path $extra 'controller.stdout.log'
$watchErr=Join-Path $extra 'controller.stderr.log'
$watcher=Start-Process -FilePath $python -ArgumentList @('-u','-B','-m','diagnostics.apc_128_20261026.resume_evaluation','watch','--authorization',"`"$proofPath`"") -WorkingDirectory $workspace -WindowStyle Hidden -RedirectStandardOutput $watchOut -RedirectStandardError $watchErr -PassThru
Start-Sleep -Seconds 2
$watchActual=Get-CimInstance Win32_Process -Filter "ProcessId=$($watcher.Id)"
if($null -eq $watchActual){throw "Resume controller exited immediately; inspect $watchErr"}
$value=[ordered]@{worker=$process;worker_created_utc=$actual.CreationDate.ToUniversalTime().ToString('o');worker_command_line=$actual.CommandLine;stdout=$stdout;stderr=$stderr;authorization=$proofPath;authorization_sha256=(Get-FileHash -LiteralPath $proofPath -Algorithm SHA256).Hash.ToLowerInvariant();watcher_pid=$watcher.Id;watcher_created_utc=$watchActual.CreationDate.ToUniversalTime().ToString('o');watcher_command_line=$watchActual.CommandLine;inherited_closed_seconds=31952.714111800044;source_cap_seconds=43200;supplement_cumulative_cap_seconds=46800;new_training_updates=0;original_saved_outputs_preserved=175}
$json=$value|ConvertTo-Json -Depth 7
[IO.File]::WriteAllText((Join-Path $root "launch_evaluation_resume_$stamp.json"),$json,[Text.UTF8Encoding]::new($false))
[IO.File]::WriteAllText((Join-Path $extra 'controller_launch.json'),$json,[Text.UTF8Encoding]::new($false))
$json
