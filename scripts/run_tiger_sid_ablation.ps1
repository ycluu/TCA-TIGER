param(
    [ValidateSet("rqvae", "rqkmeans", "rqopq", "all")]
    [string]$Method = "all",
    [string]$Split = "beauty",
    [ValidateSet("all", "sid", "tiger")]
    [string]$Stage = "all",
    [switch]$TrainRQVae
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Run-Python([string[]]$Arguments) {
    Write-Host "> python $($Arguments -join ' ')"
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
}

function Run-Method([string]$Name) {
    if (($Stage -eq "all" -or $Stage -eq "sid") -and $Name -eq "rqvae") {
        if ($TrainRQVae) {
            Run-Python @("genrec/trainers/rqvae_trainer.py", "config/tiger/amazon/rqvae.gin", "--split", $Split)
        }
        Run-Python @("genrec/trainers/sid_export_trainer.py", "config/tiger/amazon/sid_rqvae_export.gin", "--split", $Split)
    }
    if (($Stage -eq "all" -or $Stage -eq "sid") -and $Name -eq "rqkmeans") {
        Run-Python @("genrec/trainers/rqkmeans_trainer.py", "config/tiger/amazon/sid_rqkmeans.gin", "--split", $Split)
    }
    if (($Stage -eq "all" -or $Stage -eq "sid") -and $Name -eq "rqopq") {
        Run-Python @("genrec/trainers/rqkmeans_trainer.py", "config/tiger/amazon/sid_rqopq.gin", "--split", $Split)
    }
    if ($Stage -eq "all" -or $Stage -eq "tiger") {
        $tigerConfig = if ($Name -eq "rqvae") { "tiger_rqvae_artifact.gin" } else { "tiger_$Name.gin" }
        Run-Python @("genrec/trainers/tiger_trainer.py", "config/tiger/amazon/$tigerConfig", "--split", $Split)
    }
}

if ($Method -eq "all") {
    @("rqvae", "rqkmeans", "rqopq") | ForEach-Object { Run-Method $_ }
    Run-Python @("scripts/summarize_tiger_sid_ablation.py", "--split", $Split)
} else {
    Run-Method $Method
}
