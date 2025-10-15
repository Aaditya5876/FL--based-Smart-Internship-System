param(
  [string[]]$Names = @('enhanced_pfl_flow','rq2_pipeline','fl_architecture','data_flow'),
  [string]$InDir = 'docs/figures',
  [string]$OutDir = 'docs/figures'
)

$ErrorActionPreference = 'Stop'

function Require-Cmd($name) {
  $exists = (Get-Command $name -ErrorAction SilentlyContinue) -ne $null
  if (-not $exists) {
    Write-Error "Command not found: $name. Please install it (npm i -g @mermaid-js/mermaid-cli)."
  }
}

Require-Cmd 'mmdc'

foreach ($n in $Names) {
  $in = Join-Path $InDir ($n + '.mmd')
  $out = Join-Path $OutDir ($n + '.png')
  if (-not (Test-Path $in)) {
    Write-Warning "Missing input: $in"
    continue
  }
  Write-Host "[RENDER] $in -> $out"
  mmdc -i $in -o $out --backgroundColor white --scale 2
}

Write-Host "[OK] Diagrams rendered to $OutDir"

