param(
    [string]$OutputPath = "dist\CascadeRank-Audit-Demo.mp4"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$assetRoot = Join-Path $repoRoot "dist\video_assets"
$outputFile = Join-Path $repoRoot $OutputPath
$chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
$ffmpeg = (Get-Command ffmpeg -ErrorAction Stop).Source
$ffprobe = (Get-Command ffprobe -ErrorAction Stop).Source

if (-not (Test-Path -LiteralPath $chrome)) {
    throw "Chrome was not found at $chrome"
}
New-Item -ItemType Directory -Path $assetRoot -Force | Out-Null

$leakyManifest = Join-Path $repoRoot "demo_output\leaky\audit_manifest.json"
$topologyManifest = Join-Path $repoRoot "demo_output\topology_only\audit_manifest.json"
if (-not (Test-Path $leakyManifest) -or -not (Test-Path $topologyManifest)) {
    & (Join-Path $repoRoot "scripts\run_demo.ps1") -RandomTrials 100
    if ($LASTEXITCODE -ne 0) { throw "The reproducible demo run failed" }
}

$leaky = Get-Content $leakyManifest -Raw | ConvertFrom-Json
$topology = Get-Content $topologyManifest -Raw | ConvertFrom-Json
$leakyProxy = "{0:N4}" -f $leaky.evidence.metrics.proxy_gat.attack_auc
$leakyBaseline = "{0:N4}" -f $leaky.evidence.metrics.pagerank.attack_auc
$topologyProxy = "{0:N4}" -f $topology.evidence.metrics.proxy_gat.attack_auc
$topologyBaseline = "{0:N4}" -f $topology.evidence.metrics.pagerank.attack_auc

function New-DemoSlide {
    param(
        [string]$Name,
        [string]$Kicker,
        [string]$Title,
        [string]$Body,
        [string]$Accent = "#5eead4"
    )
    $path = Join-Path $assetRoot "$Name.html"
    $markup = @"
<!doctype html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box}body{margin:0;background:#08111f;color:#f8fafc;
font-family:Arial,Segoe UI,sans-serif;width:1280px;height:720px;overflow:hidden}
.frame{padding:72px 86px;height:720px;border-top:12px solid $Accent}
.kicker{font-size:22px;letter-spacing:3px;text-transform:uppercase;color:$Accent;
font-weight:700;margin-bottom:28px}.title{font-size:58px;line-height:1.08;
font-weight:800;max-width:1080px;margin-bottom:32px}.body{font-size:29px;
line-height:1.45;color:#cbd5e1;max-width:1080px}.body strong{color:#fff}
.badge{display:inline-block;margin-top:34px;border:1px solid $Accent;color:$Accent;
padding:10px 16px;border-radius:999px;font-size:20px}.mono{font-family:Consolas,monospace;
font-size:24px;background:#111c2e;padding:22px;border-radius:12px;color:#dbeafe}
</style></head><body><div class="frame"><div class="kicker">$Kicker</div>
<div class="title">$Title</div><div class="body">$Body</div></div></body></html>
"@
    [System.IO.File]::WriteAllText($path, $markup, [System.Text.Encoding]::UTF8)
    return $path
}

$slide1 = New-DemoSlide "01-title" "OpenAI Build Week / Education" `
    "CascadeRank Audit" `
    "An evidence-first <strong>Codex plugin</strong> for testing critical-node ranking claims.<br><span class='badge'>Local / Reproducible / No API key required</span>"
$slide2 = New-DemoSlide "02-codex" "Built with Codex and GPT-5.6" `
    "From an unsupported model claim to an auditable product" `
    "Codex implemented the leakage checks, repeated attack benchmark, fixed claim ledger, independent verifier, tests, plugin packaging, and release workflow. The decisive result was negative, and the product preserves it."
$slide6 = New-DemoSlide "06-verify" "Independent integrity gate" `
    "The report cannot silently rewrite the evidence" `
    "<div class='mono'>python -m cascaderank.verify --manifest audit_manifest.json --edge-csv edges.csv<br><br>valid: true<br>artifact hashes: verified<br>input hash: verified<br>verdict rules: verified</div>" "#fbbf24"
$slide7 = New-DemoSlide "07-boundary" "Evidence before narrative" `
    "NOT_SUPPORTED is a valid product outcome" `
    "CascadeRank Audit evaluates one-graph ranking evidence. It does not claim causal node importance or cross-graph generalization. The next experiment must earn those conclusions." "#fb7185"

function Save-Screenshot {
    param([string]$Source, [string]$Destination)
    $profile = Join-Path $assetRoot "chrome-profile"
    $arguments = @(
        "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--allow-file-access-from-files", "--window-size=1280,720",
        "--virtual-time-budget=4000", "--user-data-dir=$profile",
        "--screenshot=$Destination", $Source
    )
    & $chrome $arguments | Out-Null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Destination)) {
        throw "Chrome failed to render $Source"
    }
}

Save-Screenshot ([Uri]$slide1).AbsoluteUri (Join-Path $assetRoot "01-title.png")
Save-Screenshot ([Uri]$slide2).AbsoluteUri (Join-Path $assetRoot "02-codex.png")
Save-Screenshot "https://github.com/15803723000/CascadeRank-Audit" `
    (Join-Path $assetRoot "03-github.png")
Save-Screenshot ([Uri](Join-Path $repoRoot "demo_output\leaky\audit_report.html")).AbsoluteUri `
    (Join-Path $assetRoot "04-leaky.png")
Save-Screenshot ([Uri](Join-Path $repoRoot "demo_output\topology_only\audit_report.html")).AbsoluteUri `
    (Join-Path $assetRoot "05-topology.png")
Save-Screenshot ([Uri]$slide6).AbsoluteUri (Join-Path $assetRoot "06-verify.png")
Save-Screenshot ([Uri]$slide7).AbsoluteUri (Join-Path $assetRoot "07-boundary.png")

$narration = @"
Hi. This is CascadeRank Audit, an evidence-first Codex plugin for graph critical-node ranking claims.

The original project made a common methodological mistake. A graph attention network was trained on centrality signals that also constructed its target, and was then compared with those same baselines. A convincing curve would not demonstrate independent discovery.

I used Codex with GPT-5.6 to turn that failure into the product. Codex implemented provenance checks, repeated node-removal experiments, a fixed claim ledger, an independent verifier, automated tests, plugin packaging, and the reproducible release workflow shown in this repository.

The Codex skill runs two cases on the same twenty-node bridge graph. First is the intentionally leaky design. It reports direct label-feature overlap. The proxy GAT attack A U C is $leakyProxy, while PageRank reaches $leakyBaseline. Lower is better. The advantage claim is therefore not supported, and the proxy closely reconstructs its teacher.

The second case removes direct leakage. It uses constant node inputs and supervises the model with observed single-node loss in the largest connected component. The leakage flag disappears, but the model still does not win. Its attack A U C is $topologyProxy versus $topologyBaseline for PageRank. CascadeRank does not rewrite this negative result.

Every run includes six non-learned baselines, one hundred random rankings, a fifth-to-ninety-fifth percentile uncertainty envelope, and predeclared five and ten percent checkpoints.

The independent verifier recomputes the verdict rules and checks S H A two fifty six hashes for the input graph and every report artifact. If one character changes, verification fails.

This is the core educational value: Codex does not merely generate an explanation. It executes a falsifiable protocol and preserves the boundary between evidence and narrative. One-graph results do not establish causal importance or cross-graph generalization. Those conclusions must be earned by new experiments.
"@

Add-Type -AssemblyName System.Speech
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice.SelectVoice("Microsoft Zira Desktop")
$voice.Rate = 1
$wavePath = Join-Path $assetRoot "narration.wav"
$voice.SetOutputToWaveFile($wavePath)
$voice.Speak($narration)
$voice.SetOutputToDefaultAudioDevice()
$voice.Dispose()

$slides = @(
    @{File="01-title.png"; Duration=16},
    @{File="02-codex.png"; Duration=23},
    @{File="03-github.png"; Duration=20},
    @{File="04-leaky.png"; Duration=29},
    @{File="05-topology.png"; Duration=29},
    @{File="06-verify.png"; Duration=25},
    @{File="07-boundary.png"; Duration=24}
)
$concatLines = New-Object System.Collections.Generic.List[string]
foreach ($slide in $slides) {
    $imagePath = (Join-Path $assetRoot $slide.File).Replace("\", "/")
    $concatLines.Add("file '$imagePath'")
    $concatLines.Add("duration $($slide.Duration)")
}
$lastImage = (Join-Path $assetRoot $slides[-1].File).Replace("\", "/")
$concatLines.Add("file '$lastImage'")
$concatPath = Join-Path $assetRoot "slides.txt"
[System.IO.File]::WriteAllLines($concatPath, $concatLines)

& $ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i $concatPath `
    -i $wavePath -vf "fps=30,format=yuv420p" -c:v libx264 -preset medium `
    -c:a aac -b:a 160k -shortest -movflags +faststart $outputFile
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $outputFile)) {
    throw "FFmpeg failed to build the demo video"
}

$duration = [double](& $ffprobe -v error -show_entries format=duration `
    -of default=noprint_wrappers=1:nokey=1 $outputFile)
if ($duration -ge 180) { throw "Demo is too long: $duration seconds" }
$hash = (Get-FileHash $outputFile -Algorithm SHA256).Hash.ToLower()
[PSCustomObject]@{
    Video = $outputFile
    DurationSeconds = [math]::Round($duration, 2)
    Bytes = (Get-Item $outputFile).Length
    SHA256 = $hash
} | ConvertTo-Json
