param(
    [string]$OutputDir = "data\training\hwr200_hf"
)

$ErrorActionPreference = "Stop"

$files = @(
    "annotations.zip",
    "hwr200_0_19.zip",
    "hwr200_20_39.zip",
    "hwr200_40_59.zip",
    "hwr200_60_79.zip",
    "hwr200_80_99.zip",
    "hwr200_100_119.zip",
    "hwr200_120_139.zip",
    "hwr200_140_159.zip",
    "hwr200_160_179.zip",
    "hwr200_180_199.zip"
)

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

foreach ($file in $files) {
    $url = "https://huggingface.co/datasets/AntiplagiatCompany/HWR200/resolve/main/$file"
    $target = Join-Path $OutputDir $file
    Write-Output "downloading: $file"
    curl.exe `
        -L `
        --fail `
        --retry 20 `
        --retry-delay 10 `
        --retry-all-errors `
        -C - `
        -o $target `
        $url
    $item = Get-Item -LiteralPath $target
    Write-Output ("downloaded: {0} ({1:N2} GB)" -f $item.FullName, ($item.Length / 1GB))
}

Write-Output "done: $OutputDir"
