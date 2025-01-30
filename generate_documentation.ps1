$packageName = "swarmsim"
$docsDir = "Docs"

# Create docs directory if not exists
if (!(Test-Path $docsDir)) { New-Item -ItemType Directory -Path $docsDir }

Write-Host "Generating documentation for package: $packageName"

python -m pydoc -w "$packageName"
Move-Item -Path "$packageName.html" -Destination $docsDir -Force


$package_folders = Get-ChildItem -Path ".\$packageName" -Directory -Recurse
$dander_excluded = $package_folders | Where-Object { $_.Name -notlike '_*' }
foreach ($folder in $dander_excluded) {
    $moduleName = $folder.Name
    Write-Host "Generating docs for module: $packageName.$moduleName"
    python -m pydoc -w "$packageName.$moduleName"
    Move-Item -Path "$packageName.$moduleName.html" -Destination $docsDir -Force
}

$files = Get-ChildItem -Path ".\$packageName" -Recurse -File -Filter "*.py"
$dander_files_excluded = $files | Where-Object { $_.Name -notlike '_*' }
foreach ($file in $dander_files_excluded) {
    
    $module_Name = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
    $parentFolderName = Split-Path $file.DirectoryName -Leaf
    $parentFolder = Split-Path $file.DirectoryName -Parent 
    $module_name = "$parentFolderName.$module_Name"

    while ($parentFolderName -ne $packageName) {
        $parentFolderName = Split-Path $parentFolder -Leaf
        $module_name = "$parentFolderName.$module_Name"
        $parentFolder = Split-Path $parentFolder -Parent
    }
    
    Write-Host "Generating docs for module: $module_name"
    python -m pydoc -w "$module_name"
    Move-Item -Path "$module_name.html" -Destination $docsDir -Force
    #Write-Host $module_name  # You can use other properties as needed (e.g., $file.Name)
}