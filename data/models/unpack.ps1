$models = @(
    "cnn_model_20250214_222252",
    "unet_model_20250212_002522",
    "cnn_model_20250214_222252.zip",
    "unet_model_20250224_183427.zip",
    "cnn_model_20250529_005118.zip",
    "unet_model_20250529_000149.zip"
)

foreach ($model in $models) {
    Expand-Archive -Path "$model.zip" -DestinationPath . -Force
    Move-Item ".\data\models\$model" ".\$model"
}

Remove-Item ".\data" -Recurse -Force
