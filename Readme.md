pip install numpy opencv-python torch torchvision lpips scikit-image


python evaluate_inpainting.py test_manifest.csv --output evaluation_results --missing-value 255 --expected-per-bin 250


python evaluate_inpainting.py test_manifest.csv --output evaluation_results_raw --missing-value 255 --expected-per-bin 250 --compose
