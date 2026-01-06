#!/bin/bash
# Function to display script usage

# Usage: $bash run.sh ./dataset/MD_test_534_10samples/ ./dataset/MD_test_534_out/result.json


function display_manual() {
echo "Usage: $0 [INPUT_DIR] [OUTPUT_JSON]"
echo " This script performs object detection on images in the specified input directory and saves the results to the specified output file."
echo ""
echo "Parameters:"
echo " INPUT_DIR Path to the input image directory."
echo " OUTPUT_JSON Path to the output detection result file."
echo ""
echo "Example:"
echo " $0 ./input/test_images ./output/dir/result.json"
echo ""
}

# Check if the required parameters are provided

if [[ $# -ne 2 ]]; then

echo "Error: Two parameters are required."
display_manual
exit 1
fi

# Extract input and output directory files from command-line arguments
input_dir="$1"
output_json="$2"

# Perform object detection using the provided input and output directories
# (Replace the following line with your actual object detection command)

echo "Performing object detection on the image directory: $input_dir"
echo "The detection results will be saved to the output file: $output_json"

#python3 your_application.py $input_dir $output_json
python deploy/python/infer_LSW_quiet.py --model_dir=output_inference/rtdetr_focalnet_L_384_3x_coco_MD --image_dir=$input_dir --device=GPU --output_dir=$output_json --batch_size=1 --save_images=False --cpu_threads=2 --threshold=0.1
# Add your object detection command here, using the input_dir and output_json variables
# For example: python detect_objects.py --input_dir "$input_dir" --output_json "$output_json"
# Replace the above line with your actual command for object detection

echo "Object detection process has been completed."

