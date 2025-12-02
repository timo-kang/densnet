#!/bin/bash
# Check COLMAP reconstruction statistics

COLMAP_ROOT="/home/test1/workspace/densnet/training_data/dataset_25frames"

echo "=== Checking first 10 COLMAP reconstructions ==="
echo ""

count=0
for sparse_dir in $(find "$COLMAP_ROOT" -type d -path "*/sparse/0" | head -10); do
    images_file="$sparse_dir/images.txt"

    if [ -f "$images_file" ]; then
        # Count images (images.txt has 2 lines per image)
        num_images=$(($(grep -v "^#" "$images_file" | wc -l) / 2))

        seq_name=$(basename $(dirname $(dirname "$sparse_dir")))
        echo "$seq_name: $num_images images reconstructed"

        # Show first 3 image names
        echo "  Images:"
        grep -v "^#" "$images_file" | awk 'NR%2==1 {print "    " $10}' | head -3

        count=$((count + 1))
    fi
done

echo ""
echo "=== Full statistics ==="
total_sequences=0
total_images=0
min_images=999999
max_images=0

for sparse_dir in $(find "$COLMAP_ROOT" -type d -path "*/sparse/0"); do
    images_file="$sparse_dir/images.txt"

    if [ -f "$images_file" ]; then
        num_images=$(($(grep -v "^#" "$images_file" | wc -l) / 2))
        total_sequences=$((total_sequences + 1))
        total_images=$((total_images + num_images))

        if [ $num_images -lt $min_images ]; then
            min_images=$num_images
        fi
        if [ $num_images -gt $max_images ]; then
            max_images=$num_images
        fi

        # Count sequences by image count
        if [ $num_images -eq 2 ]; then
            two_img=$((two_img + 1))
        elif [ $num_images -lt 10 ]; then
            few_img=$((few_img + 1))
        elif [ $num_images -lt 20 ]; then
            medium_img=$((medium_img + 1))
        else
            many_img=$((many_img + 1))
        fi
    fi
done

if [ $total_sequences -gt 0 ]; then
    avg_images=$((total_images / total_sequences))
    echo "Total sequences with COLMAP data: $total_sequences"
    echo "Total images reconstructed: $total_images"
    echo "Average images per sequence: $avg_images"
    echo "Min images: $min_images"
    echo "Max images: $max_images"
    echo ""
    echo "Distribution:"
    echo "  Exactly 2 images: ${two_img:-0} sequences"
    echo "  3-9 images: ${few_img:-0} sequences"
    echo "  10-19 images: ${medium_img:-0} sequences"
    echo "  20+ images: ${many_img:-0} sequences"
fi
