#!/bin/bash
# Quick evaluation script - handles both subset and full evaluation

NUM_IMAGES=${1:-10}  # Default to 10 if not specified

echo "========================================"
echo "COCO Panoptic Evaluation"
echo "========================================"

# Check if we're doing subset or full evaluation
if [ $NUM_IMAGES -lt 5000 ]; then
    echo "Mode: SUBSET EVALUATION ($NUM_IMAGES images)"
    
    # Create subset ground truth if it doesn't exist
    SUBSET_GT="./datasets/coco/annotations/panoptic_val2017_subset_${NUM_IMAGES}.json"
    
    if [ ! -f "$SUBSET_GT" ]; then
        echo "Creating subset ground truth..."
        python3 create_subset_gt.py \
            --input ./datasets/coco/annotations/panoptic_val2017.json \
            --output "$SUBSET_GT" \
            --num-images $NUM_IMAGES
        echo ""
    fi
    
    GT_JSON="$SUBSET_GT"
else
    echo "Mode: FULL EVALUATION (all 5000 images)"
    GT_JSON="./datasets/coco/annotations/panoptic_val2017.json"
fi

# Check if predictions exist
PRED_JSON="./inference_coco_val/coco_val_panoptic_predictions.json"
PRED_FOLDER="./inference_coco_val/panoptic_predictions"

if [ ! -f "$PRED_JSON" ]; then
    echo "ERROR: Predictions not found at $PRED_JSON"
    echo "Run inference first!"
    exit 1
fi

if [ ! -d "$PRED_FOLDER" ]; then
    echo "ERROR: Prediction folder not found at $PRED_FOLDER"
    echo "Run inference first!"
    exit 1
fi

# Count predictions
NUM_PREDS=$(ls -1 "$PRED_FOLDER"/*.png 2>/dev/null | wc -l)
echo "Found $NUM_PREDS prediction files"

if [ $NUM_PREDS -eq 0 ]; then
    echo "ERROR: No prediction PNG files found!"
    exit 1
fi

echo ""
echo "Ground truth: $GT_JSON"
echo "Predictions: $PRED_JSON ($NUM_PREDS images)"
echo ""
echo "Starting evaluation..."
echo "========================================"

# Run evaluation
python3 -m panopticapi.evaluation \
    --gt_json_file "$GT_JSON" \
    --gt_folder ./datasets/coco/panoptic_val2017 \
    --pred_json_file "$PRED_JSON" \
    --pred_folder "$PRED_FOLDER"

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "========================================"
