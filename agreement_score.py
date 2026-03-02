import json
import os
from shapely.geometry import Polygon

def load_geojson_annotations(filepath):
    """
    Loads GeoJSON features and initializes the state for each annotation.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Handle both standard FeatureCollections and flat lists of features
    features = data.get("features", data) 
    
    annotations = []    
    for feature in features:
        # Extract the outer ring coordinates of the polygon
        coords = feature["geometry"]["coordinates"][0]
        poly = Polygon(coords)
        class_name = feature["properties"]["classification"]["name"]
        
        annotations.append({
            "polygon": poly,
            "class_name": class_name,
            "area": poly.area,
            "picked": False,
            "taken": True
        })
        
    return annotations

def calculate_agreement_score(file1_path, file2_path, iou_threshold=0.5, area_threshold=0.0):
    """
    Calculates the agreement score between two GeoJSON annotation files.
    """
    # 1. Initialization
    file1_anns = load_geojson_annotations(file1_path)
    file2_anns = load_geojson_annotations(file2_path)
    
    # 2. Sorting: Sort File 1 (Expert) by area in decreasing order
    file1_anns.sort(key=lambda x: x["area"], reverse=True)
    
    # 3. Matching Loop
    for ann1 in file1_anns:
        best_iou = 0.0
        best_match_idx = -1
        
        for j, ann2 in enumerate(file2_anns):
            # Only consider Pred boxes that are not picked and have the same class
            if not ann2["picked"] and ann1["class_name"] == ann2["class_name"]:
                
                # Calculate Intersection over Union (IoU)
                intersection_area = ann1["polygon"].intersection(ann2["polygon"]).area
                union_area = ann1["polygon"].union(ann2["polygon"]).area
                
                # Avoid division by zero just in case
                iou = intersection_area / union_area if union_area > 0 else 0.0
                
                # Keep track of the highest IoU
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = j
        
        # If the best match meets the IoU threshold, mark both as picked
        if best_iou >= iou_threshold and best_match_idx != -1:
            ann1["picked"] = True
            file2_anns[best_match_idx]["picked"] = True

    # 4. Forgiveness Filter (Area Threshold)
    all_annotations = file1_anns + file2_anns
    for ann in all_annotations:
        if not ann["picked"] and ann["area"] < area_threshold:
            ann["taken"] = False

    # ---------------------------------------------------------
    # NEW: Save all areas in increasing order to a .txt file
    # ---------------------------------------------------------
    # Extract areas and sort them ascending
    all_areas = sorted([ann["area"] for ann in all_annotations])
    
    # Save to the same directory as file1
    output_dir = os.path.dirname(file1_path)
    output_txt_path = os.path.join(output_dir, "sorted_areas.txt")
    
    with open(output_txt_path, "w") as f:
        f.write("Areas of all boxes (in increasing order):\n")
        for area in all_areas:
            f.write(f"{area}\n")
            
    print(f"--> Saved sorted areas to: {output_txt_path}")
    # ---------------------------------------------------------

    # 5. Scoring
    taken_and_picked = 0
    taken_and_not_picked = 0
    
    for ann in all_annotations:
        if ann["taken"]:
            if ann["picked"]:
                taken_and_picked += 1
            else:
                taken_and_not_picked += 1

    # Calculate final score (protecting against division by zero if all are filtered out)
    total_valid = taken_and_picked + taken_and_not_picked
    if total_valid == 0:
        return 0.0  # Or None, depending on how you want to handle empty valid sets
        
    agreement_score = taken_and_picked / total_valid
    
    # Optional: Print out the stats for debugging
    print(f"Total Taken & Picked (Matches * 2): {taken_and_picked}")
    print(f"Total Taken & Not Picked (Misses/False Alarms): {taken_and_not_picked}")
    
    return agreement_score

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Replace these with your actual file paths
    file1 = r"C:\Users\ansh\OneDrive\Desktop\desktop\wsi_extraction\agreement_score\2224413_a.geojson" 
    file2 = r"C:\Users\ansh\OneDrive\Desktop\desktop\wsi_extraction\agreement_score\2224413_c.geojson"
    
    # You can adjust your thresholds here
    SCORE = calculate_agreement_score(
        file1_path=file1, 
        file2_path=file2, 
        iou_threshold=0.1, 
        area_threshold=175483.0# Set this to 0 for strict evaluation, or higher to forgive tiny mistakes
    )
    
    print(f"Final Agreement Score: {SCORE:.4f}")