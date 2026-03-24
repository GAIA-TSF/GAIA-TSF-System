import pandas as pd
import time
import json
from qcl import QCLayer

def run_qcl_demo():
    print("\n[System] Starting QCL (Quality Control Layer) standalone demo...")
    time.sleep(1)
    
    # Initialize the QCL Gatekeeper
    qcl = QCLayer()
    
    print("\n" + "="*60)
    print(" ACT I: In-Situ Data Physical Range Validation")
    print("="*60)
    
    input(">>> [Next Step] I will now simulate sending perfect sensor data (Press ENTER)... ")
    
    # Mock a perfect Pandas DataFrame
    good_df = pd.DataFrame({
        'ph': [7.2, 6.8, 8.1], 
        'temperature': [20.1, 19.5, 22.0]
    }, index=[101, 102, 103])
    good_metadata = {'sensor_type': 'water_sonde', 'source': 'ISU'}
    
    result_1 = qcl.check(data_type='in_situ', data=good_df, metadata=good_metadata, dataset_id='DS-ISU-001')
    print(f"\n[QCL Return Result]\n{json.dumps(result_1, indent=2, ensure_ascii=False)}")
    
    
    input("\n>>> [Next Step] I will now simulate sending contaminated sensor data with pH = 15.5 (Press ENTER)... ")
    
    # Mock data with values outside the physical range
    bad_df = pd.DataFrame({
        'ph': [7.2, 15.5, 8.1],  # 15.5 exceeds the 0-14 physical range rule
        'temperature': [20.1, 19.5, 22.0]
    }, index=[104, 105, 106])
    bad_metadata = {'sensor_type': 'water_sonde', 'source': 'ISU'}
    
    result_2 = qcl.check(data_type='in_situ', data=bad_df, metadata=bad_metadata, dataset_id='DS-ISU-002')
    print(f"\n[QCL Return Result]\n{json.dumps(result_2, indent=2, ensure_ascii=False)}")


    print("\n" + "="*60)
    print(" ACT II: EO Raster Metadata Validation")
    print("="*60)
    
    input(">>> [Next Step] I will now send satellite metadata with excessive null pixels (Press ENTER)... ")
    
    # Mock EO metadata where null_pixel_pct is 5% (exceeds the 2% threshold)
    eo_metadata = {
        'null_pixel_pct': 0.05, 
        'is_geometrically_aligned': True,
        'crs': 'EPSG:4326',
        'sensor_type': 'Sentinel-2'
    }
    
    result_3 = qcl.check(data_type='eo_raster', data=None, metadata=eo_metadata, dataset_id='DS-EO-001')
    print(f"\n[QCL Return Result]\n{json.dumps(result_3, indent=2, ensure_ascii=False)}")


    print("\n" + "="*60)
    print(" ACT III: Manual Review & Data Lineage Logger")
    print("="*60)
    
    input(">>> [Next Step] I will now simulate a human expert overriding the Warn status to Pass (Press ENTER)... ")
    
    # Call the manual_review interface to override the system status
    override_msg = qcl._engine.manual_review(dataset_id='DS-EO-001', action='override', new_status='Pass')
    print(f"\n[Action Confirmed] {override_msg}")
    
    print("\n[Data Lineage Log] Let's see how the system tracks these changes for auditing:")
    print("👉 Please look at the terminal logs above marked with [QC DB LOG]:")
    print("   1. First decision: actions: ['Null Pixel Check', ...] -> status: 'Warn'")
    print("   2. After human intervention: actions: ['Manual Override to Pass'] -> status: 'Pass'")
    print("   (This proves every status change is securely tracked for compliance auditing!)")
    
    print("\n" + "="*60)
    print(" QCL Demo Finished")
    print("="*60)

if __name__ == "__main__":
    run_qcl_demo()