import utmosv2
import pandas as pd
import os

def main():
    # Setup your paths
    data_dir = r"C:\Users\ondra\Desktop\projekt\Test_dataset"
    output_csv = r"C:\Users\ondra\Desktop\projekt\test_results.csv"

    print("Downloading/Loading UTMOSv2 model (this might take a minute the first time)...")
    model = utmosv2.create_model(pretrained=True)
    
    print(f"Evaluating the 20 files in: {data_dir}")
    # The predict function handles the folder automatically
    results = model.predict(input_dir=data_dir, device="cpu", num_workers=0)
    
    # Save the results
    df = pd.DataFrame(results)
    df['file_path'] = df['file_path'].apply(os.path.basename)
    df = df.rename(columns={'file_path': 'file_name'})
    df.to_csv(output_csv, index=False)
    
    print(f"Success! Evaluated {len(results)} files.")
    print(f"Open {output_csv} to see the scores.")

if __name__ == "__main__":
    main()