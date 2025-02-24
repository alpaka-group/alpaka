import json
import re
import sys
import os

def parse_workdiv(lines, index):
    workdiv = {}
    while index < len(lines) and lines[index].startswith("WorkDiv"):
        parts = lines[index].split(":", 1)
        kernel_name = parts[0].replace("WorkDiv", "").strip()
        values = re.findall(r"\d+", parts[1]) if len(parts) > 1 else []
        
        if len(values) == 3:
            workdiv[kernel_name] = {
                "gridBlockExtent": int(values[0]),
                "blockThreadExtent": int(values[1]),
                "threadElemExtent": int(values[2])
            }
        else:
            print(f"Error: WorkDiv format issue at line {index}: {lines[index]}")
            sys.exit(1)
        
        index += 1
    return workdiv, index

def parse_kernel_performance(lines, index):
    kernels = {}
    index += 1  # Skip the header line
    while index < len(lines) and lines[index].strip():
        parts = lines[index].split()
        if len(parts) >= 6:
            kernel_name = parts[0].replace("Kernel", "").strip()
            kernels[kernel_name] = {
                "Bandwidth_GBps": float(parts[1]),
                "MinTime_s": float(parts[2]),
                "MaxTime_s": float(parts[3]),
                "AvgTime_s": float(parts[4]),
                "DataUsage_MB": float(parts[5])
            }
        index += 1
    return kernels, index

def parse_txt_to_json(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    
    data = []
    index = 0
    while index < len(lines):
        if "AcceleratorType" in lines[index]:
            entry = {}
            entry["AcceleratorType"] = lines[index].split(":")[1].strip()
            index += 1
            entry["NumberOfRuns"] = int(lines[index].split(":")[1].strip())
            index += 1
            entry["Precision"] = lines[index].split(":")[1].strip()
            index += 1
            entry["DataSize"] = int(lines[index].split(":")[1].strip())
            index += 1
            entry["DeviceName"] = lines[index].split(":")[1].strip()
            index += 1
            
            entry["WorkDiv"], index = parse_workdiv(lines, index)
            
            if index < len(lines) and "AccToHost Memcpy Time" in lines[index]:
                entry["AccToHostMemcpyTime_sec"] = float(lines[index].split(":")[1].strip())
                index += 1
                
            if index < len(lines) and "Kernels" in lines[index]:
                entry["Kernels"], index = parse_kernel_performance(lines, index)
                
            data.append(entry)
        else:
            index += 1
    
    json_filename = os.path.splitext(filename)[0] + ".json"
    with open(json_filename, "w") as json_file:
        json.dump(data, json_file, indent=4)
    
    return data

# Usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    parse_txt_to_json(filename)

