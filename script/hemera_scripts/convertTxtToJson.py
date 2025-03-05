import json
import re
import sys
import os

def parse_header_fields(lines):
    header_fields = {}
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.endswith(":") and line.count(" ") < 5:  # Likely a section heading
            break
        if ":" in line:
            key, value = line.split(":", 1)
            header_fields[key.strip()] = value.strip()
        index += 1
    return header_fields, index

def parse_sections(lines, end_marker):
    sections = {}
    index = 0
    current_key = None
    current_value = []

    while index < len(lines):
        line = lines[index].strip()
        if line == end_marker:
            if current_key:
                sections[current_key] = "\n".join(current_value).strip()
            index += 1
            break
        if line.endswith(":"):
            if current_key:
                sections[current_key] = "\n".join(current_value).strip()
            current_key = line[:-1]
            current_value = []
        elif line and not re.match(r"^[-]+$", line):
            current_value.append(line)
        index += 1

    return sections, index

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

    data = {}

    header_fields, index = parse_header_fields(lines)
    data.update(header_fields)

    sections, sec_index = parse_sections(lines[index:], "Benchmark Results:")
    data.update(sections)
    index += sec_index

    benchmark_results = []
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

            benchmark_results.append(entry)
        else:
            index += 1

    data["BenchmarkResults"] = benchmark_results

    json_filename = os.path.splitext(filename)[0] + ".json"
    with open(json_filename, "w") as json_file:
        json.dump(data, json_file, indent=4)

# Usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    parse_txt_to_json(filename)

