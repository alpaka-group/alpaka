import json
import hashlib
import re
import sys
import os

def parse_benchmark_file(input_file):
    with open(input_file, 'r') as f:
        data = f.read()

    sections = data.split('AcceleratorType:')
    results = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract metadata information
        meta = {
            "creator": "unknown",
            "datetime": "unknown",
            "hostname": "unknown",
            "moreinterestingmetadata": "unknown"
        }

        env = {}
        workdivs = {}

        # Extract properties
        for line in section.splitlines():
            line = line.strip()
            if not line:
                continue

            if re.match(r'^[A-Za-z]', line) and ':' in line:
                key, value = map(str.strip, line.split(':', 1))
                if key.startswith("WorkDiv"):
                    # Store workdivs in a dictionary for later assignment to kernels
                    kernel_name = key.replace("WorkDiv", "").strip()
                    workdivs[kernel_name] = value
                else:
                    env[key] = value

        # Add the kind information
        env["kind"] = "Babelstream"

        # Extract kernel data
        kernels_section = re.search(r'Kernels\s+Bandwidths\(GB/s\).+', section, re.DOTALL)
        if kernels_section:
            kernel_lines = kernels_section.group(0).splitlines()[1:]
            for kernel_line in kernel_lines:
                parts = kernel_line.split()
                if len(parts) >= 6:
                    kernel_name = parts[0].replace("Kernel", "")  # Remove "Kernel" from the name if present
                    kernel_data = {
                        "name": kernel_name,
                        "Bandwidth(GB/s)": parts[1],
                        "MinTime(s)": parts[2],
                        "MaxTime(s)": parts[3],
                        "AvgTime(s)": parts[4],
                        "DataUsage(MB)": parts[5],
                        "WorkDiv": workdivs.get(kernel_name, ""),  # Fetch the correct workdiv
                        "AcceleratorType": env.get("AcceleratorType", ""),
                        "NumberOfRuns": env.get("NumberOfRuns", ""),
                        "Precision": env.get("Precision", ""),
                        "DataSize(items)": env.get("DataSize(items)", ""),
                        "DeviceName": env.get("DeviceName", ""),
                        "kind": env.get("kind", "")
                    }

                    # Generate a unique ID for each kernel data point and rename it to 'id_test'
                    kernel_id = hashlib.md5(json.dumps(kernel_data, sort_keys=True).encode('utf-8')).hexdigest()
                    kernel_data["id_test"] = kernel_id

                    results.append({
                        "id_test": kernel_id,
                        "meta": meta,
                        "env": env,
                        **kernel_data
                    })

    return results

def convert_to_json(input_file, output_file):
    data = parse_benchmark_file(input_file)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.isfile(input_file):
        print(f"Error: The input file '{input_file}' does not exist.")
        sys.exit(1)

    print(f"Processing input file '{input_file}' to output file '{output_file}'")
    convert_to_json(input_file, output_file)
    print(f"Conversion completed. JSON saved to '{output_file}'")
