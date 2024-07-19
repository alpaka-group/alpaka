#!/usr/bin/env python3

import os
import sys
import json
import hashlib
import xml.etree.ElementTree as ET
import html

# Function to parse the content of a <Warning> node into a dictionary
# This is needed because Catch2 INFO macro is not printed if the test case passes, only WARNING is printed
def parse_warning_node(warning_text):
    warning_dict = {}
    for line in warning_text.strip().split('\n'):
        if ':' in line:
            field_name, value = line.split(':', 1)
            warning_dict[field_name.strip()] = value.strip()
    return warning_dict

# Function to recursively convert XML elements into a dictionary
def xml_to_dict(element):
    node_dict = {}
    # Add XML attributes directly to the dictionary
    for key, value in element.attrib.items():
        node_dict[key] = html.unescape(value)

    # Add text content of the element if it exists
    text = element.text.strip() if element.text else ''
    if text:
        node_dict['text'] = html.unescape(text)

    special_tags = {'mean', 'standardDeviation', 'outliers'}

    # Process each child element recursively
    for child in list(element):
        child_dict = xml_to_dict(child)
        if child.tag == 'Warning':
            warning_dict = parse_warning_node(child.text)
            node_dict['MetaBenchmarkTestData'] = warning_dict
        elif child.tag in special_tags:
            if child.tag not in node_dict:
                node_dict[child.tag] = {}
            node_dict[child.tag].update(child_dict)
        else:
            if child.tag not in node_dict:
                node_dict[child.tag] = []
            node_dict[child.tag].append(child_dict)

    # Convert single-item lists to single objects
    for key, value in node_dict.items():
        if isinstance(value, list) and len(value) == 1:
            node_dict[key] = value[0]

    return node_dict

# Function to convert the XML string into a JSON-compatible dictionary
def xml_to_json(xml_string):
    root = ET.fromstring(xml_string)
    root_dict = xml_to_dict(root)
    
    # Extract and structure TestCase elements
    test_cases = root_dict.pop('TestCase', [])
    
    if not isinstance(test_cases, list):
        test_cases = [test_cases]
    
    # Create the final dictionary with the root tag and TestCase entries
    final_dict = {
        root.tag: root_dict,
        'TestCase': test_cases
    }

    return final_dict

# Function to transform the JSON data into the desired format
def transform(json_data):
    output_object = []

    if 'Catch2TestRun' not in json_data:
        print("Error: 'Catch2TestRun' not found in input JSON")
        return []

    # Extract general metadata
    o1 = json_data['Catch2TestRun']

    # Create metadata and environment subobjects
    meta = {
        'creator': "unknown",
        'datetime': "unknown",
        'hostname': "unknown",
        'moreinterestingmetadata': "unknown"
    }

    env = {
        'type': 'Catch2TestRun',
        'kind': o1.get('name', 'unknown'),
        'rng-seed': o1.get('rng-seed', 'unknown'),
        'catch2-version': o1.get('catch2-version', 'unknown'),
        'OverallResults': o1.get('OverallResults', {}),
        'OverallResultsCases': o1.get('OverallResultsCases', {})
    }

    # Process each TestCase if it exists
    test_cases = json_data.get('TestCase', [])
    if not isinstance(test_cases, list):
        test_cases = [test_cases]

    for tc in test_cases:
        # Ensure BenchmarkResults is a list
        benchmark_results = tc.get('BenchmarkResults', [])
        if not isinstance(benchmark_results, list):
            benchmark_results = [benchmark_results]

        for m in benchmark_results:
            out = {}
            out['id'] = ""
            out['meta'] = meta
            out['env'] = env
            out['TestCase'] = {
                'name': tc.get('name', 'unknown'),
                'tags': tc.get('tags', 'unknown'),
                'filename': tc.get('filename', 'unknown'),
                'line': tc.get('line', 'unknown'),
                'OverallResult': tc.get('OverallResult', {}),
                'MetaBenchmarkTestData': tc.get('MetaBenchmarkTestData', {})
            }

            # Add benchmark result details
            for k in m:
                out[k] = m[k]

            # Insert md5sum as unique id of the dataset
            hash = hashlib.md5(json.dumps(out, sort_keys=True).encode('utf-8')).hexdigest()
            out['id'] = hash

            output_object.append(out)

    return output_object

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input.xml> <output.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not input_file.endswith(".xml"):
        print("Error: Input file must have a .xml extension")
        sys.exit(1)

    if not output_file.endswith(".json"):
        print("Error: Output file must have a .json extension")
        sys.exit(1)

    # Check if input file exists
    if not os.path.isfile(input_file):
        print(f"Error: Input file '{input_file}' does not exist")
        sys.exit(1)

    # Read from the input XML file
    with open(input_file, 'r') as xml_file:
        xml_content = xml_file.read()

    # Convert XML to JSON
    json_data = xml_to_json(xml_content)

    # Transform JSON to desired format
    transformed_data = transform(json_data)

    # Save the transformed JSON output to the output file
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(transformed_data, json_file, indent=4)

    print(f"XML content has been converted to JSON and transformed, saved as '{output_file}'")

