import xml.etree.ElementTree as ET
import json
import html
import sys
import os
import re

def parse_warning_node(warning_text):
    warning_dict = {}
    for line in warning_text.strip().split('\n'):
        if ':' in line:
            field_name, value = line.split(':', 1)
            warning_dict[field_name.strip()] = convert_to_number(value.strip())
    return warning_dict

def convert_to_number(value):
    # Check if the value is an integer
    if re.match(r'^-?\d+$', value):
        return int(value)
    # Check if the value is a real number
    elif re.match(r'^-?\d*\.\d+$', value):
        return float(value)
    # Return the value as a string if it's neither an integer nor a real number
    return value

def xml_to_dict(element):
    node_dict = {}
    # Directly add attributes without '@'
    for key, value in element.attrib.items():
        node_dict[key] = convert_to_number(value)

    if element.text and element.text.strip():
        node_dict[element.tag] = convert_to_number(html.unescape(element.text.strip()))

    special_tags = {'mean', 'standardDeviation', 'outliers'}

    for child in list(element):
        child_dict = xml_to_dict(child)
        if child.tag == 'Warning':
            warning_dict = parse_warning_node(child.text)
            node_dict['MetaBenchmarkTestData'] = warning_dict
        elif child.tag in special_tags:
            node_dict[child.tag] = child_dict[child.tag]
        else:
            if child.tag not in node_dict:
                node_dict[child.tag] = []
            node_dict[child.tag].append(child_dict)

    return {element.tag: node_dict}

def consolidate_dict(d):
    if isinstance(d, dict):
        for key in d:
            if isinstance(d[key], list) and len(d[key]) == 1:
                d[key] = d[key][0]
            consolidate_dict(d[key])

def xml_to_json(xml_string):
    root = ET.fromstring(xml_string)
    xml_dict = xml_to_dict(root)
    consolidate_dict(xml_dict)
    return json.dumps(xml_dict, indent=4)

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
    json_output = xml_to_json(xml_content)

    # Save the JSON output to the output file
    with open(output_file, 'w') as json_file:
        json_file.write(json_output)

    print(f"XML content has been converted to JSON and saved as '{output_file}'")

