# Script to convert XML benchmark results reported by XML reporter of Catch2 to JSON format.
# The script changes the name of the "Warning" node to MetaBenchmarkTestData node. The field:value pairs of the Warning node are stored in a dictionary and converted to JSON format.
# This script is called by 2 comman line arguments. 
# The first one is input xml file and the output is json file.
# e.g: benchmarkXml2Json BabelStreamBenchmarkResults.xml BabelStreamBenchmarkResults.json


import xml.etree.ElementTree as ET
import json
import html
import sys
import os

def parse_warning_node(warning_text):
    warning_dict = {}
    for line in warning_text.strip().split('\n'):
        if ':' in line:
            field_name, value = line.split(':', 1)
            warning_dict[field_name.strip()] = value.strip()
    return warning_dict

def xml_to_dict(element):
    node_dict = {}
    if element.attrib:
        node_dict.update(element.attrib)

    if element.text and element.text.strip():
        node_dict[element.tag] = html.unescape(element.text.strip())

    for child in element:
        child_dict = xml_to_dict(child)
        if child.tag == 'Warning':
            warning_dict = parse_warning_node(child.text)
            node_dict['MetaBenchmarkTestData'] = warning_dict
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

    print(f"XML content of '{input_file}' has been converted to JSON and saved as '{output_file}'")
    
