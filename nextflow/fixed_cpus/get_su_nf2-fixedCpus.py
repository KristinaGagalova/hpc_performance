#!/usr/bin/env python3

# Author: Dr. Kristina K. Gagalova
# Created on: 25 September 2024
# email: kristina.gagalova@curtin.edu.au

import sys
import re

# Function to convert time strings into hours
def convert_time_to_hours(time_str):
    hours = minutes = seconds = 0
    if "h" in time_str:
        hours = int(time_str.split("h")[0].strip())
        time_str = time_str.split("h")[1].strip()
    if "m" in time_str:
        minutes = int(time_str.split("m")[0].strip())
        time_str = time_str.split("m")[1].strip()
    if "s" in time_str:
        seconds = float(time_str.split("s")[0].strip())
    return hours + minutes / 60 + seconds / 3600

def format_memory(memory_value):
    """
    Correctly format memory value with a unit, handling missing values or unexpected formats.
    """
    # Handling edge cases where memory might be blank or doesn't match expected patterns
    if memory_value.strip() == '':
        return 'N/A'

    # Checking if the memory value is a valid format (numeric + unit)
    match = re.match(r'([\d.]+)\s*(\w+)', memory_value)
    if match:
        value, unit = match.groups()
        return f"{value} {unit}"
    else:
        # If format doesn't match, return the raw value
        return memory_value

def clean_name(name):
    """
    Remove everything between brackets (including brackets) from the name.
    """
    return re.sub(r'\(.*?\)', '', name).strip()

def main():
    # Check if the CPU count and data file were passed as arguments
    if len(sys.argv) != 3:
        print("Usage: python3 myscript.py <CPUs> <input_data_file>")
        sys.exit(1)

    # Get the CPU count from the command-line arguments
    cpus = int(sys.argv[1])

   # Read the input data file
    input_file = sys.argv[2]
    with open(input_file, 'r') as f:
        data = f.read()

    # Split input data into lines and handle tab-separated format
    lines = data.strip().split("\n")

    # Extract header and rows, splitting by tabs
    header = lines[0].split("\t")
    rows = [line.split("\t") for line in lines[1:]]

    # Prepare output
    output = []
    total_su = 0  # Initialize variable to store total SU

    for row in rows:
        name = clean_name(row[3])
        duration = row[8]
        peak_rss = format_memory(row[10])

        # Parse the duration to hours
        duration_hours = round(convert_time_to_hours(duration), 2)

        # Calculate SU
        cores_per_cpu = 64  # Assuming 64 cores per CPU
        SU = duration_hours * (cores_per_cpu * cpus)
        total_su += SU  # Sum the SU values

        # Append results to output
        output.append({
            'name': name,
            'duration': duration,
            'duration_hours': duration_hours,
            'peak_rss': peak_rss,
            'SU': SU,
            'CPUs': cpus,
            'total_cores': cores_per_cpu * cpus,
        })

    # Print the result in a table-like format
    print(f"{'Name'},{'Duration'},{'Hours'},{'Peak RSS'},{'SU'},{'CPUs'},{'Cores'}")
    for task in output:
        print(f"{task['name']},{task['duration']},{task['duration_hours']},{task['peak_rss']},{task['SU']},{task['CPUs']},{task['total_cores']}")

    # Print the total SU
    print("\nTotal SU: {:.4f}".format(total_su))

if __name__ == "__main__":
    main()
