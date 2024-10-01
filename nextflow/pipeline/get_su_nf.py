#/usr/bin/env python3

# Author: Dr. Kristina K. Gagalova
# Created on: 25 September 2024
# email: kristina.gagalova@curtin.edu.au

import argparse
import csv

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

# Function to calculate service units (SU)
def calculate_su(cpu_percent, duration_str, cores_per_node=64):
    duration_hours = convert_time_to_hours(duration_str)
    su = (cpu_percent / 100) * cores_per_node * duration_hours
    return su

# Function to calculate total cores based on CPU utilization
def calculate_total_cores(cpu_percent, cores_per_cpu=64):
    total_cores = (cpu_percent / 100) * cores_per_cpu
    return total_cores

# Function to parse the input file and extract data for calculation using csv reader
def parse_input_file(file_path):
    tasks = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')  # Using tab as delimiter
        next(reader)  # Skip the header row
        for row in reader:
            task_id = row[0]
            duration = row[7]  # Extract duration
            cpu_percent = float(row[9].replace('%', ''))  # Extract CPU percentage
            peak_rss = row[10]  # Extract peak_rss
            tasks.append((task_id, cpu_percent, duration, peak_rss))
    return tasks

# Function to output CPUs (not rounded), cores, SU, peak_rss, and duration in hours for each task
def output_results(tasks):
    cores_per_node = 64  # Assuming 64 cores per CPU (for Setonix)
    total_su = 0
    for task in tasks:
        task_id, cpu_percent, duration_str, peak_rss = task
        su = calculate_su(cpu_percent, duration_str, cores_per_node)
        total_su += su
        total_cores = calculate_total_cores(cpu_percent, cores_per_node)
        duration_hours = convert_time_to_hours(duration_str)

        print(f"Task {task_id}:")
        print(f"  CPUs: {cpu_percent}%")
        print(f"  Total Cores Used: {total_cores:.2f}")
        print(f"  Duration: {duration_hours:.2f} hours")
        print(f"  Peak RSS: {peak_rss}")
        print(f"  Service Units (SU): {su:.2f}")
    print(f"\nTotal Service Units (SU) for all tasks: {total_su:.2f}")

# Main function to parse arguments
def main():
    parser = argparse.ArgumentParser(description="Calculate Service Units (SU) for a job from input file.")
    
    # Define expected arguments
    parser.add_argument("file_path", type=str, help="Path to the input file")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Parse the input file
    tasks = parse_input_file(args.file_path)
    
    # Output the results
    output_results(tasks)

if __name__ == "__main__":
    main()
