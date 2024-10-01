#!/usr/bin/env python3

# Author: Dr. Kristina K. Gagalova
# Created on: 25 September 2024
# email: kristina.gagalova@curtin.edu.au

import subprocess
import time
import os
import sys

def create_slurm_script(num_cpus, command, job_name, account, partition):
    """
    Create a Slurm job script to run a command with the specified number of CPUs and exclusive node access.

    Parameters:
    num_cpus (int): Number of CPUs for the Slurm job
    command (str): Command to run in the job, with placeholders {cores} and {cpus} for the number of cores and output directory
    job_name (str): Name of the job
    account (str): Slurm account to use for the job
    partition (str): Slurm partition to submit to

    Returns:
    str: The filename of the generated Slurm script
    """
    # Calculate the total number of cores
    cores = num_cpus * 64  # 64 cores per CPU on Setonix

    # Replace the {cores} and {cpus} placeholders in the command with the actual number of cores and CPUs
    command_with_cores = command.replace("{cores}", str(cores)).replace("{cpus}", f"{num_cpus}")

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --ntasks={num_cpus}
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00  # Adjust this as necessary
#SBATCH --account={account}  # Specify your account
#SBATCH --partition={partition}  # Specify the partition
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mem-per-cpu=200GB

echo "Running on $SLURM_JOB_NODELIST"
echo "Using {num_cpus} CPUs and {cores} cores"

# Record start time
start_time=$(date +%s)

# Run the command with the number of cores and output directory
{command_with_cores}

# Record end time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

# Write the elapsed time to a file
echo "Elapsed time: $elapsed_time seconds" > job-{num_cpus}.time
"""

    # Write the Slurm script to a file
    script_filename = f"slurm_job_{num_cpus}.sh"
    with open(script_filename, 'w') as script_file:
        script_file.write(script_content)

    return script_filename

def submit_slurm_job(script_filename):
    """
    Submit the Slurm job and return the job ID.

    Parameters:
    script_filename (str): The Slurm script to submit

    Returns:
    str: The job ID of the submitted job
    """
    result = subprocess.run(['sbatch', script_filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    if result.returncode != 0:
        return f"Error submitting job: {result.stderr}"  # Return error if job submission fails

    output = result.stdout
    try:
        # Extract the job ID from the output
        job_id = output.strip().split()[-1]
        return job_id
    except IndexError:
        return "Error: Failed to extract job ID from sbatch output."

def wait_for_job_completion(job_id):
    """
    Wait for the Slurm job to complete by checking its status with sacct.

    Parameters:
    job_id (str): The Slurm job ID to check

    Returns:
    None
    """
    while True:
        result = subprocess.run(['sacct', '-j', job_id, '--format=State', '--noheader'], stdout=subprocess.PIPE, universal_newlines=True)
        job_status = result.stdout.strip()

        # If the job is completed, failed, or cancelled, exit the loop
        if "COMPLETED" in job_status or "FAILED" in job_status or "CANCELLED" in job_status:
            break
        time.sleep(10)  # Wait 10 seconds before checking again

def calculate_su(num_cpus, cores_per_cpu, wall_time_hours, su_cost_per_core_hour):
    """
    Function to calculate Service Units (SU) for an HPC job based on the number of CPUs.

    Parameters:
    num_cpus (int): Number of CPUs used in the job
    cores_per_cpu (int): Number of cores per CPU (e.g., 64 cores for AMD Milan on Setonix)
    wall_time_hours (float): Wall-clock time the job runs in hours
    su_cost_per_core_hour (float): SU cost per core-hour

    Returns:
    float: Total SU cost for the job
    """
    num_cores = num_cpus * cores_per_cpu
    total_su = num_cores * wall_time_hours * su_cost_per_core_hour
    return total_su

def read_wall_time(cpu_count):
    """
    Read the wall time from the job-{cpu}.time file.

    Parameters:
    cpu_count (int): The number of CPUs used in the job

    Returns:
    float: The elapsed wall time in seconds
    """
    try:
        with open(f"job-{cpu_count}.time", "r") as time_file:
            time_data = time_file.read().strip()
            elapsed_time = float(time_data.split()[2])  # Extract the elapsed time
            return elapsed_time / 3600  # Convert seconds to hours
    except FileNotFoundError:
        return 0

def main():
    # Command-line argument parsing
    if len(sys.argv) < 5:
        print("Usage: python get_su.py <cpu_list> <command> <account> <partition>")
        print("Example: python get_su.py 2,4,8,16,32 'my_tool --consCores {cores} -o output_{cpus}' my_account my_partition")
        sys.exit(1)

    # Parse CPUs to test, the command, account, and partition
    cpu_list_str = sys.argv[1]
    command = sys.argv[2]
    account = sys.argv[3]
    partition = sys.argv[4]

    # Open a log file to store only the final output
    with open("job_output.log", "w") as log_file:
        # Convert the CPU list to integers
        cpu_list = [int(cpu) for cpu in cpu_list_str.split(",")]

        # Constants
        cores_per_cpu = 64  # Number of cores per CPU on Setonix
        su_cost_per_core_hour = 1  # SU cost per core-hour on Setonix

        # Write the table header to the file
        log_file.write(f"{'CPUs':<10}{'Cores':<10}{'SU':<10}\n")
        log_file.write("="*30 + "\n")

        # Loop through different CPU configurations
        job_ids = {}

        for num_cpus in cpu_list:
            # Create a Slurm job script for this configuration
            job_name = f"test_job_{num_cpus}_cpus"
            script_filename = create_slurm_script(num_cpus, command, job_name, account, partition)

            # Submit the Slurm job and get the job ID
            job_id = submit_slurm_job(script_filename)
            if "Error" not in job_id:
                job_ids[num_cpus] = job_id

        # Wait for all jobs to finish
        for num_cpus, job_id in job_ids.items():
            wait_for_job_completion(job_id)

        # Once all jobs are done, calculate SU and write the results to the log file
        for num_cpus in cpu_list:
            # Read the wall time from the job-{cpu}.time file
            wall_time_hours = read_wall_time(num_cpus)

            # Calculate SU
            total_su = calculate_su(num_cpus, cores_per_cpu, wall_time_hours, su_cost_per_core_hour)

            # Write the final results in tabular format to the log file
            num_cores = num_cpus * cores_per_cpu
            log_file.write(f"{num_cpus:<10}{num_cores:<10}{total_su:<10.2f}\n")
            log_file.flush()

if __name__ == "__main__":
    main()

