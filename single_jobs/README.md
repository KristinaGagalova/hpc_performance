## Launch the same command with different CPU amount and get time

How to run the script:
```
python3 get_su_threads_cpus.py <cpus> [command] [account] [job_partition] 
```    
* cpus = comma delimited list of cpus to test (ex: 2,4,8)             
* command = command to be benchmarked, must be between quotations, number of cpus and threads will be added to the command according to Setonix specifications
* account = hpc account      
* job_partition = job cluster partition to use for jobs     

