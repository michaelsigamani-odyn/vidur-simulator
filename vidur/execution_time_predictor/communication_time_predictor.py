import time
from vidur.entities import Batch
from enum import Enum
from vidur.config import ReplicaConfig, BaseExecutionTimePredictorConfig, BaseModelConfig
from vidur.execution_time_predictor.SimAIWorkload import SimAIWorkload, WorkItem
from typing import Dict
import subprocess
import csv
import os
import hashlib

# TPTimePredictor
# allreduce
# TPTimePredictor is a singleton
# Only consider single allreduce time
class TPTimePredictor:
    def __init__(self,
        model_config: BaseModelConfig,
        replica_config: ReplicaConfig,
        predictor_config: BaseExecutionTimePredictorConfig
    ):
        assert model_config.num_layers % replica_config.num_pipeline_stages == 0
        self.num_layers_per_pp_stage = model_config.num_layers // replica_config.num_pipeline_stages
        self.hidden_size = model_config.embedding_dim
        self.predictor_config = predictor_config
        self.replica_config = replica_config
        # TODO ct: change to sizeof(tensor.dtype)
        # TODO: dynamically adjust dtype from config
        # :  config  dtype
        self.tensor_size = 2
        self.workload: SimAIWorkload = SimAIWorkload(
            tp_size=replica_config.tensor_parallel_size,
            ep_size=1,
            pp_size=replica_config.num_pipeline_stages,
            vpp_size=self.num_layers_per_pp_stage,
            ga_num=1,
            world_size=replica_config.world_size,
            pp_comm=0
        )
        self.simai_dir = os.path.abspath(predictor_config.simai_dir)
        self.simai_ns3_binary = f'{self.simai_dir}/bin/SimAI_simulator'
        self.simai_analytical_binary = f'{self.simai_dir}/bin/SimAI_analytical'
        # self.workload_path = '/tmp'
        self.workload_path = f'{self.simai_dir}/tmp_simai_inference_workload'
        # workload_path，
        # Ensure workload_path exists, create if not exists
        if not os.path.exists(self.workload_path):
            os.makedirs(self.workload_path)
        else:
            assert os.path.isdir(self.workload_path), f"Workload path exists but is not a directory: {self.workload_path}"
        
        # self.workload_path = '/disk2/futianhao/software3/sim-ai-inference-n/odyn_simulator_output/tmp_simai_workload'
        self.cache: Dict[int, float] = {}


    # >   workload  command
    # > rewrite: add two features to reuse same workloads and results of same commands
    def get_execution_time(self, batch: Batch):
        self.workload.flush()
        num_tokens_in_batch = batch._total_num_tokens_rounded
        all_reduce_bytes = self.hidden_size * num_tokens_in_batch * self.tensor_size
        
        # ，all_reduce_bytes
        # Use a tuple containing all relevant parameters as cache key instead of just all_reduce_bytes
        cache_key = (self.hidden_size, num_tokens_in_batch, self.tensor_size)
        
        # ，
        # If result is already in cache, return directly
        
        if cache_key in self.cache:
            return (self.cache[cache_key]
                + self.predictor_config.nccl_cpu_launch_overhead_ms
                + self.predictor_config.nccl_cpu_skew_overhead_per_device_ms
                * self.replica_config.tensor_parallel_size**1.25)
        
        # workload
        # WorkItem
        # Generate unique identifier for workload and command
        # Generate hash based on all relevant parameters of WorkItem
            
        
        # TODO(tianhao909): add layer0 str(1) "ALLREDUCE" etc. to hash computation, currently hardcoded
        # TODO(tianhao909):  layer0 str(1) "ALLREDUCE"  hash ，
            
        work_item_data = (
            "layer0" + 
            str(1) + 
            "ALLREDUCE" + 
            str(self.hidden_size) +
            str(num_tokens_in_batch) +
            str(self.tensor_size)
        )
        
        workload_identifier = hashlib.md5(work_item_data.encode()).hexdigest()
        
        
        # workload
        # Check if corresponding workload file already exists
        workload_file = f'{self.workload_path}/allreduce_{workload_identifier}.txt'
        
        # workload，
        # Generate if workload file does not exist
        if not os.path.exists(workload_file):
            # layer
            # Assume each layer is homogeneous
            self.workload.append_work_item(
                WorkItem(
                    name="layer0",
                    forward_compute_time=1,
                    forward_comm="ALLREDUCE",
                    forward_comm_size=all_reduce_bytes
                )
            )
            self.workload.dump_file(workload_file)
        
        topo = os.path.abspath(self.predictor_config.simai_simulation_topo)
        conf = os.path.abspath(self.predictor_config.simai_simulation_config)
        
        # 
        # Check if result cache for corresponding command already exists
        command_identifier = hashlib.md5(
            f'{workload_identifier}_{topo}_{conf}'.encode()
        ).hexdigest()
        result_file = f'{self.simai_dir}/ncclFlowModel_EndToEnd_{command_identifier}.csv'
        
        # ，
        # Run command if result file does not exist
        if not os.path.exists(result_file):
            # simai
            # Call simai
            command = f'AS_SEND_LAT=6 AS_NVLS_ENABLE=1 {self.simai_ns3_binary} -t 16 -w {workload_file} -n {topo} -c {conf}'
            result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=self.simai_dir)
            if result.returncode != 0:
                print(f'{command} failed. ret: {result.returncode}')
            
            # 
            # Move result file to identified filename
            original_result_file = f'{self.simai_dir}/ncclFlowModel_EndToEnd.csv'
            if os.path.exists(original_result_file):
                os.rename(original_result_file, result_file)
            else:
                print(f'Error: all_reduce_bytes: {all_reduce_bytes} not allowed by simai')
                # fallback to sklearn_execution_time_predictor
                return -1
        
        # allreduce latency
        # Get allreduce latency from result file
        with open(result_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = list(reader)
            # TODO: chentong fix this
            if len(rows) == 0: 
                print(f'Error: all_reduce_bytes: {all_reduce_bytes} not allowed by simai')
                # fallback to sklearn_execution_time_predictor
                return -1
            
            # total comm
            # simaius，vidurms
            # Second column of last row is total comm
            # simai returns us, vidur requires ms
            latency = float(rows[-1][1]) * 1e-3
            
            self.cache[cache_key] = latency
        
        return (self.cache[cache_key]
            # TODO: chentong whether we need these?
            # can these parameters be integreted into simai?
            + self.predictor_config.nccl_cpu_launch_overhead_ms
            + self.predictor_config.nccl_cpu_skew_overhead_per_device_ms
            * self.replica_config.tensor_parallel_size**1.25)
        
    # >   workload  command
    # > rewrite: add two features to reuse same workloads and results of same commands
    def get_execution_time_by_simai_analytical(self, batch: Batch):
        """
        Predict communication time using SimAI analytical tool.
        Args: batch: Batch object containing batch information to process.
        Returns: float: Predicted execution time (ms), returns -1 on error.

        SimAI
        Args: batch: Batch，
        Returns: float: （），-1
        """
        self.workload.flush()
        num_tokens_in_batch = batch._total_num_tokens_rounded
        all_reduce_bytes = self.hidden_size * num_tokens_in_batch * self.tensor_size
        
        # ，all_reduce_bytes
        # Use a tuple containing all relevant parameters as cache key instead of just all_reduce_bytes
        cache_key = (self.hidden_size, num_tokens_in_batch, self.tensor_size)
        
        # ，
        # If result is already in cache, return directly
        if cache_key in self.cache:
            return (self.cache[cache_key]
                + self.predictor_config.nccl_cpu_launch_overhead_ms
                + self.predictor_config.nccl_cpu_skew_overhead_per_device_ms
                * self.replica_config.tensor_parallel_size**1.25)
        
        
        # TODO(tianhao909): add layer0 str(1) "ALLREDUCE" etc. to hash computation, currently hardcoded
        #  workload 
        # Generate unique identifier for workload and command
        #  WorkItem 
        # Generate hash based on all relevant parameters of WorkItem
        # TODO(tianhao909):  layer0 str(1) "ALLREDUCE"  hash ，
            
        work_item_data = (
            "layer0" + 
            str(1) + 
            "ALLREDUCE" + 
            str(self.hidden_size) +
            str(num_tokens_in_batch) +
            str(self.tensor_size)
        )
        
        # Generate MD5 hash as workload identifier
        # MD5
        workload_identifier = hashlib.md5(work_item_data.encode()).hexdigest()
        
        
        # workload
        # Check if there is already a corresponding workload file
        workload_file = f'{self.workload_path}/allreduce_{workload_identifier}.txt'
        
        # workload，
        # Generate if workload file does not exist
        if not os.path.exists(workload_file):
            # layer
            # Assume each layer is homogeneous
            self.workload.append_work_item(
                WorkItem(
                    name="layer0",            # Layer name
                    forward_compute_time=1,    # Forward compute time
                    forward_comm="ALLREDUCE",  # Forward communication type
                    forward_comm_size=all_reduce_bytes  # Forward communication size (bytes)
                )
            )
            self.workload.dump_file(workload_file)
        

        # 
        # Get absolute paths for topology file and configuration file
        topo = os.path.abspath(self.predictor_config.simai_simulation_topo)
        conf = os.path.abspath(self.predictor_config.simai_simulation_config)
        
        # 
        # Check if result cache for corresponding command already exists
        command_identifier = hashlib.md5(
            f'{workload_identifier}_{topo}_{conf}'.encode()
        ).hexdigest()
        # result_file = f'{self.simai_dir}/ncclFlowModel_EndToEnd_{command_identifier}.csv'
        result_file = f'{self.simai_dir}/analytical_EndToEnd_{command_identifier}.csv'
        
        # ，
        # Run command if result file does not exist
        if not os.path.exists(result_file):
            # simai
            # Call simai
            # command = f'AS_SEND_LAT=6 AS_NVLS_ENABLE=1 {self.simai_ns3_binary} -t 16 -w {workload_file} -n {topo} -c {conf}'
            # ./bin/SimAI_analytical -w example/workload_analytical.txt -g 9216 -g_p_s 8 -r test- -busbw example/busbw.yaml
            
            cmd_g_p_s = 8          
            cmd_r= 'analytical_'    # Result file prefix
            cmd_busbw = f'{self.simai_dir}/example/busbw.yaml'
            # command = f'{self.simai_analytical_binary} -w {workload_file} -g {self.replica_config.world_size} -g_p_s {cmd_g_p_s} -r {cmd_r} -busbw {cmd_busbw}'
            command = f'{self.simai_analytical_binary} -w {workload_file} -g {self.replica_config.world_size} -g_p_s {cmd_g_p_s} -r {cmd_r}'
            result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=self.simai_dir)
            if result.returncode != 0:
                print(f'{command} failed. ret: {result.returncode}')
            
            # 
            # Move result file to identified filename
            # original_result_file = f'{self.simai_dir}/ncclFlowModel_EndToEnd.csv'
            # original_result_file = f'{self.simai_dir}/analytical_EndToEnd.csv'
            original_result_file = f'{self.simai_dir}/results/analytical_EndToEnd.csv'
            if os.path.exists(original_result_file):
                os.rename(original_result_file, result_file)
            else:
                print(f'Error: all_reduce_bytes: {all_reduce_bytes} not allowed by simai')
                # fallback to sklearn_execution_time_predictor
                return -1
        
        # allreduce latency
        # Get allreduce latency from result file
        with open(result_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = list(reader)
            # TODO: chentong fix this
            if len(rows) == 0: 
                print(f'Error: all_reduce_bytes: {all_reduce_bytes} not allowed by simai')
                # fallback to sklearn_execution_time_predictor
                return -1
            
            # total comm
            # simaius，vidurms
            # Second column of last row is total comm
            # simai returns us, vidur requires ms
            # latency = float(rows[-1][1]) * 1e-3
            
            # 5（），
            # Get latency data from index 5 position (microseconds), convert to milliseconds
            latency = float(rows[-1][5]) * 1e-3  
            
            # TODO(tianhao909): handle near-zero TP communication amounts (e.g. 65535 bytes -> 0 latency)
            # TODO(tianhao909):  TP  0， 65535 bytes  latency=0
            # When communication amount is 3276800, latency=0.015ms
            # assert all_reduce_bytes>0 and latency > 0, f"> Debug: all_reduce_bytes={all_reduce_bytes} latency={latency} need to be >=0"
            
            # ，
            # Store result in cache for future requests with same parameters
            self.cache[cache_key] = latency

        
        # ，
        # Return final predicted execution time, including cached latency and additional overhead
        return (self.cache[cache_key]
            # TODO: chentong whether we need these?
            # can these parameters be integreted into simai?
            + self.predictor_config.nccl_cpu_launch_overhead_ms
            + self.predictor_config.nccl_cpu_skew_overhead_per_device_ms
            * self.replica_config.tensor_parallel_size**1.25)
        
