from typing import List, Tuple, Dict 

from vidur.logger import init_logger
from vidur.entities import Replica, Request  
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler 

from abc import ABC, abstractmethod  
from typing import Dict, List, Tuple  

from vidur.config import SimulationConfig  
from vidur.entities import Replica, Request  
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry 
from vidur.scheduler.replica_scheduler.replica_scheduler_registry import (  
    ReplicaSchedulerRegistry,
)

logger = init_logger(__name__)

# >
from vidur.entities.task import Task, TaskType
from vidur.entities.flow import Flow, FlowType
from vidur.entities.interconnect import DummyLink
from vidur.entities.replica import Replica, ReplicaType
from vidur.entities.request import Request, RequestType

# TODO(tianhao909): rename class - not strictly Splitwise anymore, more like PD-separation scheduler
# TODO(tianhao909): ， Splitwise， PD 
class SplitwiseGlobalScheduler(BaseGlobalScheduler):  # Splitwise Global Scheduler.
    def __init__(self, config: SimulationConfig, replicas: Dict[int, Replica]):
        # Call parent class initialization method
        super().__init__(config, replicas)
        
        self._config = config  # Save configuration object as instance private attribute
        # self._config = SimulationConfig
        self._replicas = replicas  # Save replica dictionary as instance private attribute, key is replica ID, value is replica object
        self._num_replicas = len(self._replicas)  # Calculate and save total number of replicas
        
        # TODO(tianhao909): make pd_node_ratio configurable
        # TODO(tianhao909):  pd_node_ratio 
        # self.pd_node_ratio = 0.5 
        self.pd_node_ratio =  self._replicas[0].pd_node_ratio
        assert self.pd_node_ratio >= 0 and self.pd_node_ratio <= 1, "pd_node_ratio must be between 0 and 1"
        # self._sub_scheduler = self._config.splitwise_scheduler_sub_scheduler  # Get sub-scheduler type from configuration
        # TODO(tianhao909): make _sub_scheduler configurable
        # TODO(tianhao909):  _sub_scheduler 
        # self._sub_scheduler = "round_robin"
        self._sub_scheduler = "lor"
        
        assert self._sub_scheduler != "splitwise"  # Assert sub-scheduler type cannot be "splitwise" to avoid circular dependency

        self._num_prefill_nodes = self._num_replicas * self.pd_node_ratio  # Calculate number of prefill nodes
        self._num_decode_nodes = self._num_replicas - self._num_prefill_nodes  # Calculate number of decode nodes (total nodes - prefill nodes)

        assert self._num_prefill_nodes > 0  # Assert number of prefill nodes must be greater than 0
        assert self._num_decode_nodes > 0  # Assert number of decode nodes must be greater than 0
        
        # print(f"> Debug: self._request_queue={self._request_queue}")
        # SimulationConfig
        
        # ， 4 ， 0.5：
        # _num_prefill_nodes = 2
        # _num_decode_nodes = 2
        #  2  (ID 0, 1)  （ _prefill_replicas ）
        #  2  (ID 2, 3)  （ _decode_replicas ）
        
        # For example, if there are 4 replicas with ratio 0.5:
        # _num_prefill_nodes = 2
        # _num_decode_nodes = 2
        # First 2 replicas (ID 0, 1) are dedicated to prefill tasks (_prefill_replicas)
        # Last 2 replicas (ID 2, 3) are dedicated to decode tasks (_decode_replicas)
        
        # ，ID
        # Create prefill replica dictionary, filter replicas with ID less than number of prefill nodes
        self.prefill_replicas = {}
        for replica_id, replica in self._replicas.items():
            if replica_id < self._num_prefill_nodes:
                self.prefill_replicas[replica_id] = replica
                replica.replica_type = ReplicaType.PREFILL

        # ，ID
        # Create decode replica dictionary, filter replicas with ID greater than or equal to number of prefill nodes
        self.decode_replicas = {}
        for replica_id, replica in self._replicas.items():
            if replica_id >= self._num_prefill_nodes:
                self.decode_replicas[replica_id] = replica
                replica.replica_type = ReplicaType.DECODE

        # ； mix pool
        # Remove registration; replace with mix pool
        self.prefill_scheduler = self.get_global_scheduler(self.prefill_replicas)  # Create prefill scheduler
        self.decode_scheduler = self.get_global_scheduler(self.decode_replicas)  # Create decode scheduler

        # , vidurreplicasinstances
        # Instance related, vidur uses replicas instead of instances
        # self.instances = []  # Track all available instances 
        self.replicas = [] 

        # 
        # Request queues
        self.pending_queue = []  # Pending request queue
        self.executing_queue = []  # Executing request queue
        self.completed_queue = []  # Completed request queue

        # sw  ,  ， vidurvidurexecutor
        # sw executor related, may not be added, vidur has its own executor
        
        # self.executor_type = ExecutorType.CentralExecutor  # Default executor type
        # self.executors = {}  # Store executors with request ID as key
        
        # sw scheduler operation logger
        # logger_name = f"schedulers/{self.application.application_id}"  # Logger name
        # level = logging.DEBUG if self.debug else logging.INFO  # Set log level based on debug mode
        # os.makedirs("schedulers", exist_ok=True)  # Create log directory (if not exists)
        # self.scheduler_logger = utils.file_logger(logger_name, level=level)  # Create file logger
        # self.scheduler_logger.info("time,action,info")  # Write log header
        
        # class KVScheduler(Scheduler):init
        # self.prompt_processors = prompt_processors  # Processors for handling prompts
        # self.token_processors = token_processors  # Processors for handling tokens
        
        # gpu pd node
        # GPU classification into p and d, consider whether to change to node granularity
        
        # self.prompt_processors = []
        # self.token_processors = []
        self.prefil_gpus = []
        self.decode_gpus = []
        # self.prompt_instances = []  # List of instances for handling prompts
        # self.token_instances = []  # List of instances for handling tokens
        
        
        # class MixedPoolScheduler(KVScheduler): init
        
        # self.prompt_max_pending_batch_tokens = prompt_max_pending_batch_tokens  # Maximum pending tokens for prompt batching
        # self.token_max_pending_batch_tokens = token_max_pending_batch_tokens  # Maximum pending tokens for token batching
        # self.transfer_bandwidth = transfer_bandwidth * 1024**3 # Convert to B/s (bytes/second)
        
        # > 0， replica ， p replica  d replica
        # > initialize to 0, wait for replica classification into p replica and d replica
        self.prompt_max_pending_batch_tokens = 0
        self.token_max_pending_batch_tokens = 0

        # self.prompt_replicas = []
        # self.mixed_replicas = []
        # self.token_replicas = []
        # self.prefill_replicas = []
        self.mixed_replicas = []
        # self.decode_replicas = []
        
        
        # self.prompt_instances = []  # Prompt instance list
        # self.mixed_instances = []  # Mixed instance list (can handle prompts and tokens)
        # self.token_instances = []  # Token instance list
        
        # TODO(tianhao909): add transfer_bandwidth to config input
        # TODO(tianhao909): ， config 
        self.transfer_bandwidth = 0
        self.transfer_bandwidth = 200 * 1024**3 # Gbpsbps
        
        self.p_request_counter = 0
        self.d_request_counter = 0

        
    def get_global_scheduler(self, replicas: Dict[int, Replica]):  # Used to get global scheduler for specified replica set
        from vidur.scheduler.global_scheduler.global_scheduler_registry import GlobalSchedulerRegistry  # Import global scheduler registry
        # Get corresponding scheduler instance from registry based on sub-scheduler type and return
        
        # 1. Input: Receive a string parameter key_str (e.g. "random", "round_robin", "lor", etc.)
        # 2. Processing: Call GlobalSchedulerType.from_str(key_str) to convert string to corresponding enum value
        # 3. Output: Return corresponding GlobalSchedulerType enum value
        return GlobalSchedulerRegistry.get_from_str(self._sub_scheduler, self._config, replicas)
        # return GlobalSchedulerRegistry.get_from_str()
    
    def is_queue_long(self, replica: Replica, task: Task):
        """
        
        Check if prompt queue is too long
        """
        
        # token, 
        # If waiting token count exceeds maximum limit, queue is too long
        if len(replica.pending_queue) > 0 and \
            replica.sched_pending_tokens + task.tokens_per_iteration > self.prefill_max_pending_batch_tokens:
            return True
        else:
            return False
        
    def is_memory_loaded(self, replica: Replica, tasks):
        """
        （）
        Check if replica is loaded by task
        """
        # 
        # Calculate total memory required by tasks
        request_memory = sum(task.max_memory(replica) for task in tasks)
        #   ;  
        # If total memory exceeds instance maximum memory, memory is insufficient; otherwise memory is sufficient
        if replica.sched_memory + request_memory >= replica.max_memory:
            return True
        else:
            return False
    
    def find_best_prefill_replica(self, prefill_replicas: Dict[int, Replica], prefill_task: Task):
        """
        ，
        Check if prompt queue is too long, find best prompt instance
        """
        if len(prefill_replicas) == 0:
            return None 
        # replica.max_request_tokens
        prefill_replica = min(prefill_replicas,
                              key=lambda replica: replica.sched_pending_tokens)
        if self.is_queue_long(prefill_replica, prefill_task):
            return None
        return prefill_replica
        
        # replica_id = min(pending_requests_map.items(), key=lambda x: x[1])[0]  # Find replica ID with minimum pending requests
        # pending_requests_map[replica_id] += 1  # Increment pending requests count for that replica
        # request_mapping.append((replica_id, request))  # Add replica ID and request to mapping result
        
    def find_best_decode_replica(self, decode_replicas: Dict[int, Replica], prefill_task: Task, decode_task: Task):
        """
        ，token
        Check if instance memory is full, find best token instance
        """
        if len(decode_replicas) == 0:
            return None
        decode_replica = min(decode_replicas,
                              key=lambda replica: replica.sched_pending_tokens)
        if self.is_memory_loaded(decode_replica,[prefill_task, decode_task]):
            return None
        return decode_replica 
    
    def add_kv_cache_transfer(self, request, src_replica, dest_replica, bandwidth):
        """
        ，prompt->tokenprompt->kvtransfer->token。
        Add flow node to request graph to convert prompt->token request to prompt->kvtransfer->token request.
        """
        prefill_task = request.root_node  # Get root node of request (prompt task)
        decode_task = next(request.successors(prefill_task))  # Get next task of prompt task (token task)

        # 
        # Create new tasks and flows

        flow_size = request.estimate_kv_cache_size(  # Estimate KV cache size
                                        num_tokens=prefill_task.prompt_size,
                                        replica=src_replica)
        
        kv_transfer_flow = request.create_flow(FlowType.KVCacheTransfer,  # Create KV cache transfer flow
                                               size=flow_size,
                                               src=src_replica,
                                               dest=dest_replica)
        kv_transfer_flow.notify = True  #  Enable transfer completion notification

        # Update request's directed acyclic graph (DAG)
        request.flow_node = kv_transfer_flow  # Record flow node
        request.dag.remove_edge(prefill_task, decode_task)  # Remove edge from prompt task to token task
        request.dag.add_edge(prefill_task, kv_transfer_flow)  # Add edge from prompt task to KV transfer flow
        request.dag.add_edge(kv_transfer_flow, decode_task)  # Add edge from KV transfer flow to token task

        # 
        # Assign tasks and flows to instances and links
        prefill_task.instance = src_replica  # Assign prompt task to source instance
        decode_task.instance = dest_replica  # Assign token task to destination instance
        # Note: Simulate latency by adding configurable bandwidth link
        kv_transfer_flow.link = DummyLink(name="DummyLink",  # Create virtual link
                                          bandwidth=bandwidth)

        
        
    
    def schedule(self) -> List[Tuple[int, Request]]:  # Execute scheduling logic method, return scheduling result list
        self.sort_requests()  # Sort request queue
        request_mapping = []  # Store request mapping results
        
        # map  p d arrive time； 
        # Add p and d arrival times when mapping
        prefill_request_mapping = []
        decode_request_mapping = []
        
        # Each ReplicaScheduler object has the following attributes:
        # replica_id: Unique identifier of the replica
        # num_pending_requests: Current number of pending requests
        # Create pending request count mapping table
        # Record replica ID and pending request count for each replica scheduler
        # Iterate through all replica schedulers

        # 1  pending_requests_map
        # 1 First create an empty dictionary pending_requests_map
        pending_requests_map = {}
        
        # 2  self._replica_schedulers.values()  replica_scheduler
        # 2 Iterate through each replica_scheduler in self._replica_schedulers.values()
        
        # > ； index ；
        # > No batch arrival; use index to record last arrival
        for replica_scheduler in self._replica_schedulers.values():
            replica_id = replica_scheduler.replica_id
            num_pending_requests = replica_scheduler.num_pending_requests
            pending_requests_map[replica_id] = num_pending_requests
        
        # At this point: pending_requests_map={0: 0, 1: 0}
        
        
        #  pending_prefill_requests_map pending_prefill_requests_map；
        # Initialize pending_prefill_requests_map, assign values based on current state
        pending_prefill_requests_map = {}
        pending_prefill_requests_map = {}
        for prefill_replica_id, prefill_replica in self.prefill_replicas.items():
            num_pending_requests = len(prefill_replica.pending_tasks)
            pending_prefill_requests_map[prefill_replica_id] = num_pending_requests
        # pending_prefill_requests_map={0: 0, 1: 0}
        
        
        #  pending_decode_requests_map； pending_decode_requests_map；
        # Initialize pending_decode_requests_map, assign values based on current state
        pending_decode_requests_map = {}
        for decode_replica_id, decode_replica in self.decode_replicas.items():
            num_pending_requests = len(decode_replica.pending_tasks)
            pending_decode_requests_map[decode_replica_id] = num_pending_requests
        # pending_decode_requests_map={2: 0, 3: 0}
        
        # print(f"> Debug: pending_requests_map: {pending_requests_map} \
        #         pending_prefill_requests_map: {pending_prefill_requests_map} \
        #         pending_decode_requests_map: {pending_decode_requests_map}")
        
        
        # self._request_queuerequestprefill_task  decode_task dag
        # Create prefill_task and decode_task for each request in self._request_queue, create dag graph
        
        
        requests_to_remove = list()
        for request in self._request_queue:
            # >
            requests_to_remove.append(request)
            #  
            # For convenient retrieval 
            request.global_scheduler = self
            
            # print(f"> Debug: request:request.prefill_arrived_at {request.prefill_arrived_at} request.decode_arrived_at  {request.decode_arrived_at}")
            
            prefill_task = request.create_task(task_type=TaskType.PROMPT,prompt_size=request.num_prefill_tokens)
            decode_task = request.create_task(task_type=TaskType.TOKEN,token_size=request.num_decode_tokens - 1)
            # update DAG
            request.dag.add_edge(prefill_task, decode_task)
            request.root_node = prefill_task
            
            prefill_replica = None

            # fth p req  p replica lor ：
            # vidur's lor:
            # replica_id = min(pending_prefill_requests_map.items(), key=lambda x: x[1])[0]
            
            # vidur's round-robin
            replica_id = self.p_request_counter % len(self.prefill_replicas)
            self.p_request_counter += 1
            
            prefill_replica = self.prefill_replicas[replica_id]
            request.prefill_replica_id = replica_id
            prefill_request_mapping.append((replica_id,request))
            
            decode_replica = None
            
            
            # vidurlor；  ；
            # vidur's lor; find shortest, might not find the shortest
            # replica_id = min(pending_decode_requests_map.items(), key=lambda x: x[1])[0]
            
            # vidur's round-robin
            replica_id = (self.d_request_counter % len(self.decode_replicas)) + len(self.prefill_replicas)
            self.d_request_counter += 1
            
            decode_replica = self.decode_replicas[replica_id]
            request.decode_replica_id = replica_id
            decode_request_mapping.append((decode_replica.id, request))
            
            # TODO(tianhao909): remove unused task DAG code (low priority)
            # TODO(tianhao909):  task DAG （）
            # task inherits from req; or let task construct req
            # task  req； task  req
            
            '''
            if prefill_replica != decode_replica:  # If prompt instance and token instance are different
                # ============================================================
                # [] add_to_pool + DAG + sched_* 
                # 
                #  Splitwise PD :
                # 1. add_to_pool(): requestreplica.pending_requests
                #    - prefill: batch_end_event.py remove，
                #    - decode: ，
                # 2. request DAG (prefill_task, decode_task): DAG，
                #    _get_next_batch()_request_queue，DAG
                # 3. add_kv_cache_transfer(): flow node，
                # 4. sched_memory/sched_pending_tokens: 
                #
                # ，。
                # ============================================================
                logger.debug(f"[Redundant code check ()] schedule(): "
                      f"req={request.id}, p_replica={prefill_replica.id}, d_replica={decode_replica.id}")
                logger.debug(f"  add_to_pool(prefill_task): added to p_replica.pending_requests "
                      f"( p_replica.pending_requests, len={len(prefill_replica.pending_requests)})")
                logger.debug(f"  add_to_pool(decode_task): added to d_replica.pending_requests "
                      f"( d_replica.pending_requests, len={len(decode_replica.pending_requests)}) [redundant, ]")
                prefill_replica.add_to_pool(prefill_task)  
                # decode_replica.add_to_pool(decode_task)  
                decode_replica.add_to_pool(decode_task)  # [] decodepending_requests  
                
                # [] KV cache transfer/DAG
                # KV
                # Transfer KV cache between instances
                self.add_kv_cache_transfer(request,
                                        prefill_replica,
                                        decode_replica,
                                        self.transfer_bandwidth)
                
                # [] sched_memory 
                prefill_replica.sched_memory += prefill_task.max_memory(prefill_replica)  # Update prompt instance memory usage
                decode_replica.sched_memory += prefill_task.max_memory(decode_replica) + \
                                            decode_task.max_memory(decode_replica)  # Update token instance memory usage
            else:
                # Run on same instance
                prefill_task.instance = prefill_replica  # Assign prompt task to this instance
                decode_task.instance = decode_replica  # Assign token task to this instance
                prefill_replica.sched_memory += prefill_task.max_memory(prefill_replica) + \
                                                decode_task.max_memory(prefill_replica)  # Update instance memory usage
                prefill_task.chain = [decode_task]  # Set token task as successor of prompt task
            
            # [] sched_pending_tokens 
            prefill_replica.sched_pending_tokens += prefill_task.prompt_size  # Update prompt instance pending token count
            decode_replica.sched_pending_tokens += 1  # Update token instance pending token count
            '''
        # >
        for req in requests_to_remove:
            self._request_queue.remove(req)
        
        # print(f"> Debug: pending_requests_map: {pending_requests_map} \
        #         pending_prefill_requests_map: {pending_prefill_requests_map} \
        #         pending_decode_requests_map: {pending_decode_requests_map}")
        
        # print(f"> Debug: prefill_request_mapping: {prefill_request_mapping} decode_request_mapping: {decode_request_mapping}")
        
        # > ，request； request replica
        # > Currently dynamically added, stored in request; request knows which replica to route to
        return prefill_request_mapping
        
