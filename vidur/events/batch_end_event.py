from typing import List

from vidur.entities import Batch
from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

# >
from vidur.entities.request import RequestType
from vidur.entities.replica import ReplicaType


logger = init_logger(__name__)


# micro-batchpipeline
# A micro-batch execution ends in the pipeline
class BatchEndEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int, batch: Batch):
        super().__init__(time, EventType.BATCH_END)

        self._replica_id = replica_id
        self._batch = batch

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent

        # > batch
        # > batch completion triggers the next one
        self._batch.on_batch_end(self.time)
        replica_scheduler = scheduler.get_replica_scheduler(self._replica_id)
        replica_scheduler.on_batch_end(self._batch)

        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_batch_end(
            self.time, self._batch, self._replica_id, memory_usage_percent
        )

        
        
        logger.debug(f"time={self._time} Generates ReplicaScheduleEvent from event {self._id} {self._event_type}, "
            f"replica_id={self._replica_id}")
            
        # replicamicro-batchpipeline
        # replica continues to add the next micro-batch to the pipeline
        
        # ，Splitwise
        # Get global scheduler type to determine if it's Splitwise scheduling policy
        events = [ReplicaScheduleEvent(self.time, self._replica_id)]
        
        # >: vidur； pd； pd
        # >: Previous code was native vidur; PD separation is added processing; Without PD separation, it won't enter the following path
       
        # Check if Splitwise scheduling policy is used
        # TODO(tianhao909): test if non-PD separation works normally
        # TODO(tianhao909):  PD 
        if hasattr(scheduler, '__class__') and scheduler.__class__.__name__ == 'SplitwiseGlobalScheduler':
            # ，D
            # For each request in the batch, check if it needs to be transferred to D replica
            for request in self._batch.requests:
                #  fy： batch ： p batch； d batch； batch request type； 
                # fy: batch types: p batch; d batch; Determine all request types inside the batch from outside;
               
                # p batch；  d batch；  pd req batch
                # Determine if it's pure p batch; pure d batch; or mixed pd req batch
              
                # prefill
                # If the request has completed prefill stage
                if request.is_prefill_complete and request.request_type == RequestType.DECODE \
                    and replica_scheduler.replica.replica_type == ReplicaType.PREFILL:
                    # DECODE
                    # Modify request type to DECODE
                    # request.request_type = RequestType.DECODE

                    
                    # TODO(tianhao909): add P2P transmission bandwidth delay overhead here
                    # TODO(tianhao909):  P2P 
                    # transfer_delay = calculate_p2p_transfer_delay(request)
                    # request.decode_arrived_at += transfer_delay
                    # transfer_delay = 1 # > assumption
                    # transfer_delay = 10 # > assumption
                    
                    # request.pd_p2p_comm_size = request.estimate_kv_cache_size()
                    assert request.num_processed_tokens == request.num_prefill_tokens + 1, \
                        "processed tokens must equal prefill tokens + 1 at this point"
                    request.pd_p2p_comm_size = request.estimate_kv_cache_size( request.num_processed_tokens, replica_scheduler.replica)

                    # replica_scheduler.replica
                    # replica_scheduler.replica.
                    # transfer_delay = request.pd_p2p_comm_size / (request.bandwidth - request.bandwidth_used)
                    # transfer_delay = request.pd_p2p_comm_size / request.bandwidth
                    
                    # TODO(tianhao909): determine bandwidth from topology with contention modeling
                    # TODO(tianhao909): bandwidth  topo ，
                   
                    # request.pd_p2p_comm_bandwidth = 400*1024*1024*1024
                    request.pd_p2p_comm_bandwidth = replica_scheduler.replica.pd_p2p_comm_bandwidth*1024*1024*1024/8
                    assert request.pd_p2p_comm_size < float('inf') and request.pd_p2p_comm_size > 0 and request.pd_p2p_comm_bandwidth > 0, \
                        "P2P communication size and bandwidth must be valid"
                    request.pd_p2p_comm_time = request.pd_p2p_comm_size / request.pd_p2p_comm_bandwidth
                    
                    
                    # decodeprefill
                    # Set decode stage arrival time to prefill completion time
                    request.decode_arrived_at = request.prefill_completed_at + request.pd_p2p_comm_time
                    
                    # P
                    # Remove request from P replica
                    
                    # TODO(tianhao909): write small-token test cases for memory logic validation
                    # TODO(tianhao909):  req p  d  token ；
                  
                    # >  replica  req， 
                    # > risk: When replica clears requests, corresponding memory blocks should also be cleared
                    p_replica_scheduler = replica_scheduler
                    if request in p_replica_scheduler.replica.pending_requests:
                        # ，kvcache
                        # print(f">  {request.id} :")
                        # p_replica_scheduler.replica.get_remaining_kv_cache_capacity()
                        
                        # 
                        p_replica_scheduler.replica.pending_requests.remove(request)
                        
                        # 
                        # p_replica_scheduler.replica.release_request_kv_cache_memory(request)
                        # print(f">  {request.id} Prefill")
                        # p_replica_scheduler.replica.get_remaining_kv_cache_capacity()
                        
                    # TODO(tianhao909): ensure corresponding storage is also cleared
                    # TODO(tianhao909): 
                    
                    # D，D
                    # Add request to D replica, get corresponding D replica and add request
                  
                    d_replica_scheduler = scheduler.get_replica_scheduler(request.decode_replica_id)
                    # d_replica.pending_requests.append(request)
                    
                    # D
                    # Generate D replica scheduling event
                    events.append(ReplicaScheduleEvent(request.decode_arrived_at, request.decode_replica_id))
                    
                    logger.debug(f"pd d-path time={self._time} Generates ReplicaScheduleEvent from event {self._id} {self._event_type}, "
                        f"decode_replica_id={request.decode_replica_id} len(events)={len(events)}")
        
                
                if request._num_processed_tokens >= request._num_prefill_tokens:            
                    # print(f"> self.decode_arrived_at={self.decode_arrived_at} self.request_type={self.request_type} self.prefill_completed_at={self.prefill_completed_at} self._is_prefill_complete={self._is_prefill_complete}")
                    assert request.decode_arrived_at < float("inf") and request.request_type == RequestType.DECODE and request.prefill_completed_at > 0 and request._is_prefill_complete == True, \
                        "post-prefill request must have valid decode_arrived_at and be in DECODE state"

        # Call memory info logging function (disabled)
        # （）
        # self._log_memory_info(scheduler)
                        
        return events

    def _log_memory_info(self, scheduler: BaseGlobalScheduler) -> None:
        """
        Get and print memory capacity info for prefill and decode replicas.
        prefilldecode
        """
        # Get all replicas from scheduler
        # schedulerreplica
        # Use scheduler._replica_schedulers to get all replica IDs
        # scheduler_replica_schedulersID
        replica_ids = list(scheduler._replica_schedulers.keys())
        
        # Separate prefill and decode replica info
        # prefilldecode
        prefill_replica_info = {}
        decode_replica_info = {}
        
        for replica_id in replica_ids:
            replica_scheduler = scheduler.get_replica_scheduler(replica_id)
            replica = replica_scheduler.replica
            
            # Get TP and PP parameters / TPPP
            tensor_parallel_size = replica._replica_config.tensor_parallel_size
            pipeline_parallel_size = replica._replica_config.num_pipeline_stages
            
            # Create param_counter from replica_config
            # replica_configparam_counter
            param_counter = replica._replica_config._param_counter if hasattr(replica._replica_config, '_param_counter') else None
            if param_counter is None:
                # If replica has no _param_counter, create from replica_config
                # replica_param_counter，replica_config
                from vidur.utils.param_counter import ParamCounter
                param_counter = ParamCounter(replica._replica_config)
            
            # Get model params memory usage / 
            total_params = param_counter.get_num_parameters_per_device()
            # Convert bytes to GB / bytesGB
            total_params_gb = total_params / (1024**3)
            
            # Create memory_planner from replica_config and replica
            # replica_configreplicamemory_planner
            from vidur.scheduler.utils.memory_planner import MemoryPlanner
            memory_planner = MemoryPlanner(replica._replica_config, replica)
            
            # Get reserved KV cache memory / kvcache
            max_batch_size = memory_planner.get_max_batch_size()
            kv_cache_per_request = memory_planner._get_kv_cache_memory_per_device_per_request()
            memory_for_kv_cache = kv_cache_per_request * max_batch_size
            
            # Convert bytes to GB / bytesGB
            memory_for_kv_cache_gb = memory_for_kv_cache / (1024**3)
            
            # Get actual running requests' KV cache memory
            # Note: Replica has no running_requests attr, so we only count pending_requests
            # requestkvcache
            # ：Replicarunning_requests，pending_requests
            pending_requests_count = len(replica.pending_requests)
            actual_kv_cache_memory = kv_cache_per_request * pending_requests_count
            actual_kv_cache_memory_gb = actual_kv_cache_memory / (1024**3)
            
            # Compute whole-replica values (per-GPU * TP * PP)
            # replica（GPU × TP × PP）
            total_memory_replica_gb = replica.total_memory_gb * tensor_parallel_size * pipeline_parallel_size
            params_memory_replica_gb = total_params_gb * tensor_parallel_size * pipeline_parallel_size
            reserved_kv_cache_memory_replica_gb = memory_for_kv_cache_gb * tensor_parallel_size * pipeline_parallel_size
            actual_running_kv_cache_memory_replica_gb = actual_kv_cache_memory_gb * tensor_parallel_size * pipeline_parallel_size
            
            # Store info / 
            replica_info = {
                'total_memory_gb': replica.total_memory_gb,
                'total_memory_replica_gb': total_memory_replica_gb,
                'params_memory_gb': total_params_gb,
                'params_memory_replica_gb': params_memory_replica_gb,
                'reserved_kv_cache_memory_gb': memory_for_kv_cache_gb,
                'reserved_kv_cache_memory_replica_gb': reserved_kv_cache_memory_replica_gb,
                'actual_running_kv_cache_memory_gb': actual_kv_cache_memory_gb,
                'actual_running_kv_cache_memory_replica_gb': actual_running_kv_cache_memory_replica_gb,
                'active_requests_count': pending_requests_count,
                'max_batch_size': max_batch_size,
                'tp': tensor_parallel_size,
                'pp': pipeline_parallel_size
            }
            
            if replica.replica_type == ReplicaType.PREFILL:
                prefill_replica_info[replica_id] = replica_info
            elif replica.replica_type == ReplicaType.DECODE:
                decode_replica_info[replica_id] = replica_info
        
        #  | Print memory info
        logger.info("=" * 100)
        logger.info("Memory Usage Statistics (GB) ():")
        logger.info("-" * 100)
        
        if prefill_replica_info:
            logger.info("Prefill Replica Memory Info (Prefill):")
            for pid, info in prefill_replica_info.items():
                logger.info(f"  Replica ID {pid} (TP={info['tp']}, PP={info['pp']}):")
                logger.info(f"    Per-GPU total mem (GPU): {info['total_memory_gb']:.2f} GB ({info['total_memory_gb']*1024:.2f} MB)")
                logger.info(f"    Replica total mem (Replica): {info['total_memory_replica_gb']:.2f} GB ({info['total_memory_replica_gb']*1024:.2f} MB)")
                logger.info(f"    Per-GPU model params (GPU): {info['params_memory_gb']:.2f} GB ({info['params_memory_gb']*1024:.2f} MB)")
                logger.info(f"    Replica model params (Replica): {info['params_memory_replica_gb']:.2f} GB ({info['params_memory_replica_gb']*1024:.2f} MB)")
                logger.info(f"    Per-GPU reserved KV cache (GPUkvcache): {info['reserved_kv_cache_memory_gb']:.2f} GB ({info['reserved_kv_cache_memory_gb']*1024:.2f} MB)")
                logger.info(f"    Replica reserved KV cache (Replicakvcache): {info['reserved_kv_cache_memory_replica_gb']:.2f} GB ({info['reserved_kv_cache_memory_replica_gb']*1024:.2f} MB)")
                logger.info(f"    Per-GPU actual KV cache (GPUkvcache): {info['actual_running_kv_cache_memory_gb']:.2f} GB ({info['actual_running_kv_cache_memory_gb']*1024:.2f} MB)")
                logger.info(f"    Replica actual KV cache (Replicakvcache): {info['actual_running_kv_cache_memory_replica_gb']:.2f} GB ({info['actual_running_kv_cache_memory_replica_gb']*1024:.2f} MB)")
                logger.info(f"    Active requests (): {info['active_requests_count']}")
                logger.info(f"    Max batch size (): {info['max_batch_size']}")
            logger.info("-" * 100)
        
        if decode_replica_info:
            logger.info("Decode Replica Memory Info (Decode):")
            for did, info in decode_replica_info.items():
                logger.info(f"  Replica ID {did} (TP={info['tp']}, PP={info['pp']}):")
                logger.info(f"    Per-GPU total mem (GPU): {info['total_memory_gb']:.2f} GB ({info['total_memory_gb']*1024:.2f} MB)")
                logger.info(f"    Replica total mem (Replica): {info['total_memory_replica_gb']:.2f} GB ({info['total_memory_replica_gb']*1024:.2f} MB)")
                logger.info(f"    Per-GPU model params (GPU): {info['params_memory_gb']:.2f} GB ({info['params_memory_gb']*1024:.2f} MB)")
                logger.info(f"    Replica model params (Replica): {info['params_memory_replica_gb']:.2f} GB ({info['params_memory_replica_gb']*1024:.2f} MB)")
                logger.info(f"    Per-GPU reserved KV cache (GPUkvcache): {info['reserved_kv_cache_memory_gb']:.2f} GB ({info['reserved_kv_cache_memory_gb']*1024:.2f} MB)")
                logger.info(f"    Replica reserved KV cache (Replicakvcache): {info['reserved_kv_cache_memory_replica_gb']:.2f} GB ({info['reserved_kv_cache_memory_replica_gb']*1024:.2f} MB)")
                logger.info(f"    Per-GPU actual KV cache (GPUkvcache): {info['actual_running_kv_cache_memory_gb']:.2f} GB ({info['actual_running_kv_cache_memory_gb']*1024:.2f} MB)")
                logger.info(f"    Replica actual KV cache (Replicakvcache): {info['actual_running_kv_cache_memory_replica_gb']:.2f} GB ({info['actual_running_kv_cache_memory_replica_gb']*1024:.2f} MB)")
                logger.info(f"    Active requests (): {info['active_requests_count']}")
                logger.info(f"    Max batch size (): {info['max_batch_size']}")
            logger.info("-" * 100)
        logger.info("=" * 100)


    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "event_type": self.event_type,
            "batch_id": self._batch.id,
        }