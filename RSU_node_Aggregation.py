import copy
import torch
from utils.aggregation_strategy import once_state_dicts, fed_avg,fed_adam,fed_avgm,fed_nova,fed_prox, fed_adagrad, fed_moon

class RSUServer:
    def __init__(self, rsu_id, agg_strategy_name, covered_clients=None, position=None):
        self.rsu_id = rsu_id
        self.covered_clients = covered_clients if covered_clients else []
        self.position = position
        self.head_updates = []
        self.tail_updates = []

        # FedNova 所需参数
        self.client_sample_nums = []
        self.client_local_steps = []

        # FedAdaGrad 所需参数
        self.head_accumulator = {}
        self.tail_accumulator = {}

        self.aggregated_head = None
        self.aggregated_tail = None
        self.global_head = None
        self.global_tail = None
        self.connected_clients = []
        self.agg_strategy_name = agg_strategy_name.lower()
        self.set_aggregation_strategy(self.agg_strategy_name)

    def receive_update(self, head_state_dict=None, tail_state_dict=None, sample_num=None, local_steps=None):
        """
        接收来自客户端的 head 和 tail 模型参数
        """
        if head_state_dict and "head" not in head_state_dict:
            raise ValueError("head_state_dict 应该包含 'head' 键")
        if tail_state_dict and "tail" not in tail_state_dict:
            raise ValueError("tail_state_dict 应该包含 'tail' 键")

        if head_state_dict:
            self.head_updates.append(head_state_dict["head"])
        if tail_state_dict:
            self.tail_updates.append(tail_state_dict["tail"])

        if sample_num is not None:
            self.client_sample_nums.append(sample_num)
        if local_steps is not None:
            self.client_local_steps.append(local_steps)

    def bind_vehicle(self, vehicle_id):
        self.connected_clients.append(vehicle_id)

    """
       def receive_update(self, head_state_dict=None, tail_state_dict=None):
       
        接收来自客户端的 head 和 tail 模型参数
        
        if head_state_dict and "head" not in head_state_dict:
            raise ValueError("head_state_dict 应该包含 'head' 键")
        if tail_state_dict and "tail" not in tail_state_dict:
            raise ValueError("tail_state_dict 应该包含 'tail' 键")

        if head_state_dict:
            self.head_updates.append(head_state_dict)
        if tail_state_dict:
            self.tail_updates.append(tail_state_dict)

    """

    def set_aggregation_strategy(self, strategy_name):
        if strategy_name == "fed_avg":
            self.agg_fn = fed_avg
        elif strategy_name == "fed_prox":
            self.agg_fn = fed_prox
        elif strategy_name == "fed_avgm":
            self.agg_fn = fed_avgm
        elif strategy_name == "fed_nova":
            self.agg_fn = fed_nova
        elif strategy_name == "fed_adam":
            self.agg_fn = fed_adam
        elif strategy_name == "fed_adagrad":
            self.agg_fn = fed_adagrad
        elif strategy_name == "fed_moon":
            self.agg_fn = fed_moon
        else:
            raise ValueError(f"未知聚合策略：{strategy_name}")

    def _init_accumulator_if_needed(self, accumulators, reference_params):
        for k in reference_params:
            if k not in accumulators:
                accumulators[k] = torch.zeros_like(reference_params[k])

    def local_aggregate(self, round_num=0, quantizer_name=None, **kwargs):

        if not self.head_updates or not self.tail_updates:
            print(f"[{self.head_updates}] 为空")
            return None

        """
        对 head 和 tail 模型分别进行聚合
        """
        if len(self.head_updates) == 1 and len(self.tail_updates) == 1:
            self.aggregated_head = self.head_updates[0]
            self.aggregated_tail = self.tail_updates[0]
            print(f"[{self.rsu_id}] ⚠️ 仅 1 个客户端，跳过聚合，直接返回上传模型")
            return {
                "head": self.aggregated_head,
                "tail": self.aggregated_tail
            }
        # FedNova 处理逻辑
        if self.agg_strategy_name == "fed_nova":
            self.aggregated_head = fed_nova(
                self.head_updates,
                sample_nums=self.client_sample_nums,
                local_steps=self.client_local_steps
            )
            self.aggregated_tail = fed_nova(
                self.tail_updates,
                sample_nums=self.client_sample_nums,
                local_steps=self.client_local_steps
            )
            print(f"[{self.rsu_id}] ✅ FedNova 聚合完成：clients = {len(self.head_updates)}")
            return {
                "head": self.aggregated_head,
                "tail": self.aggregated_tail
            }

        #  FedAdagrad 处理逻辑
        if self.agg_strategy_name == "fed_adagrad":
            self._init_accumulator_if_needed(self.head_accumulator, kwargs["prev_params"]["head"])
            self._init_accumulator_if_needed(self.tail_accumulator, kwargs["prev_params"]["tail"])

            self.aggregated_head, self.head_accumulator = self.agg_fn(
                self.head_updates,
                prev_params=kwargs["prev_params"]["head"],
                accumulators=self.head_accumulator,
                lr=kwargs.get("lr", 0.01),
                epsilon=kwargs.get("epsilon", 1e-8)
            )
            self.aggregated_tail, self.tail_accumulator = self.agg_fn(
                self.tail_updates,
                prev_params=kwargs["prev_params"]["tail"],
                accumulators=self.tail_accumulator,
                lr=kwargs.get("lr", 0.01),
                epsilon=kwargs.get("epsilon", 1e-8)
            )
            print(f"[{self.rsu_id}] ✅ FedAdagrad 聚合完成")
            return {
                "head": self.aggregated_head,
                "tail": self.aggregated_tail
            }
        # FedMOON 聚合逻辑（本质上等价于 FedAvg）
        if self.agg_strategy_name == "fed_moon":
            self.aggregated_head = fed_moon(self.head_updates)
            self.aggregated_tail = fed_moon(self.tail_updates)
            print(f"[{self.rsu_id}] ✅ FedMOON 聚合完成（本质 FedAvg）：clients = {len(self.head_updates)}")
            return {
                "head": self.aggregated_head,
                "tail": self.aggregated_tail
            }

        # FedAdam / FedAvgM / FedAvg 等策略
        if self.head_updates:
            head_states = self.head_updates
            if self.agg_strategy_name == "fed_adam":
                self.aggregated_head, kwargs["momentums"]["head"], kwargs["velocities"]["head"] = self.agg_fn(
                    head_states,
                    prev_params=kwargs["prev_params"]["head"],
                    momentums=kwargs["momentums"]["head"],
                    velocities=kwargs["velocities"]["head"],
                    round_num=round_num,
                    lr=kwargs.get("lr", 0.001),
                    beta1=kwargs.get("beta1", 0.9),
                    beta2=kwargs.get("beta2", 0.999)
                )
            elif self.agg_strategy_name == "fed_avgm":
                self.aggregated_head, kwargs["momentums"]["head"] = self.agg_fn(
                    head_states,
                    prev_params=kwargs["prev_params"]["head"],
                    momentum=kwargs["momentums"]["head"],
                    beta=kwargs.get("beta", 0.9),
                    eta=kwargs.get("eta", 1.0)
                )
            else:
                #self.aggregated_head = self.agg_fn(head_states, sample_nums=self.client_sample_nums)
                if quantizer_name.lower() == "signsgd":
                    print(f"[{self.rsu_id}] ⚠️ SignSGD 模式下跳过 sample_nums，加权平均改为简单平均")
                    self.aggregated_head = self.agg_fn(head_states)
                else:
                    self.aggregated_head = self.agg_fn(head_states, sample_nums=self.client_sample_nums)
        if self.tail_updates:
            tail_states = self.tail_updates
            # tail_states = [update["tail"] for update in self.tail_updates]
            if self.agg_strategy_name == "fed_adam":
                self.aggregated_tail, kwargs["momentums"]["tail"], kwargs["velocities"]["tail"] = self.agg_fn(
                    tail_states,
                    prev_params=kwargs["prev_params"]["tail"],
                    momentums=kwargs["momentums"]["tail"],
                    velocities=kwargs["velocities"]["tail"],
                    round_num=round_num,
                    lr=kwargs.get("lr", 0.001),
                    beta1=kwargs.get("beta1", 0.9),
                    beta2=kwargs.get("beta2", 0.999)
                )
            elif self.agg_strategy_name == "fed_avgm":
                self.aggregated_tail, kwargs["momentums"]["tail"] = self.agg_fn(
                    tail_states,
                    prev_params=kwargs["prev_params"]["tail"],
                    momentum=kwargs["momentums"]["tail"],
                    beta=kwargs.get("beta", 0.9),
                    eta=kwargs.get("eta", 1.0)
                )
            else:
                #self.aggregated_tail = self.agg_fn(tail_states, sample_nums=self.client_sample_nums,round_num=round_num, **kwargs)
                if quantizer_name and quantizer_name.lower() == "signsgd":
                    self.aggregated_tail = self.agg_fn(tail_states)
                else:
                    self.aggregated_tail = self.agg_fn(tail_states, sample_nums=self.client_sample_nums,
                                                       round_num=round_num, **kwargs)

        print(f"[{self.rsu_id}] ✅ 聚合完成：head({len(self.head_updates)}), tail({len(self.tail_updates)})")
        return {
            "head": self.aggregated_head,
            "tail": self.aggregated_tail
        }

    def distribute_model(self, global_model_dict):
        """
        下发 global_head 和 global_tail 到车辆
        """
        self.global_head = global_model_dict.get("head", None)
        self.global_tail = global_model_dict.get("tail", None)

    def reset(self):
        self.head_updates = []
        self.tail_updates = []
        self.aggregated_head = None
        self.aggregated_tail = None
        self.connected_clients = []

        # ✅ 清空 FedNova 需要的状态
        self.client_sample_nums = []
        self.client_local_steps = []
