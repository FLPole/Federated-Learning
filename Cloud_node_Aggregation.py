# cloud_node.py


"""
import torch
import copy

class CloudServer:
    def __init__(self, head_template: torch.nn.Module, tail_template: torch.nn.Module, body_model: torch.nn.Module):
        self.global_head = copy.deepcopy(head_template)
        self.global_tail = copy.deepcopy(tail_template)
        self.body_model = body_model  # 云端固定的主体模型，不参与分发与聚合

    def distribute_model(self):

        #返回当前的 head 和 tail 参数（供客户端初始化或更新）

        return {
            "head": copy.deepcopy(self.global_head.state_dict()),
            "tail": copy.deepcopy(self.global_tail.state_dict())
        }

    def aggregate(self, head_updates, tail_updates):

        #聚合来自客户端上传的 head 和 tail 模型更新（参数平均）


        def average_state_dicts(updates):
            device = updates[0][next(iter(updates[0]))].device  # 获取第一个参数的设备
            avg = {k: v.float().to(device).clone() for k, v in updates[0].items()}
            for upd in updates[1:]:
                for k in avg:
                    avg[k] += upd[k].float().to(device)
            for k in avg:
                avg[k] /= len(updates)
            return avg

        def is_valid_state_dict(state_dict):
            for k, v in state_dict.items():
                if torch.isnan(v).any() or torch.isinf(v).any():
                    print(f"[WARN] 上传参数异常: {k} nan/inf 检查失败")
                    return False
            return True

        # ✅ 过滤掉无效上传
        head_updates = [h for h in head_updates if h is not None and is_valid_state_dict(h)]
        tail_updates = [t for t in tail_updates if t is not None and is_valid_state_dict(t)]

        if not head_updates:
            print("[WARN] Cloud 聚合失败：无有效 head 参数上传")
            return

        if not tail_updates:
            print("[WARN] Cloud 聚合警告：无有效 tail 参数上传，仅聚合 head")


        if head_updates:
            avg_head = average_state_dicts(head_updates)
            self.global_head.load_state_dict(avg_head)

        if tail_updates:
            avg_tail = average_state_dicts(tail_updates)
            self.global_tail.load_state_dict(avg_tail)

        print(f"✅ 成功聚合 head({len(head_updates)}) 与 tail({len(tail_updates)}) 参数")


"""

import torch
import copy

class CloudServer:
    def __init__(self, head_template: torch.nn.Module, tail_template: torch.nn.Module, body_model: torch.nn.Module):
        self.global_head = copy.deepcopy(head_template)
        self.global_tail = copy.deepcopy(tail_template)
        self.body_model = body_model  # 云端固定的主体模型，不参与分发与聚合

    def distribute_model(self):
        """
        返回当前的 head 和 tail 参数（供客户端初始化或更新）
        """
        return {
            "head": copy.deepcopy(self.global_head.state_dict()),
            "tail": copy.deepcopy(self.global_tail.state_dict())
        }

    def aggregate(self, head_updates, tail_updates, quantizer_name=None):
        """
        聚合来自 RSU 上传的 head 和 tail 模型参数（均值聚合）
        """
        # ========= 辅助函数 =========
        def average_state_dicts(updates):
            # 参数校验
            assert len(updates) > 0, "average_state_dicts: 输入为空！"
            # 取第一个 state_dict 的设备和结构
            device = next(iter(updates[0].values())).device
            keys = updates[0].keys()
            avg = {k: v.float().to(device).clone() for k, v in updates[0].items()}
            for upd in updates[1:]:
                for k in keys:
                    avg[k] += upd[k].float().to(device)
            for k in keys:
                avg[k] /= len(updates)
            return avg

        def is_valid_state_dict(state_dict):
            for k, v in state_dict.items():
                nan_cnt = torch.isnan(v).sum().item()
                inf_cnt = torch.isinf(v).sum().item()
                if nan_cnt > 0 or inf_cnt > 0:
                    print(f"[WARN] 上传参数异常: {k} nan={nan_cnt}, inf={inf_cnt}")
                    return False
            return True

        # ========== Debug：上传信息概览 ==========
        print(f"[Cloud] 聚合前，收到 head={len(head_updates)}，tail={len(tail_updates)} 条上传")
        # 过滤掉 None 和含 nan/inf 的上传
        valid_head = [h for h in head_updates if h is not None and is_valid_state_dict(h)]
        valid_tail = [t for t in tail_updates if t is not None and is_valid_state_dict(t)]
        print(f"[Cloud] 有效 head={len(valid_head)}，有效 tail={len(valid_tail)}")

        if not valid_head:
            print("[WARN] Cloud 聚合失败：无有效 head 参数上传，本轮参数保持不变")
            return

        # ======= 聚合 head ========
        #avg_head = average_state_dicts(valid_head)
        if quantizer_name and quantizer_name.lower() == "signsgd":
            print("[Cloud] ⚠️ SignSGD 模式下使用 simple sum 聚合")
            avg_head = sum_state_dicts(valid_head)
        else:
            avg_head = average_state_dicts(valid_head)
        self.global_head.load_state_dict(avg_head)
        print(f"[Cloud] head 聚合完成：{len(valid_head)} 个 RSU")
        # 统计参数分布
        #for k, v in avg_head.items():
            #print(f"    [head][{k}] mean={v.mean().item():.6f}, std={v.std().item():.6f}, nan={torch.isnan(v).sum().item()}, inf={torch.isinf(v).sum().item()}")

        # ======= 聚合 tail ========
        if valid_tail:
            #avg_tail = average_state_dicts(valid_tail)
            if quantizer_name and quantizer_name.lower() == "signsgd":
                print("[Cloud] ⚠️ SignSGD 模式下 tail 使用 simple sum 聚合")
                avg_tail = sum_state_dicts(valid_tail)
            else:
                avg_tail = average_state_dicts(valid_tail)

            self.global_tail.load_state_dict(avg_tail)
            print(f"[Cloud] tail 聚合完成：{len(valid_tail)} 个 RSU")
            #for k, v in avg_tail.items():
                #print(f"    [tail][{k}] mean={v.mean().item():.6f}, std={v.std().item():.6f}, nan={torch.isnan(v).sum().item()}, inf={torch.isinf(v).sum().item()}")
        else:
            print("[WARN] Cloud 聚合警告：无有效 tail 参数上传，仅聚合 head")

        print(f"✅ 成功聚合 head({len(valid_head)}) 与 tail({len(valid_tail)}) 参数")

def sum_state_dicts(updates):
    assert len(updates) > 0, "sum_state_dicts: 输入为空"
    device = next(iter(updates[0].values())).device
    keys = updates[0].keys()
    summed = {k: torch.zeros_like(v).to(device) for k, v in updates[0].items()}
    for upd in updates:
        for k in keys:
            summed[k] += upd[k].to(device)
    return summed
