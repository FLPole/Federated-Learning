import os
import torch
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from utils.tools import normalize_id
from torch import nn
import torch.optim as optim
from PIL import Image
import json
import glob
import copy

SUPPORTED_KEYS = ["scene", "timeofday", "weather"]

class VehicleNode:
    def __init__(self, vehicle_id, position, speed, dataset_name, aggregation_strategy, is_connected=False, snr=None):
        self.vehicle_id = vehicle_id
        self.position = position
        self.speed = speed
        self.connected = is_connected
        self.snr = snr if snr is not None else 25

        self.train_loader = None
        self.eligible_to_train = self._can_train()
        self.eligible_to_upload = self._can_upload()
        self.compute_power = self._assign_compute_power()
        self.aggregation_strategy = aggregation_strategy.lower()
        self.dataset_name = dataset_name
        self.dataset = self._load_partitioned_data()

        # FedMOON 所用参数
        self.prev_head = None
        self.prev_tail = None

    def _can_train(self):
        return self.connected and self.speed < 30

    def _can_upload(self):
        return self.connected and self.snr >= 15

    def _assign_compute_power(self):
        if self.speed > 30:
            return "high"
        elif self.speed > 10:
            return "medium"
        else:
            return "low"

    def _load_partitioned_data(self):
        # data_id = normalize_id(self.vehicle_id)
        data_id = self.vehicle_id

        if self.dataset_name.lower() == "mnist":
            path = f"data/mnist/partitions/{data_id}_train.pt"
            if not os.path.exists(path):
                print(f"[{self.vehicle_id}] ⚠️ 数据未找到，请先生成数据集")
                self.train_loader = None
                return
            xs, ys = torch.load(path, weights_only=True)

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

            class MNIST_RGB_Dataset(torch.utils.data.Dataset):
                def __init__(self, xs, ys, transform=None):
                    self.xs = xs
                    self.ys = ys
                    self.transform = transform

                def __len__(self):
                    return len(self.xs)

                def __getitem__(self, idx):
                    img = self.xs[idx].expand(3, 28, 28)  # [3, 28, 28]
                    img = transforms.ToPILImage()(img)  # PIL Image
                    if self.transform:
                        img = self.transform(img)  # Resize + ToTensor
                    label = self.ys[idx]
                    return img, label

            dataset = MNIST_RGB_Dataset(xs, ys, transform)
            print(f"[{self.vehicle_id}] ✅ 加载本地 MNIST 数据，共 {len(xs)} 条样本")

        elif self.dataset_name.lower() == "bdd":
            path = f"data/BDD100K/clients/{data_id}_bdd.pt"
            if not os.path.exists(path):
                print(f"[{self.vehicle_id}] ⚠️ BDD 数据未找到，请先生成")
                self.train_loader = None
                return
            img_paths, labels = torch.load(path)

            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 保持与 ViT 输入一致
                transforms.ToTensor()
            ])

            class BDDSimpleDataset(torch.utils.data.Dataset):
                def __init__(self, img_paths, labels, transform=None):
                    self.img_paths = img_paths
                    self.labels = labels
                    self.transform = transform

                def __len__(self):
                    return len(self.img_paths)

                def __getitem__(self, idx):
                    img_path = os.path.normpath(self.img_paths[idx] + ".jpg")
#                   #img_path = os.path.normpath(self.img_paths[idx].replace("\\", "/") + ".jpg")
                    image = Image.open(img_path).convert("RGB")
                    if self.transform:
                        image = self.transform(image)
                    label = self.labels[idx]
                    return image, label

            dataset = BDDSimpleDataset(img_paths, labels, transform)
            print(f"[{self.vehicle_id}] ✅ 加载 BDD 图像数据，共 {len(dataset)} 张图")

        elif self.dataset_name.lower() == "cifar10":
            path = f"data/cifar10/partitions/{data_id}_train.pt"
            if not os.path.exists(path):
                print(f"[{self.vehicle_id}] ⚠️ CIFAR10 数据未找到，请先生成数据集")
                self.train_loader = None
                return
            xs, ys = torch.load(path, weights_only=True)

            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 适配 ViT 输入
                transforms.ToTensor()
            ])

            class CIFARDataset(torch.utils.data.Dataset):
                def __init__(self, xs, ys, transform=None):
                    self.xs = xs
                    self.ys = ys
                    self.transform = transform

                def __len__(self):
                    return len(self.xs)

                def __getitem__(self, idx):
                    img = transforms.ToPILImage()(self.xs[idx])  # 将 tensor 转为 PIL
                    if self.transform:
                        img = self.transform(img)
                    label = self.ys[idx]
                    return img, label

            dataset = CIFARDataset(xs, ys, transform)
            print(f"[{self.vehicle_id}] ✅ 加载 CIFAR10 数据，共 {len(xs)} 条样本")


        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        #self.train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    def local_train(self, model_dict, lr):
        if not self.eligible_to_train:
            print(f"[{self.vehicle_id}] ❌ 未连接 RSU 或速度过快，跳过训练")
            return None

        if not self.train_loader:
            print(f"[{self.vehicle_id}] ❌ 无本地数据，跳过训练")
            return None

        printed_device_info = False

        head = model_dict["head"]
        body = model_dict["body"]  # 不更新，只 forward
        tail = model_dict["tail"]

        # 将 head 和 tail 放入同一个 optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        head, body, tail = head.to(device), body.to(device), tail.to(device)
        body.eval()  # 固定 cloud 模型

        # 保存全局初始参数，用于 FedProx 正则项
        if self.aggregation_strategy == "fed_prox":
            global_head = copy.deepcopy(head)
            global_tail = copy.deepcopy(tail)

        # 如果是 FedMOON，初始化 prev_head 和 prev_tail
        use_fedmoon = self.aggregation_strategy == "fed_moon"
        if use_fedmoon and (not hasattr(self, "prev_head") or self.prev_head is None):
            self.prev_head = copy.deepcopy(head)
            self.prev_tail = copy.deepcopy(tail)

        #epochs = {"low": 5, "medium": 3, "high": 1}[self.compute_power]
        epochs = {"low": 1, "medium": 1, "high": 1}[self.compute_power]
        print(f"[{self.vehicle_id}] ✅ 开始训练 {epochs} epochs（SNR: {self.snr}, 连接: {self.connected}）")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(list(head.parameters()) + list(tail.parameters()), lr=lr, momentum=0.9)

        head.train()
        tail.train()

        #total_loss = 0.0
        for epoch in range(epochs):
            total_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs = inputs.to(device)

                # ✅ 安全处理 labels
                if isinstance(labels, dict):
                    labels = labels["scene"]
                labels = torch.as_tensor(labels).long().to(device)

                # ✅ 设备检查打印
                if not printed_device_info:
                    print(
                        f"[{self.vehicle_id}]  inputs: {inputs.device}, labels: {labels.device}, type: {type(labels)}")
                    printed_device_info = True

                """
                if self.dataset_name.lower() == "mnist":
                    labels = labels.to(device)
                elif self.dataset_name.lower() == "bdd":
                    if isinstance(labels, dict):
                        labels = labels["scene"]
                    labels = labels.to(device)
                elif self.dataset_name.lower() == "cifar10":
                    labels = labels.long().to(device)
                """




                optimizer.zero_grad()

                # 三段式 forward
                features = head(inputs)  # [B, N, D]
                mid_token = body(features)  # [B, D]
                outputs = tail(mid_token)  # [B, num_classes]


                loss = criterion(outputs, labels)

                # 如果是 FedProx，加入正则项, 如果是 FedMOON， 则计算更新前后差异
                if self.aggregation_strategy== "fed_prox":
                    mu = 0.01
                    prox_loss = 0.0
                    for param, global_param in zip(head.parameters(), global_head.parameters()):
                        prox_loss += ((param - global_param.detach()) ** 2).sum()
                    for param, global_param in zip(tail.parameters(), global_tail.parameters()):
                        prox_loss += ((param - global_param.detach()) ** 2).sum()
                    loss += (mu / 2) * prox_loss
                elif use_fedmoon:
                    moon_loss_weight = 1.0
                    cos = torch.nn.CosineSimilarity(dim=-1)
                    current_head_vec = torch.cat([p.flatten() for p in head.parameters()])
                    prev_head_vec = torch.cat([p.flatten() for p in self.prev_head.parameters()])
                    global_head_vec = torch.cat([p.flatten() for p in model_dict["head"].parameters()])
                    sim_prev = cos(current_head_vec, prev_head_vec)
                    sim_global = cos(current_head_vec, global_head_vec)
                    moon_loss = -torch.log(torch.exp(sim_global) / (torch.exp(sim_prev) + torch.exp(sim_global)))
                    loss += moon_loss_weight * moon_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"[{self.vehicle_id}] 易 Epoch {epoch + 1}/{epochs} Loss: {total_loss:.4f}")
            

        if not self.eligible_to_upload:
            print(f"[{self.vehicle_id}] ⚠️ 通信质量差，未上传模型")
            return None

        #  新增 FedNova 所需字段
        sample_num = len(self.train_loader.dataset)
        local_steps = epochs * len(self.train_loader)

        # 更新 FedMOON 历史模型
        if use_fedmoon:
            self.prev_head = copy.deepcopy(head)
            self.prev_tail = copy.deepcopy(tail)

        return {
            "head": head.state_dict(),
            "tail": tail.state_dict(),
            "sample_num": sample_num,
            "local_steps": local_steps,
            "final_loss": total_loss
        }

    # === 以下静态方法不变，可在主程序中调用生成数据 ===
    @staticmethod
    def generate_mnist_data(dataset_name, num_clients, alpha, test_split=0.2, seed=42):
        from torchvision import datasets as dsets

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        save_dir = f"data/{dataset_name.lower()}/partitions"
        os.makedirs(save_dir, exist_ok=True)
        transform = transforms.ToTensor()

        if dataset_name == "MNIST":
            full_dataset = dsets.MNIST(root="data", train=True, download=True, transform=transform)
        elif dataset_name == "FashionMNIST":
            full_dataset = dsets.FashionMNIST(root="data", train=True, download=True, transform=transform)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        data = full_dataset.data
        targets = full_dataset.targets
        class_indices = {i: (targets == i).nonzero(as_tuple=True)[0].tolist() for i in range(10)}
        client_indices = [[] for _ in range(num_clients)]
        class_distribution = np.zeros((num_clients, 10))

        for cls in range(10):
            cls_idx = class_indices[cls]
            np.random.shuffle(cls_idx)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
            split_indices = np.split(cls_idx, proportions)
            for client_id, idx in enumerate(split_indices):
                client_indices[client_id].extend(idx)
                class_distribution[client_id, cls] += len(idx)

        for cid in range(num_clients):
            indices = client_indices[cid]
            np.random.shuffle(indices)
            xs = data[indices].unsqueeze(1).float() / 255.0
            ys = targets[indices]

            n_total = len(xs)
            n_test = int(test_split * n_total)
            x_train, y_train = xs[:n_total - n_test], ys[:n_total - n_test]
            x_test, y_test = xs[n_total - n_test:], ys[n_total - n_test:]

            torch.save((x_train, y_train), os.path.join(save_dir, f"veh{cid + 1:03d}_train.pt"))
            torch.save((x_test, y_test), os.path.join(save_dir, f"veh{cid + 1:03d}_test.pt"))

        plt.figure(figsize=(10, 6))
        class_distribution = class_distribution / class_distribution.sum(1, keepdims=True)
        for i in range(10):
            plt.bar(np.arange(num_clients), class_distribution[:, i],
                    bottom=class_distribution[:, :i].sum(1), label=f"Class {i}")
        plt.xlabel("Client ID")
        plt.ylabel("Proportion")
        plt.title(f"{dataset_name} Class Distribution (α={alpha})")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(f"data/{dataset_name.lower()}/partition_vis.png")
        plt.close()

    @staticmethod
    def generate_bdd_data(label_json_dir, image_dir, num_clients, alpha, non_iid, save_dir="data/BDD100K/clients", seed=42):
        random.seed(seed)
        np.random.seed(seed)

        label_files = glob.glob(os.path.join(label_json_dir, "*.json"))
        labels_json = []
        for file_path in label_files:
            with open(file_path, "r") as f:
                labels_json.append(json.load(f))
        print(f"✅ 读取 {len(labels_json)} 个 JSON 标签文件")

        data = []
        for item in labels_json:
            image_name = item["name"]
            full_path = os.path.join(image_dir, image_name)
            label_dict = {}
            for key in SUPPORTED_KEYS:
                label = item["attributes"].get(key, None)
                if label:
                    label_dict[key] = label
            if len(label_dict) == len(SUPPORTED_KEYS):
                data.append((full_path, label_dict))

        print(f"✅ 解析到 {len(data)} 张 BDD 图像")

        label_maps = {k: {v: i for i, v in enumerate(sorted(set(d[k] for _, d in data)))} for k in SUPPORTED_KEYS}
        encoded_data = [(path, {k: label_maps[k][v] for k, v in labels.items()}) for path, labels in data]

        client_data = [[] for _ in range(num_clients)]
        class_distribution = np.zeros((num_clients, len(label_maps["scene"])))

        if non_iid:
            label_to_indices = defaultdict(list)
            for i, (_, labels) in enumerate(encoded_data):
                label_to_indices[labels["scene"]].append(i)
            for label, indices in label_to_indices.items():
                np.random.shuffle(indices)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
                split_indices = np.split(indices, proportions)
                for cid, idxs in enumerate(split_indices):
                    for i in idxs:
                        client_data[cid].append(encoded_data[i])
                        class_distribution[cid, label] += 1
        else:
            np.random.shuffle(encoded_data)
            split_size = len(encoded_data) // num_clients
            for cid in range(num_clients):
                client_data[cid] = encoded_data[cid * split_size: (cid + 1) * split_size]
                for _, labels in client_data[cid]:
                    class_distribution[cid, labels["scene"]] += 1

        os.makedirs(save_dir, exist_ok=True)
        for cid, records in enumerate(client_data):
            image_paths = [r[0] for r in records]
            label_dicts = [r[1] for r in records]
            torch.save((image_paths, label_dicts), os.path.join(save_dir, f"veh{cid+1:03d}_bdd.pt"))

        print(f"✅ 已保存 {num_clients} 个客户端的 BDD 图像任务数据")

        # ✅ 输出 scene 类别并用于图例
        scene_classes = set()
        for _, label in data:
            scene_classes.add(label["scene"])
        scene_classes = sorted(scene_classes)
        print("BDD100K 中的 scene 类别有：", scene_classes)

        # ✅ 可视化类分布
        class_distribution_normalized = class_distribution / class_distribution.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(class_distribution.shape[1]):
            ax.bar(np.arange(num_clients), class_distribution_normalized[:, i],
                   bottom=class_distribution_normalized[:, :i].sum(1), label=f"{scene_classes[i]}")
        ax.set_xlabel("Client ID")
        ax.set_ylabel("Proportion")
        ax.set_title(f"BDD Scene Class Distribution (alpha={alpha})")
        ax.legend()
        plt.tight_layout()
        plt.show()
        os.makedirs("data/BDD100K", exist_ok=True)
        plt.savefig("data/BDD100K/partition_vis.png")
        plt.close()
        print("✅ 分布可视化图保存至 data/BDD100K/partition_vis.png")

    @staticmethod
    def generate_bdd_global_test(label_json_dir, image_dir, save_path="data/BDD100K/global_test.pt"):
        label_files = glob.glob(os.path.join(label_json_dir, "*.json"))
        labels_json = []
        for file_path in label_files:
            with open(file_path, "r") as f:
                labels_json.append(json.load(f))

        data = []
        for item in labels_json:
            image_name = item["name"]
            full_path = os.path.join(image_dir, image_name)
            label_dict = {}
            for key in SUPPORTED_KEYS:
                label = item["attributes"].get(key, None)
                if label:
                    label_dict[key] = label
            if len(label_dict) == len(SUPPORTED_KEYS):
                data.append((full_path, label_dict))

        label_maps = {k: {v: i for i, v in enumerate(sorted(set(d[k] for _, d in data)))} for k in SUPPORTED_KEYS}
        encoded_data = [(path, {k: label_maps[k][v] for k, v in labels.items()}) for path, labels in data]

        image_paths = [r[0] for r in encoded_data]
        label_dicts = [r[1] for r in encoded_data]
        torch.save((image_paths, label_dicts), save_path)
        print(f"✅ 已保存全局测试集到 {save_path}，共 {len(image_paths)} 张图")

    @staticmethod
    def generate_cifar10_data(num_clients, alpha=0.5, test_split=0.2, seed=42):
        from torchvision import datasets as dsets

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        save_dir = "data/cifar10/partitions"
        os.makedirs(save_dir, exist_ok=True)
        transform = transforms.ToTensor()

        full_dataset = dsets.CIFAR10(root="data", train=True, download=True, transform=transform)
        data = full_dataset.data  # numpy array [N, H, W, C]
        targets = np.array(full_dataset.targets)

        class_indices = {i: np.where(targets == i)[0].tolist() for i in range(10)}
        client_indices = [[] for _ in range(num_clients)]
        class_distribution = np.zeros((num_clients, 10))

        for cls in range(10):
            cls_idx = class_indices[cls]
            np.random.shuffle(cls_idx)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
            split_indices = np.split(cls_idx, proportions)
            for client_id, idx in enumerate(split_indices):
                client_indices[client_id].extend(idx)
                class_distribution[client_id, cls] += len(idx)

        for cid in range(num_clients):
            indices = client_indices[cid]
            np.random.shuffle(indices)
            xs = torch.tensor(data[indices]).permute(0, 3, 1, 2).float() / 255.0  # [B, C, H, W]
            ys = torch.tensor(targets[indices])

            n_total = len(xs)
            n_test = int(test_split * n_total)
            x_train, y_train = xs[:n_total - n_test], ys[:n_total - n_test]
            x_test, y_test = xs[n_total - n_test:], ys[n_total - n_test:]

            torch.save((x_train, y_train), os.path.join(save_dir, f"veh{cid}_train.pt"))
            torch.save((x_test, y_test), os.path.join(save_dir, f"veh{cid}_test.pt"))

        # 可视化
        plt.figure(figsize=(10, 6))
        class_distribution = class_distribution / class_distribution.sum(1, keepdims=True)
        for i in range(10):
            plt.bar(np.arange(num_clients), class_distribution[:, i],
                    bottom=class_distribution[:, :i].sum(1), label=f"Class {i}")
        plt.xlabel("Client ID")
        plt.ylabel("Proportion")
        plt.title(f"CIFAR10 Class Distribution (α={alpha})")
        plt.legend()
        plt.tight_layout()
        plt.savefig("data/cifar10/partition_vis.png")
        plt.show()
        print("✅ 分布可视化图保存至 data/cifar10/partition_vis.png")

    @staticmethod
    def generate_cifar10_global_test(save_path="data/cifar10/partitions/global_test.pt"):
        from torchvision import datasets as dsets
        import torchvision.transforms as transforms

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        transform = transforms.ToTensor()
        test_dataset = dsets.CIFAR10(root="data", train=False, download=True, transform=transform)

        # 提取全部测试样本
        xs = torch.stack([img for img, _ in test_dataset])
        ys = torch.tensor([label for _, label in test_dataset])

        torch.save((xs, ys), save_path)
        print(f"✅ 已保存 CIFAR10 全局测试集，共 {len(xs)} 条样本，路径：{save_path}")

