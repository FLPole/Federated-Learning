import argparse
from nodes.vehicle_node_Aggregation import VehicleNode
from federated import federated_training
import os
import torch

#os.environ["HF_HUB_DISABLE_IPV6"] = "1"

if __name__ == "__main__":

    if  torch.cuda.is_available():
        _ = torch.tensor([0.], device="cuda")



    parser = argparse.ArgumentParser(description="Federated Learning with SUMO Simulation")

    # 添加命令行参数
    parser.add_argument("--dataset", type=str, default="BDD",
                        choices=["MNIST", "BDD", "CIFAR10"], help="Dataset to use")
    parser.add_argument("--agg", type=str, default="Fed_AVG",
                        choices=["Fed_AVG", "Fed_Prox", "Fed_AVGM", "Fed_Nova", "Fed_Adam", "Fed_AdaGrad", "Fed_MOON"],
                        help="Aggregation strategy")
    parser.add_argument("--quantizer", type=str, default="MSCQ",
                        choices=["Adaquant", "PowerSGD", "SignSGD", "TernGrad", "QSGD", "MSCQ"],
                        help="Gradient quantizer")
    parser.add_argument("--body", type=str, default="ViT",
                        choices=["ViT", "CNN", "ResNet", "DeiT"], help="Cloud body model")
    parser.add_argument("--rounds", type=int, default=200, help="Number of communication rounds")
    parser.add_argument("--snr", type=int, default=25, help="Default SNR for vehicle nodes")
    parser.add_argument("--sumo_cfg", type=str, default="sumo_config/osm.sumocfg", help="Path to SUMO config file")
    parser.add_argument("--clients", type=int, default=20, help="Number of clients (vehicles)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        choices=[0.1, 0.5, 1.0, 10.0], help="Dirichlet alpha for non-IID partitioning")
    parser.add_argument("--level", type=int, default=int(1e7),
                        choices=[0.1, 0.5, 1.0, 10],
                        help="2 bit量化，s为1或2；4 bit量化，s为7或8；6 bit量化，s为31或32；8 bit量化，s为127或128；")
    parser.add_argument("--initial_lr", type=float, default=0.01, help="Set the initial learning rate")

    args = parser.parse_args()

    # 数据集预处理
    dataset_name = args.dataset
    if dataset_name.lower() == "mnist":
        VehicleNode.generate_mnist_data(dataset_name, num_clients=args.clients, alpha=args.alpha)
    elif dataset_name.lower() == "bdd":
        VehicleNode.generate_bdd_data(
            label_json_dir="data/BDD100K/bdd100k_labels/train",
            image_dir="data/BDD100K/bdd100k_images/train",
            num_clients=args.clients,
            alpha=args.alpha,
            non_iid=True
        )
        VehicleNode.generate_bdd_global_test(
            label_json_dir="data/BDD100K/bdd100k_labels/val",
            image_dir="data/BDD100K/bdd100k_images/val"
        )
    elif dataset_name.lower() == "cifar10":
        VehicleNode.generate_cifar10_data(
            num_clients=args.clients,
            alpha=args.alpha,
            test_split=0.2,
            seed=42
        )
        VehicleNode.generate_cifar10_global_test(save_path="data/cifar10/partitions/global_test.pt")

    # 联邦学习主函数
    federated_training(
        sumocfg_path=args.sumo_cfg,
        dataset_name=dataset_name,
        comm_rounds=args.rounds,
        snr_default=args.snr,
        body_model_name=args.body,
        agg_strategy_name=args.agg,
        quantizer_name=args.quantizer,
        quantizer_level = args.level,
        initial_lr=args.initial_lr,
        alpha=args.alpha
    )
