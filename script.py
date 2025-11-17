import os
import time
import argparse
import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from clearml import Task

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--in_dim", type=int, default=128)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--num_samples", type=int, default=100_000)
    return p.parse_args()

def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, device

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    Task.init("Multi Node Pytorch", "Multi Node Pytorch")
    args = parse_args()
    torch.backends.cudnn.benchmark = True

    rank, world_size, local_rank, device = setup_distributed()
    is_main = rank == 0

    # ----- Tiny synthetic classification problem -----
    torch.manual_seed(0)
    X = torch.randn(args.num_samples, args.in_dim)
    true_W = torch.randn(args.in_dim, args.num_classes)
    logits = X @ true_W + 0.1 * torch.randn(args.num_samples, args.num_classes)
    y = logits.argmax(dim=1)

    # Ensure equal steps across ranks: trim to multiple of world_size * batch_size
    global_bs = world_size * args.batch_size
    usable = (len(X) // global_bs) * global_bs
    if usable == 0:
        raise ValueError(
            f"num_samples={len(X)} too small for world_size*batch_size={global_bs}"
        )
    if usable != len(X) and is_main:
        print(f"Trimming dataset from {len(X)} to {usable} samples for even batching.")
    X, y = X[:usable], y[:usable]

    dataset = TensorDataset(X, y)
    sampler = DistributedSampler(
        dataset,
        shuffle=True,
        drop_last=True,  # <-- critical to avoid uneven final batches
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,  # matches sampler behavior
    )

    # ----- Simple MLP -----
    model = nn.Sequential(
        nn.Linear(args.in_dim, 256),
        nn.ReLU(),
        nn.Linear(256, args.num_classes),
    ).to(device)

    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(ddp_model.parameters(), lr=args.lr)

    if is_main:
        print(f"[World {world_size}] Starting training on {torch.cuda.get_device_name(device)}")
    dist.barrier()  # sync before starting

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        sampler.set_epoch(epoch)
        running_loss = 0.0
        n_batches = 0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = ddp_model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        # Average the epoch loss across all ranks
        loss_tensor = torch.tensor([running_loss / max(n_batches, 1)], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        if is_main:
            print(f"Epoch {epoch:02d} | loss={loss_tensor.item():.4f} | time={time.time()-epoch_start:.2f}s")

    # Sync before saving to avoid teardown races
    dist.barrier()
    if is_main:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({"model": ddp_model.module.state_dict()}, "checkpoints/model.pt")
        print("Saved checkpoints/model.pt")
    dist.barrier()

    cleanup_distributed()

if __name__ == "__main__":
    main()
