import torch
import joblib

MEMORY_EXT = ".mem"

def save_checkpoint(progress, dqn_online, dqn_target, optimizer, filename):
    checkpoint = {
        "progress": progress,
        "dqn_online": dqn_online.state_dict(),
        "dqn_target": dqn_target.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(dqn_online, dqn_target, optimizer, filename):
    checkpoint = torch.load(filename)
    progress = checkpoint["progress"]
    dqn_online.load_state_dict(checkpoint["dqn_online"])
    dqn_target.load_state_dict(checkpoint["dqn_target"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return progress
