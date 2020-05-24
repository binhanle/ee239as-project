import torch

def save_checkpoint(mem_buffer, progress, dqn_online, dqn_target, optimizer, filename):
    checkpoint = {
        "mem_buffer": mem_buffer,
        "progress": progress,
        "dqn_online": dqn_online.state_dict(),
        "dqn_target": dqn_target.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(dqn_online, dqn_target, optimizer, filename):
    checkpoint = torch.load(filename)
    mem_buffer = checkpoint['mem_buffer']
    progress = checkpoint['progress']
    dqn_online.load_state_dict(checkpoint['dqn_online'])
    dqn_target.load_state_dict(checkpoint['dqn_target'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return mem_buffer, progress
