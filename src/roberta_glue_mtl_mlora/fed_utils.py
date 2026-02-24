from collections import defaultdict
import torch


def aggregate_A_matrix(client_matrices, weights_dict):
    """Stack A matrices along dim 1 with weights."""
    A_blocks = []
    for i, A_k in enumerate(client_matrices):
        client_key = f"client_{i + 1}"
        p_k = weights_dict.get(client_key, 1.0)
        p_k_t = torch.tensor(p_k, device=A_k.device, dtype=A_k.dtype)
        A_scaled = A_k * p_k_t
        A_blocks.append(A_scaled)
    return torch.cat(A_blocks, dim=1)

def aggregate_B_matrix(client_matrices):
    """Stack B matrices along dim 2."""
    return torch.cat(client_matrices, dim=2)

def aggregate_lambdas(client_matrices):
    """Create block-diagonal Lambda matrices per task."""
    num_tasks = client_matrices[0].shape[0] if len(client_matrices[0].shape) == 3 else 1
    Lambda_global_tasks = []
    
    for t in range(num_tasks):
        Lambda_blocks = []
        client_ranks = []
        
        for Lambda_k in client_matrices:
            if len(Lambda_k.shape) == 3:
                Lam_t = Lambda_k[t]
            else:
                Lam_t = Lambda_k
            r_k = Lam_t.shape[0]
            client_ranks.append(r_k)
            Lambda_blocks.append(Lam_t)
        
        total_rank = sum(client_ranks)
        device = Lambda_blocks[0].device
        dtype = Lambda_blocks[0].dtype
        Lambda_global = torch.zeros((total_rank, total_rank), dtype=dtype, device=device)
        
        start_row = 0
        start_col = 0
        for i, Lambda_k in enumerate(Lambda_blocks):
            r_k = client_ranks[i]
            Lambda_global[start_row:start_row+r_k, start_col:start_col+r_k] = Lambda_k
            start_row += r_k
            start_col += r_k
        Lambda_global_tasks.append(Lambda_global)
    
    return torch.stack(Lambda_global_tasks, dim=0)

def aggregate_B_w_matrix(client_matrices, weights_dict):
    """Average B_w matrices with optional weighting."""
    B_w_list = []
    for i, B_w in enumerate(client_matrices):
        client_key = f"client_{i + 1}"
        p_k = weights_dict.get(client_key, None)
        if p_k is not None:
            p_k_t = torch.tensor(p_k, device=B_w.device, dtype=B_w.dtype)
            B_w = B_w * p_k_t
        B_w_list.append(B_w)
    
    aggregated = torch.stack(B_w_list, dim=0).sum(dim=0)
    if weights_dict is None:
        aggregated = aggregated / len(B_w_list)
    return aggregated


def print_shapes_per_parameter(client_weights):
    """
    Print shapes of all parameters in client_weights.
    
    Args:
        client_weights: dict with parameter tensors
    """
    if not client_weights:
        print("No client weights to print")
        return
    
    # Print shapes for each key
    for key in sorted(client_weights.keys()):
        print(f"{key}: {tuple(client_weights[key].shape)}")



def aggregate_lora_parameters(client_weights, weights_dict={ "client_1": 0.6, "client_2": 0.4 }):
    """
    Aggregate LoRA parameters across clients by matching full keys.
    For each unique parameter key, collect matrices from all clients and aggregate.
    
    Args:
        client_weights: list of dicts, each containing parameter key -> tensor mappings
        weights_dict: dict mapping client_id -> weight for weighted aggregation
    
    Returns:
        dict: aggregated parameters with the same key format as input
    """
    # Collect all unique keys across all clients
    all_keys = set()
    for client in client_weights:
        all_keys.update(client.keys())
    
    aggregated = {}
    
    # Process each unique key
    for key_string in sorted(all_keys):
        # Extract lora_param from the key (last part after split by .)
        lora_param = key_string.split(".")[-1]
        
        # Collect matrices from all clients with this key
        client_matrices = []
        for client in client_weights:
            if key_string in client:
                client_matrices.append(client[key_string])
        
        if not client_matrices:
            continue
        
        print(f"Processing key: {key_string}, lora_param: {lora_param}")
        
        # Aggregate based on lora_param type
        if lora_param == "lora_A":
            aggregated_matrix = aggregate_A_matrix(client_matrices, weights_dict)
        elif lora_param == "lora_B":
            aggregated_matrix = aggregate_B_matrix(client_matrices)
        elif lora_param == "lora_lambdas":
            aggregated_matrix = aggregate_lambdas(client_matrices)
        elif lora_param == "lora_B_w":
            aggregated_matrix = aggregate_B_w_matrix(client_matrices, weights_dict)
        else:
            print(f"Unknown lora_param type: {lora_param}, skipping")
            continue
        
        aggregated[key_string] = aggregated_matrix
    
    return aggregated