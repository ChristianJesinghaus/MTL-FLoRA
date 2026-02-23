from collections import defaultdict
import torch

#TODO: Integreate weights per clients
def aggregate_A_matrix(result, layer, layer_param, weights_dict):
    lora_param = "lora_A"
    A_blocks = []

    inner = result[layer][lora_param].get(layer_param, {})

    for client, A_k in inner.items():
        if not client.startswith("client_"):
            continue
        p_k = weights_dict.get(client, None)     # e.g. |D_k| / sum_j |D_j|
        if p_k is None:
            raise KeyError(f"Weight for {client} not found in weights_dict")
        # Ensure weight is a tensor on the same device and dtype as A_k
        p_k_t = torch.tensor(p_k, device=A_k.device, dtype=A_k.dtype)
        A_k_scaled = A_k * p_k_t     # Scale A by p_k (FLORA-style)
        
        A_blocks.append(A_k_scaled)

    if len(A_blocks) == 0:
        # No client blocks present; return an empty tensor
        result[layer][lora_param][layer_param]["aggregated"] = torch.tensor([], device='cpu')
        return result

    # Stack vertically along rank dimension
    A_global = torch.cat(A_blocks, dim=1)

    result[layer][lora_param][layer_param]["aggregated"] = A_global
    return result

def aggreate_B_matrix(result, layer, layer_param):
    lora_param = "lora_B"
    B_blocks = []

    inner = result[layer][lora_param].get(layer_param, {})

    for client, B_k in inner.items():
        if not client.startswith("client_"):
            continue
        B_blocks.append(B_k)

    if len(B_blocks) == 0:
        # No client blocks present; return an empty tensor
        result[layer][lora_param][layer_param]["aggregated"] = torch.tensor([], device='cpu')
        return result

    # Stack horizontally along rank dimension on the device of the first block
    device = B_blocks[0].device
    stacked_B = torch.cat([b.to(device) for b in B_blocks], dim=2)

    result[layer][lora_param][layer_param]["aggregated"] = stacked_B
    return result



def aggregate_lambdas(result, layer, layer_param):
    lora_param = "lora_lambdas"
    
    inner = result[layer][lora_param].get(layer_param, {})
    clients = [k for k in inner.keys() if k.startswith("client_")]
    if not clients:
        result[layer][lora_param][layer_param]["aggregated"] = torch.tensor([], device='cpu')
        return result

    num_tasks = len(inner[clients[0]])  # e.g., Lambda_k is a list/tensor per task

    # Will store one block-diagonal Lambda per task
    Lambda_global_tasks = []

    for t in range(num_tasks):
        Lambda_blocks = []
        client_ranks = []

        # Collect all clients' Lambda for task t
        for client in clients:
            Lambda_k = inner[client][t]  # task t
            r_k = Lambda_k.shape[0]
            client_ranks.append(r_k)
            Lambda_blocks.append(Lambda_k)

        total_rank = sum(client_ranks)
        # Create the global Lambda matrix on the same device and dtype as client Lambdas
        device = Lambda_blocks[0].device
        dtype = Lambda_blocks[0].dtype
        Lambda_global = torch.zeros((total_rank, total_rank), dtype=dtype, device=device)

        # Fill block-diagonal
        start_row = 0
        start_col = 0
        for i, Lambda_k in enumerate(Lambda_blocks):
            r_k = client_ranks[i]
            Lambda_global[start_row:start_row+r_k, start_col:start_col+r_k] = Lambda_k
            start_row += r_k
            start_col += r_k
        Lambda_global_tasks.append(Lambda_global)

    # Stack all tasks along first dimension
    Lambda_global_tasks = torch.stack(Lambda_global_tasks, dim=0)
    result[layer][lora_param][layer_param]["aggregated"] = Lambda_global_tasks

    #print(f"Aggregated Lambda shape (tasks, total_rank, total_rank): {Lambda_global_tasks.shape}")
    return result




def aggregate_B_w_matrix(result, layer, layer_param, weights_dict):
    """
    Aggregates B_w matrices across clients by averaging (optionally weighted by dataset size).

    Args:
        result: dict containing per-client B_w matrices
        layer: str, layer name in the dict
        layer_param: str, the specific parameter name under the lora param
        weights_dict: dict mapping client_id -> weight (p_k). If None, do simple average
    """
    lora_param = "lora_B_w"
    inner = result[layer][lora_param].get(layer_param, {})
    clients = [k for k in inner.keys() if k.startswith("client_")]
    if not clients:
        result[layer][lora_param][layer_param]["aggregated"] = torch.tensor([], device='cpu')
        return result

    num_tasks = len(inner[clients[0]])  # assume same number of tasks

    # Initialize list for aggregated B_w
    aggregated_B_w = []

    for t in range(num_tasks):
        # Collect B_w for task t from all clients
        task_B_w_list = []
        for client in clients:
            B_w_t = inner[client][t]
            if weights_dict is not None:
                # scale by client weight p_k as tensor on same device/dtype
                p_k = weights_dict.get(client, None)
                if p_k is None:
                    raise KeyError(f"Weight for {client} not found in weights_dict")
                p_k_t = torch.tensor(p_k, device=B_w_t.device, dtype=B_w_t.dtype)
                B_w_t = B_w_t * p_k_t
            task_B_w_list.append(B_w_t)

        # Sum (or average)
        task_B_w_global = torch.stack(task_B_w_list, dim=0).sum(dim=0)
        if weights_dict is None:
            task_B_w_global = task_B_w_global / len(clients)  # simple average
        aggregated_B_w.append(task_B_w_global)

    # Stack tasks into a tensor
    aggregated_B_w = torch.stack(aggregated_B_w, dim=0)
    result[layer][lora_param][layer_param]["aggregated"] = aggregated_B_w

    #print(f"Aggregated B_w shape: {aggregated_B_w.shape}")
    return result


def print_shapes_per_parameter(client_weights):
    """
    Print shapes of all parameters across client_weights.
    
    Args:
        client_weights: list of dicts with parameter tensors
    """
    if not client_weights:
        print("No client weights to print")
        return
    
    # Collect all unique keys
    all_keys = set()
    for c_dict in client_weights:
        all_keys.update(c_dict.keys())
    
    # Print shapes for each key
    for key in sorted(all_keys):
        print(f"\n{key}:")
        for i, c_dict in enumerate(client_weights):
            if key in c_dict:
                print(f"  client_{i+1}: {tuple(c_dict[key].shape)}")


def convert_result_to_dict(result): # TODO refactor: return dict instead of List[Dict]
    """
    Convert the aggregated `result` structure back into a list of client-style dicts.

    Args:
        result: dict mapping layer -> lora_param -> { 'client_i': tensor, ... , 'aggregated': tensor }

    Returns:
        List[dict]: one dict per client where keys match the original format:
            "encoder.encoder.layer.{layer}.attention.self.query.{lora_param}"
    """
    # discover number of clients by scanning keys like 'client_1', 'client_2', ...
    max_idx = 0
    for layer in result:
        for lora_param in result[layer]:
            for layer_param in result[layer][lora_param]:
                for k in result[layer][lora_param][layer_param].keys():
                    if k.startswith("client_"):
                        try:
                            idx = int(k.split("_")[1])
                            if idx > max_idx:
                                max_idx = idx
                        except Exception:
                            continue

    if max_idx == 0:
        return []

    client_dicts = [dict() for _ in range(max_idx)]

    for layer in result:
        for lora_param in result[layer]:
            for layer_param in result[layer][lora_param]:
                # Get original name if available, otherwise construct it
                original_name = result[layer][lora_param][layer_param].get("original_name", None)
                assert original_name is not None, f"Original name missing in result structure for {layer},{lora_param},{layer_param}."
                
                for client_key, value in result[layer][lora_param][layer_param].items():
                    if not client_key.startswith("client_"):
                        # skip aggregated or other non-client entries
                        continue
                    try:
                        idx = int(client_key.split("_")[1]) - 1
                    except Exception:
                        continue

                    key = original_name
                
                    client_dicts[idx][key] = value

    # If there is an aggregated value for a param, set it for all clients
    for layer in result:
        for lora_param in result[layer]:
            for layer_param in result[layer][lora_param]:
                agg = result[layer][lora_param][layer_param].get('aggregated', None)
                if agg is None:
                    continue
                # Get original name if available, otherwise construct it
                original_name = result[layer][lora_param][layer_param].get("original_name", None)
                if original_name:
                    key = original_name
                else:
                    key = f"encoder.encoder.layer.{layer}.attention.self.query.{lora_param}"
                for idx in range(max_idx):
                    client_dicts[idx][key] = agg

    return client_dicts


# TODO: Set correct weights per clients
def aggregate_lora_parameters(client_weights, weights_dict={ "client_1": 0.5, "client_2": 0.5 }):

    zipped_client_layers = []

    for client in client_weights:
        zipped_client_layers.append(list(client.items()))

    data = list(zip(*zipped_client_layers))
    result = defaultdict(lambda: defaultdict(dict))

    for item in data:
        key_string = item[0][0]
        layer = int(key_string.split("layer.")[1].split(".")[0])
        lora_param = key_string.split(".")[-1]
        layer_param = key_string.split(".")[-2]
        
        # Ensure container exists for this layer_param and store original parameter name
        result[layer][lora_param].setdefault(layer_param, {})
        result[layer][lora_param][layer_param]["original_name"] = key_string
        
        # Store values for each client
        for client_idx, (key, value) in enumerate(item):
            result[layer][lora_param][layer_param][f"client_{client_idx + 1}"] = value
        
    # Example access:

    for layer in result:
        for lora_param in result[layer]:
            for layer_param in result[layer][lora_param]:
                if lora_param == "lora_B_w":
                    result = aggregate_B_w_matrix(result, layer, layer_param, weights_dict)

                elif lora_param == "lora_B":
                    result = aggreate_B_matrix(result, layer, layer_param)
                
                elif lora_param == "lora_A":
                    result = aggregate_A_matrix(result, layer, layer_param, weights_dict)
                
                elif lora_param == "lora_lambdas":
                    result = aggregate_lambdas(result, layer, layer_param)

    client_weights = convert_result_to_dict(result)[0]   

    return client_weights