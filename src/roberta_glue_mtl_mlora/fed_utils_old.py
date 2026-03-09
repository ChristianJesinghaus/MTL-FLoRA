import torch

'''
Alternative implementation of Approach I
(Works the same way as fed_utils)
'''

def average_mtl_weights(client_weights):
    client_A = []
    client_B = []
    client_lambdas = []
    client_B_w = []

    for weights in client_weights:
        client_A.append({k: v for k, v in weights.items() if k.endswith("lora_A")})
        client_B.append({k: v for k, v in weights.items() if "lora_B" in k and not k.endswith("lora_B_w")})
        client_lambdas.append({k: v for k, v in weights.items() if k.endswith("lora_lambdas")})
        client_B_w.append({k: v for k, v in weights.items() if k.endswith("lora_B_w")})
    
    print(f"[DEBUG] aggregate_mtl_weights: Found {len(client_A[0])} lora_A, {len(client_B[0])} lora_B, {len(client_lambdas[0])} lora_lambdas, {len(client_B_w[0])} lora_B_w in first client")
    
    a_stacked = fed_avg(client_A)
    b_stacked = fed_avg(client_B)
    lambdas_stacked = fed_avg(client_lambdas)
    b_w_avg =  fed_avg(client_B_w)

    agg_weights = {**a_stacked, **b_stacked, **lambdas_stacked, **b_w_avg}
    print(f"[DEBUG] aggregate_mtl_weights: Created {len(agg_weights)} aggregated weights")
    
    return agg_weights



def stack_A(client_A, client_p, hidden, lora_r):
    device = next(iter(client_A[0].values())).device
    num_clients = len(client_A)
    
    stacked = dict()
    for layer in client_A[0]:
        stacked[layer] = torch.cat([client_p[i]*client_A[i][layer] for i in range(num_clients)], dim=1).to(device) # stack As along lora_r for each layer

    assert next(iter(stacked.values())).shape==torch.Size([1, lora_r, hidden]), f"As stacked incorrectly: {next(iter(stacked.values())).shape}"
    return stacked

def stack_B(client_B, num_B, hidden, lora_r):
    device = next(iter(client_B[0].values())).device
    num_clients = len(client_B)
    stacked = dict() #dict.fromkeys(client_B[0], torch.zeros([num_B, hidden, lora_r]))
    for layer in client_B[0]:
        stacked[layer] = torch.cat([client_B[i][layer] for i in range(num_clients)], dim=2).to(device) # stack Bs along lora_r for each layer

    assert next(iter(stacked.values())).shape==torch.Size([num_B, hidden, lora_r]), "Bs stacked incorrectly"
    return stacked


def stack_lambdas(client_lambdas, num_tasks, lora_r):
    device = next(iter(client_lambdas[0].values())).device
    dtype = next(iter(client_lambdas[0].values())).dtype
    num_clients = len(client_lambdas)
    stacked = dict.fromkeys(client_lambdas[0], torch.zeros([num_tasks, lora_r, lora_r], dtype=dtype))

    for layer in client_lambdas[0]:
        lambdas = [client_lambdas[i][layer] for i in range(num_clients)]
        sizes = [l.shape[1] for l in lambdas] # accounting for heterogeneous lora ranks
        offset = 0
        for l, r in zip(lambdas, sizes): # stack lambdas diagonally
            stacked[layer][:, offset:offset+r, offset:offset+r] = l
            offset += r

    return stacked

def avg_B_w(client_B_w, num_tasks, num_B):
    num_clients = len(client_B_w)
    avg = copy.deepcopy(client_B_w[0])

    for layer in client_B_w[0]:
        for i in range(1, num_clients):
            avg[layer] += client_B_w[i][layer]
        avg[layer] = avg[layer] / num_clients

    return avg

def aggregate_mtl_weights(client_weights, client_p, hidden=768, num_B=3, num_tasks=2, lora_r=8):
    client_A = []
    client_B = []
    client_lambdas = []
    client_B_w = []

    for weights in client_weights:
        client_A.append({k: v for k, v in weights.items() if k.endswith("lora_A")})
        client_B.append({k: v for k, v in weights.items() if "lora_B" in k and not k.endswith("lora_B_w")})
        client_lambdas.append({k: v for k, v in weights.items() if k.endswith("lora_lambdas")})
        client_B_w.append({k: v for k, v in weights.items() if k.endswith("lora_B_w")})
    

    a_stacked = stack_A(client_A, client_p, hidden, lora_r)
    b_stacked = stack_B(client_B, num_B, hidden, lora_r)
    lambdas_stacked = stack_lambdas(client_lambdas, num_tasks, lora_r)
    b_w_avg = avg_B_w(client_B_w, num_tasks, num_B)

    agg_weights = {**a_stacked, **b_stacked, **lambdas_stacked, **b_w_avg}    
    return agg_weights