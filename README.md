<h1 align="center"> MTL-FLoRA: Federated Low-Rank Adaptation
for Multi-Task Learning </h1> 

<h5 align="center"><em>Joshua Heitbreder, Julia Köpp, Christian Jesinghaus</em>
</br>

## Introduction
We present MTL-FLoRA, a federated extension of MTL-
LoRA for multi-task parameter-efficient fine-tuning of large transformer
models that supports heterogeneous LoRA ranks across clients. Building
on FLoRA’s observation that naive FedAvg over LoRA factors intro-
duces aggregation noise, we lift the MTL-LoRA adapter parameteriza-
tion (A, Λt, Bi, wt) into a federated setting and develop two server-side
aggregation strategies: (I) an extended stacking scheme that aggregates
all adapter tensors, including matrix-valued shared-knowledge weights
wi
t, and (II) an exact client-wise stacking construction with scalar
mixture weights and blockwise normalization. Both strategies stack client
parameters into disjoint rank blocks (block-diagonal Λt and aligned A/B
slices), thereby avoiding cross terms and ensuring that the global update
matches the intended, data-proportional FedAvg update. 
<div style="align: center;">
  <img width="482" height="477" alt="image" src="https://github.com/user-attachments/assets/f8f5a733-2a29-44e6-86b9-35e8ce445a40" />
</div>

## Usage and Reproducing
Code cleanup is still in progress.
