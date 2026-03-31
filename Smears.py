import torch
from torch import Tensor, nn

class SmearGate(nn.Module):
    def __init__(self, dim: int, smear_dim: int | None = None):
        super().__init__()
        self.smear_dim = smear_dim if smear_dim is not None else dim
        self.gate = nn.Parameter(torch.zeros(self.smear_dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)

        if self.smear_dim < x.size(-1):
            x_target = x[..., :self.smear_dim]
            x_smeared = (1 - g) * x_target + g * x_prev[..., :self.smear_dim]
            return torch.cat([x_smeared, x[..., self.smear_dim:]], dim=-1)
        else:
            return (1 - g) * x + g * x_prevù


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class SmearGate2(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    
    def forward(self, x: Tensor) -> Tensor:
        # Match dtype and reshape for broadcasting
        g = torch.sigmoid(self.gate.to(dtype=x.dtype)).view(1, 1, -1)
        
        x_target = x[..., :self.dim]
        
        # FAST SHIFT: Use F.pad instead of allocating zeros and concatenating.
        # This pads the 2nd-to-last dimension (sequence length) with 1 zero at the start.
        x_prev = F.pad(x_target[:, :-1, :], pad=(0, 0, 1, 0))
        
        # FAST MATH: .lerp() is a fused kernel that computes: x_target + g * (x_prev - x_target)
        # This is mathematically identical to (1-g)*x_target + g*x_prev, but faster and saves memory.
        smeared = x_target.lerp(x_prev, g)
        
        # Use .size(-1) instead of .shape[-1] as it is safer for TorchScript/Tracing
        if x.size(-1) > self.dim:
            untouched = x[..., self.dim:]
            return torch.cat([smeared, untouched], dim=-1)
        
        return smeared

class SmearGate3(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2  # Pre-calculate the split point
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    
    def forward(self, x: Tensor) -> Tensor:
        # Match dtype and reshape for broadcasting
        g = torch.sigmoid(self.gate.to(dtype=x.dtype)).view(1, 1, -1)
        
        # Isolate the entire target portion to be smeared
        x_target = x[..., :self.dim]
        
        # Split into the two halves
        x_target1 = x_target[..., :self.half_dim]
        x_target2 = x_target[..., self.half_dim:]
        
        # FAST SHIFT: Pad first, then slice the end. 
        # This guarantees it won't crash even if sequence length is 1.
        # Shift 1st half by 1 token (pad sequence dim by 1 zero at start)
        x_prev1 = F.pad(x_target1, pad=(0, 0, 1, 0))[:, :-1, :]
        
        # Shift 2nd half by 2 tokens (pad sequence dim by 2 zeros at start)
        x_prev2 = F.pad(x_target2, pad=(0, 0, 2, 0))[:, :-2, :]
        
        # Combine the shifted histories back together
        x_prev = torch.cat([x_prev1, x_prev2], dim=-1)
        
        # Single fused math operation for both halves simultaneously
        smeared = x_target.lerp(x_prev, g)
        
        # Handle untouched remainder safely
        if x.size(-1) > self.dim:
            untouched = x[..., self.dim:]
            return torch.cat([smeared, untouched], dim=-1)
        
        return smeared


class Smear3Gate(nn.Module):
    def __init__(self, dim: int, smear_dim: int | None = None):
        super().__init__()
        self.smear_dim = smear_dim if smear_dim is not None else dim
        self.gates = nn.Parameter(torch.zeros(3, self.smear_dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        w = torch.softmax(self.gates.to(dtype=x.dtype), dim=0).unsqueeze(1).unsqueeze(1) # [3, 1, 1, smear_dim]
        
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        x_prev2 = torch.cat([torch.zeros_like(x[:, :2]), x[:, :-2]], dim=1)

        if self.smear_dim < x.size(-1):
            x_smeared = (w[0] * x[..., :self.smear_dim] + 
                         w[1] * x_prev[..., :self.smear_dim] + 
                         w[2] * x_prev2[..., :self.smear_dim])
            return torch.cat([x_smeared, x[..., self.smear_dim:]], dim=-1)
        else:
            return w[0] * x + w[1] * x_prev + w[2] * x_prev2


class Smear4Gate(nn.Module):
    def __init__(self, dim: int, smear_dim: int | None = None):
        super().__init__()
        self.smear_dim = smear_dim if smear_dim is not None else dim
        # 4 logits for [x, x_prev, x_prev2, x_prev3]
        self.gates = nn.Parameter(torch.zeros(4, self.smear_dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        w = torch.softmax(self.gates.to(dtype=x.dtype), dim=0).unsqueeze(1).unsqueeze(1) # [4, 1, 1, smear_dim]
        
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        x_prev2 = torch.cat([torch.zeros_like(x[:, :2]), x[:, :-2]], dim=1)
        x_prev3 = torch.cat([torch.zeros_like(x[:, :3]), x[:, :-3]], dim=1)

        if self.smear_dim < x.size(-1):
            x_smeared = (w[0] * x[..., :self.smear_dim] + 
                         w[1] * x_prev[..., :self.smear_dim] + 
                         w[2] * x_prev2[..., :self.smear_dim] + 
                         w[3] * x_prev3[..., :self.smear_dim])
            return torch.cat([x_smeared, x[..., self.smear_dim:]], dim=-1)
        else:
            return w[0] * x + w[1] * x_prev + w[2] * x_prev2 + w[3] * x_prev3


class NanoSmearGate(nn.Module):
    def __init__(self, dim, gate_dim: int = 12):
        super().__init__()
        self.gate_dim = gate_dim
        self.smear_lambda = nn.Parameter(torch.ones(1))
        # Replace nn.Linear with a simple Parameter vector
        self.weight = nn.Parameter(torch.zeros(gate_dim))

    def forward(self, x: Tensor) -> Tensor:
        # Do the "linear projection" via pointwise multiply and sum.
        # Triton compiles this instantly without cuBLAS overhead.
        gate_in = x[..., :self.gate_dim]
        logits = (gate_in * self.weight).sum(dim=-1, keepdim=True)
        g = self.smear_lambda * torch.sigmoid(logits) 
        
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return x + g * x_prev

class NanoSmear3Gate(nn.Module):
    def __init__(self, gate_dim: int = 12, smear_lambda: float = 1.0):
        super().__init__()
        self.gate_dim = gate_dim
        self.smear_lambda = smear_lambda
        # Computes 2 distinct gate values from the token prefix
        self.smear_gate = nn.Linear(gate_dim, 2, bias=False)
        nn.init.zeros_(self.smear_gate.weight)

    def forward(self, x: Tensor) -> Tensor:
        gate_in = x[..., :self.gate_dim]
        g = self.smear_lambda * torch.sigmoid(self.smear_gate(gate_in)) # [B, T, 2]
        g1, g2 = g[..., 0:1], g[..., 1:2]
        
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        x_prev2 = torch.cat([torch.zeros_like(x[:, :2]), x[:, :-2]], dim=1)
        
        return x + g1 * x_prev + g2 * x_prev2


class NanoSmear4Gate(nn.Module):
    def __init__(self, gate_dim: int = 12, smear_lambda: float = 1.0):
        super().__init__()
        self.gate_dim = gate_dim
        self.smear_lambda = smear_lambda
        # Computes 3 distinct gate values from the token prefix
        self.smear_gate = nn.Linear(gate_dim, 3, bias=False)
        nn.init.zeros_(self.smear_gate.weight)

    def forward(self, x: Tensor) -> Tensor:
        gate_in = x[..., :self.gate_dim]
        g = self.smear_lambda * torch.sigmoid(self.smear_gate(gate_in)) # [B, T, 3]
        g1, g2, g3 = g[..., 0:1], g[..., 1:2], g[..., 2:3]
        
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        x_prev2 = torch.cat([torch.zeros_like(x[:, :2]), x[:, :-2]], dim=1)
        x_prev3 = torch.cat([torch.zeros_like(x[:, :3]), x[:, :-3]], dim=1)
        
        return x + g1 * x_prev + g2 * x_prev2 + g3 * x_prev3