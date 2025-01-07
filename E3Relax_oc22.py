# %%
import math
import torch
from torch import nn
from torch_scatter import scatter
from graph_utils import ScaledSiLU, AtomEmbedding, RadialBasis
from graph_utils import vector_norm

class E3Relax(nn.Module):
    def __init__(
        self,
        hidden_channels=512,
        num_layers=3,
        num_rbf=128,
        cutoff=6.,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        num_elements=83
    ):
        super(E3Relax, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        #### Learnable parameters #############################################

        self.atom_emb = AtomEmbedding(hidden_channels, num_elements)
        self.tags_emb = AtomEmbedding(hidden_channels, 3)
        self.lin_comb = nn.Linear(hidden_channels*2, hidden_channels)

        self.radial_basis = RadialBasis(
            num_radial=num_rbf,
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        self.message_passing_layers = nn.ModuleList()
        self.message_update_layers = nn.ModuleList()
        self.structure_update_layers = nn.ModuleList()

        for i in range(num_layers):
            self.message_passing_layers.append(
                MessagePassing(hidden_channels, num_rbf)
            )
            self.message_update_layers.append(SelfUpdating(hidden_channels))
            self.structure_update_layers.append(StructureUpdating(hidden_channels))

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(self, data):
        pos = data.pos
        cell = data.cell
        
        cell_offsets = data.cell_offsets.float()
        edge_index = data.edge_index

        neighbors = data.neighbors
        z = data.atomic_numbers.long()
        tags = data.tags.long()

        x = self.lin_comb(torch.cat([self.atom_emb(z), self.tags_emb(tags)], dim=-1))
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)
 
        #### Interaction blocks ###############################################
        pos_list = []
        for n in range(self.num_layers):
            j, i = edge_index
            abc_unsqueeze = cell.repeat_interleave(neighbors, dim=0)
            pos_diff = pos[j] + torch.einsum("a p, a p v -> a v", cell_offsets, abc_unsqueeze) - pos[i]
            edge_dist = vector_norm(pos_diff, dim=-1)
            edge_vector = -pos_diff / edge_dist.unsqueeze(-1)
            edge_rbf = self.radial_basis(edge_dist)  # rbf * evelope
            edge_feat = edge_rbf

            dx, dvec = self.message_passing_layers[n](
                x, vec, edge_index, edge_feat, edge_vector
            )

            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

            dx, dvec = self.message_update_layers[n](x, vec)
            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

            pos = self.structure_update_layers[n](x, vec, pos)

            pos_list.append(pos)

        return pos_list

class StructureUpdating(nn.Module):
    def __init__(
        self,
        hidden_channels
    ):
        super(StructureUpdating, self).__init__()

        self.attention_pos = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        self.pos_delta = nn.Linear(hidden_channels, 1, bias=False)

    def forward(self, x, vec, pos):
        # pos attention
        a_pos = torch.softmax(self.attention_pos(x), dim=-1).unsqueeze(1)

        # update coordinate
        pos_delta_double = self.pos_delta(vec).squeeze(-1) + (a_pos * vec).sum(-1)
        pos_new = pos + pos_delta_double / 2

        return pos_new

class MessagePassing(nn.Module):
    def __init__(
        self,
        hidden_channels,
        edge_feat_channels,
    ):
        super(MessagePassing, self).__init__()

        self.hidden_channels = hidden_channels

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, hidden_channels*3),
        )
        self.edge_proj = nn.Linear(edge_feat_channels, hidden_channels*3)

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

    def forward(self, x, vec, edge_index, edge_rbf, edge_vector):
        j, i = edge_index

        rbf_h = self.edge_proj(edge_rbf)

        x_h = self.x_proj(x)
        x_ji1, x_ji2, x_ji3 = torch.split(x_h[j] * rbf_h * self.inv_sqrt_3, self.hidden_channels, dim=-1)

        vec_ji = x_ji1.unsqueeze(1) * vec[j] + x_ji2.unsqueeze(1) * edge_vector.unsqueeze(2)
        vec_ji = vec_ji * self.inv_sqrt_h

        d_vec = scatter(vec_ji, index=i, dim=0, dim_size=x.size(0)) 
        d_x = scatter(x_ji3, index=i, dim=0, dim_size=x.size(0))
        
        return d_x, d_vec
    
class SelfUpdating(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.vec_proj = nn.Linear(
            hidden_channels, hidden_channels * 2, bias=False
        )
        self.xvec_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(self, x, vec):
        vec1, vec2 = torch.split(
            self.vec_proj(vec), self.hidden_channels, dim=-1
        )

        x_vec_h = self.xvec_proj(
            torch.cat(
                [x, torch.sqrt(torch.sum(vec2**2, dim=-2) + 1e-8)], dim=-1
            )
        )
        xvec1, xvec2, xvec3 = torch.split(
            x_vec_h, self.hidden_channels, dim=-1
        )

        gate = torch.tanh(xvec3)
        dx = xvec2 * self.inv_sqrt_2 + x * gate

        dvec = xvec1.unsqueeze(1) * vec1

        return dx, dvec
    

# %%