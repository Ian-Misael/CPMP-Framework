import torch
import torch.nn as nn

class LinearInterStackLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # --- PASO 1: Contexto atiende a Stacks (Context <- Stacks) ---
        # Query: Contexto, Key/Value: Stacks
        self.attn_context = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1_context = nn.LayerNorm(d_model)
        
        self.ffn_context = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2_context = nn.LayerNorm(d_model)

        # --- PASO 2: Stacks atienden al Contexto (Stacks <- Context) ---
        # Query: Stacks, Key/Value: Contexto
        self.attn_stacks = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1_stacks = nn.LayerNorm(d_model)

        self.ffn_stacks = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2_stacks = nn.LayerNorm(d_model)

    def forward(self, stacks, context, src_key_padding_mask=None):
        """
        stacks: [B, S_len, d_model]
        context: [B, 1, d_model]
        src_key_padding_mask: [B, S_len] (True para ignorar padding)
        """
        # --- 1. Actualizar el Contexto Global ---
        ctx_attn_out, _ = self.attn_context(
            query=context,
            key=stacks,
            value=stacks,
            key_padding_mask=src_key_padding_mask
        )
        context = context + ctx_attn_out  # Add
        context = self.norm1_context(context)  # Norm
        
        # FFN del Contexto
        ctx_ffn_out = self.ffn_context(context)
        context = context + ctx_ffn_out
        context = self.norm2_context(context)

        # --- 2. Actualizar los Stacks ---
        # Aquí NO necesitamos key_padding_mask porque el contexto (K/V) es de longitud 1 y siempre es válido.
        stacks_attn_out, _ = self.attn_stacks(
            query=stacks,
            key=context,
            value=context
        )
        stacks = stacks + stacks_attn_out  # Add
        stacks = self.norm1_stacks(stacks)  # Norm
        
        # FFN de los Stacks
        stacks_ffn_out = self.ffn_stacks(stacks)
        stacks = stacks + stacks_ffn_out
        stacks = self.norm2_stacks(stacks)

        return stacks, context

import torch
import torch.nn as nn
from models.transformer import Transformer

class CPMPTransformer(Transformer):
    def __init__(self, H_dim, C_dim, X_dim, d_model=64, nhead=8, num_layers=2, ff_dim_multiplier=4, dropout=0.1):
        super().__init__(
            H_dim=H_dim,
            C_dim=C_dim,
            X_dim=X_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim_multiplier=ff_dim_multiplier,
            dropout=dropout
        )
        self.d_model = d_model
        self.H_dim = H_dim
        self.X_dim = X_dim
        self.C_dim = C_dim
        
        self.input_projection = nn.Linear(C_dim, d_model)
        
        # Token CLS: Representará el resumen de la pila
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.intra_stack_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * ff_dim_multiplier, dropout, batch_first=True),
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        self.x_projection = nn.Linear(X_dim, d_model)
        self.fusion_layer = nn.Linear(d_model * 2, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        
        self.global_context_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Reemplazamos el TransformerEncoder por nuestra secuencia de capas lineales
        self.linear_inter_stack_layers = nn.ModuleList([
            LinearInterStackLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=d_model * ff_dim_multiplier, 
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        self.origin_proj = nn.Linear(d_model, d_model)
        self.dest_proj = nn.Linear(d_model, d_model)

    def encode(self, L, X, S, H, memory=None):
        """
        S: (batch_size, S_len, H, C_dim)
        X: (batch_size, S_len, X_dim)
        memory: dict (opcional) {tuple_state: embedding_tensor}
        """
        batch_size, S_len, H_max, C_dim = L.shape
        device = L.device

        # s_mask es [B, S_len], True para stacks válidos, False para padding.
        s_mask = torch.arange(S_len, device=device).expand(batch_size, S_len) < S.unsqueeze(1)
        
        if memory is None:
            memory = {}

        # 1. Identificación de estados para caché
        state_keys = [tuple(s.detach().cpu().numpy().flatten()) for s in L]
        missing_indices = [i for i, key in enumerate(state_keys) if key not in memory]
        
        # Tensor final que vamos a rellenar
        stack_embeddings = torch.zeros((batch_size, S_len, self.d_model), device=device)

        # 2. Si hay estados nuevos, procesarlos por el modelo
        if len(missing_indices) > 0:
            S_to_process = L[missing_indices] # [B', S, H, C]
            X_to_process = X[missing_indices]
            curr_B = S_to_process.shape[0]

            # 1. Preparar Máscara de Padding (True donde hay -1)
            # S_to_process == -1 en todas sus features N
            padding_mask = (S_to_process == -1).all(dim=-1) # [B', S, H]
            
            # 2. Proyección y Reshape
            x = self.input_projection(S_to_process.float()) # [B', S, H, d_model]
            x = x.view(curr_B * S_len, H_max, self.d_model) # [B'*S, H, d_model]
            
            # 3. Añadir CLS Token al inicio de cada secuencia (pila)
            cls_tokens = self.cls_token.expand(curr_B * S_len, 1, -1) # [B'*S, 1, d_model]
            x = torch.cat((cls_tokens, x), dim=1) # [B'*S, H+1, d_model]

            # 5. Máscara de atención para el CLS y contenedores reales
            # El CLS nunca es padding (False). Los contenedores son padding si eran -1.
            cls_mask = torch.zeros((curr_B * S_len, 1), dtype=torch.bool, device=device)
            full_padding_mask = padding_mask.view(curr_B * S_len, H_max)
            full_padding_mask = torch.cat((cls_mask, full_padding_mask), dim=1) # [B'*S, H+1]

            # 6. Intra-stack Attention
            # src_key_padding_mask hace que los -1 no influyan en el softmax
            x_out = self.intra_stack_attention(x, src_key_padding_mask=full_padding_mask)

            # 7. Pooling: Tomamos solo el output de la posición del CLS (índice 0)
            stack_vertical_info = x_out[:, 0, :].view(curr_B, S_len, self.d_model)
            
            # 8. Fusion con X
            x_external_info = self.x_projection(X_to_process)
            combined = torch.cat([stack_vertical_info, x_external_info], dim=-1)
            processed = self.fusion_norm(self.fusion_layer(combined))
            
            # Aplicamos la máscara S a los embeddings recién calculados
            # Los stacks fuera de S se ponen en 0 (o un valor neutral)
            current_s_mask = s_mask[missing_indices].unsqueeze(-1)
            final_embeddings = (processed * current_s_mask).to(torch.float32)
            
            stack_embeddings[missing_indices] = final_embeddings
            
            # Guardar en memoria (siempre en float32)
            for i, original_idx in enumerate(missing_indices):
                memory[state_keys[original_idx]] = final_embeddings[i].detach()

        # 4. Recuperar los que ya estaban en memoria
        existing_indices = [i for i, key in enumerate(state_keys) if key in memory and i not in missing_indices]
        for i in existing_indices:
            stack_embeddings[i] = memory[state_keys[i]].to(device)

        return stack_embeddings, memory

    def decode(self, stack_embeddings, L, X, S, H):
        batch_size, S_len, H_max, C_dim = L.shape
        device = L.device

        # 1. Máscara para la atención Inter-Stack (True = padding / ignorar)
        inter_padding_mask = ~(torch.arange(S_len, device=device).expand(batch_size, S_len) < S.unsqueeze(1))
        
        # Expandimos el vector global al tamaño del batch actual
        # De [1, 1, d_model] a [batch_size, 1, d_model]
        global_ctx = self.global_context_token.expand(batch_size, -1, -1)
        
        # Ejecutamos las capas cíclicamente (num_layers veces)
        for layer in self.linear_inter_stack_layers:
            stack_embeddings, global_ctx = layer(
                stack_embeddings, 
                global_ctx, 
                src_key_padding_mask=inter_padding_mask
            )
            
        # x_global ahora contiene los embeddings de los stacks contextualizados linealmente
        x_global = stack_embeddings
        
        q_origin = self.origin_proj(x_global)
        k_dest = self.dest_proj(x_global)
        
        logits_matrix = torch.matmul(q_origin, k_dest.transpose(-1, -2)) / (self.d_model**0.5)

        # 2. Máscaras de validez de movimiento
        mask_diag = torch.eye(S_len, device=device).bool().unsqueeze(0)
        
        # NUEVA MÁSCARA: Bloquear cualquier stack fuera de S
        # Si el índice de origen >= S o destino >= S, movimiento inválido
        out_of_bounds = inter_padding_mask # [B, S_len] (True donde es inválido)
        mask_out_S_origin = out_of_bounds.unsqueeze(2).expand(-1, -1, S_len)
        mask_out_S_dest = out_of_bounds.unsqueeze(1).expand(-1, S_len, -1)

        # (Lógica anterior de vacío/lleno)
        is_origin_empty = (L == -1).all(dim=-1).all(dim=2) 
        h_idx = (H - 1).view(batch_size, 1, 1, 1).expand(-1, S_len, 1, C_dim)
        target_height_slice = torch.gather(L, 2, h_idx)
        is_dest_full = ~(target_height_slice.squeeze(2) == -1).all(dim=-1) 
        
        mask_origin = is_origin_empty.unsqueeze(2).expand(-1, -1, S_len)
        mask_dest = is_dest_full.unsqueeze(1).expand(-1, S_len, -1)
        
        # Unimos todas las restricciones
        invalid_action_mask = (
            mask_diag | 
            mask_origin | 
            mask_dest | 
            mask_out_S_origin |  # No puedes mover desde un stack que no existe
            mask_out_S_dest      # No puedes mover hacia un stack que no existe
        )
        
        logits_matrix = logits_matrix.masked_fill(invalid_action_mask, -1e4)

        # 3. Aplanar excluyendo la diagonal
        indices = torch.arange(S_len, device=device)
        src_grid = indices.view(-1, 1).repeat(1, S_len)
        dst_grid = indices.view(1, -1).repeat(S_len, 1)
        mask_no_diag = src_grid != dst_grid
        
        logits = logits_matrix[:, mask_no_diag].view(batch_size, -1)
        
        return logits