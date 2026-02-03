import numpy as np
import torch
import torch.nn as nn

from model.pytorch.dcrnn_cell import DCGRUCell

device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)]
        )

    def forward(self, inputs, hidden_state=None):
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        return output, torch.stack(hidden_states)


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.horizon = int(model_kwargs.get('horizon', 1))
        self.input_dim = int(model_kwargs.get('decoder_input_dim', self.output_dim))
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        #self.projection_layer = nn.Linear(self.rnn_units*2, self.output_dim)

        # 新增映射层：将 y_extra (32维) -> 64维
        self.y_map = nn.Linear(32, self.rnn_units)

        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type, input_dim=self.input_dim)
             for _ in range(self.num_rnn_layers)]
        )

    def forward(self, inputs, hidden_state=None, y_extra=None):
        hidden_states = []
        output = inputs  # 初始输入 (上一时间步输出)

        # --- RNN多层传播 ---
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state  # 最后一层的输出就是当前步隐藏状态

        #print("output", output.shape)
        '''
        output_reshaped = output.view(-1, self.num_nodes, self.rnn_units)  # [B, N, rnn_units]
        if y_extra is not None:
            y_mapped = self.y_map(y_extra)  # [B, N, rnn_units]
            #print("output nodes sample:", output_reshaped.shape)  # 第0个batch前5个节点
            #print("y_mapped nodes sample:", y_mapped.shape)




            #print("output nodes sample:", output_reshaped[0, :66, 0])  # 第0个batch前5个节点
            #print("y_mapped nodes sample:", y_mapped[0, :66, 0])
            fused = torch.cat([output_reshaped, y_mapped], dim=-1)  # [B, N, rnn_units*2]
        else:
            fused = output_reshaped
        projected = self.projection_layer(fused.view(-1, fused.shape[-1]))  # [B*N, output_dim]
        output = projected.view(-1, self.num_nodes * self.output_dim)  # [B, N*output_dim]
        '''
        # --- y_extra 融合 ---

        #output = output.view(-1, self.num_nodes, self.rnn_units)  # [B, N, rnn_units]
        if y_extra is not None:
            # y_extra: [B, N, 32]
            y_mapped = self.y_map(y_extra)

            #print("output nodes sample:", output[0, :66, 0])  # 第0个batch前5个节点
            #print("y_mapped nodes sample:", y_mapped[0, :66, 0])

            #print("y_mapped",y_mapped.shape)
            y_mapped = y_mapped.view(y_mapped.shape[0], -1)# [B, N*64]
            fused = output + y_mapped
            #print("y_mapped2",y_mapped.shape)# 残差融合
        else:
            fused = output


        #print("y_extra",y_extra.shape)
        # --- 线性映射得到最终输出 ---
        projected = self.projection_layer(fused.view(-1, self.rnn_units))

        #projected = self.projection_layer(fused.view(-1, fused.shape[-1]))  # [B*N, output_dim]
        output = projected.view(-1, self.num_nodes * self.output_dim)



        return output, torch.stack(hidden_states)



class DCRNNModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, logger, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.encoder_model = EncoderModel(adj_mx, **model_kwargs)
        self.decoder_model = DecoderModel(adj_mx, **model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self._logger = logger
        # 额外特征维度
        #self.y_extra_dim = 32  # y的第2~33维

    def encoder(self, inputs):
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)
        return encoder_hidden_state

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):

            if labels is not None:
                y_t = labels[t].view(batch_size, self.num_nodes, -1)
                y_extra = y_t[:, :, 1:33]  # 当前步 y 的 2~33维特征
            else:
                y_extra = torch.zeros((batch_size, self.num_nodes, 32), device=device)

            # 前向传播
            decoder_output, decoder_hidden_state = self.decoder_model(
                decoder_input, decoder_hidden_state, y_extra=y_extra
            )

            # 偶尔打印监控
            #if torch.rand(1).item() < 0.001:
                #print("decoder_input sample[0]:", decoder_input[0, :10])

            #decoder_input = decoder_output  # teacher forcing逻辑保持不变

            # --- 拼接 y_extra ---
            '''
            if labels is not None:
                y_t = labels[t].view(batch_size, self.num_nodes, -1)
                y_extra = y_t[:, :, 1:33]  # teacher forcing 对第 2~33 维
                #print("🔹 labels:", labels.shape)
                #print("🔹 y_t:", y_t.shape)
                #print("🔹 y_extra:", y_extra.shape)
            else:
                y_extra = torch.zeros((batch_size, self.num_nodes, self.y_extra_dim), device=device)
                print("无外部输入，取0")

            #decoder_input_concat = decoder_input.view(batch_size, self.num_nodes * self.output_dim)

            decoder_input_reshaped = decoder_input.view(batch_size, self.num_nodes, 1)
            decoder_input_concat = torch.cat([decoder_input_reshaped, y_extra], dim=-1)
            decoder_input_concat = decoder_input_concat.view(batch_size, self.num_nodes * 33)
            

            # 打印第一条样本的前几个数值看看
            if torch.rand(1).item() < 0.001:  # 控制只偶尔打印，避免太多输出
                print("decoder_input_concat sample[0]:", decoder_input_concat[0, :10])

            # 前向传播
            decoder_output, decoder_hidden_state = self.decoder_model(
                decoder_input_concat, decoder_hidden_state
            )
            '''

            # --- curriculum learning 对第一维控制 ---
            if self.training and self.use_curriculum_learning and labels is not None and batches_seen is not None:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    # 用 ground truth 第一维替换 decoder_input 的第一维
                    decoder_input = labels[t][..., :self.decoder_model.output_dim]
                else:
                    decoder_input = decoder_output
            else:
                decoder_input = decoder_output

            outputs.append(decoder_output)

        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, labels=None, batches_seen=None):
        encoder_hidden_state = self.encoder(inputs)
        self._logger.debug("Encoder complete, starting decoder")
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        self._logger.debug("Decoder complete")
        if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )
        return outputs
