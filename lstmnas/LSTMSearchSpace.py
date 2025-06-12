from tensorflow.keras.preprocessing.sequence import pad_sequences

class LSTMSearchSpace(object):

    def __init__(self):
        self.vocab, self.vocab_layers, self.vocab_seq, self.vocab_cnn, self.vocab_rnn, self.vocab_units_0, self.vocab_units_1, self.vocab_units_2, self.vocab_direction, self.vocab_state = self.vocab_dict()

    def vocab_dict(self):
        layers_id = []
        rnn_id = []
        units_0_id = []
        units_1_id = []
        units_2_id = []
        direction_id = []  
        state_id = []
        seq_id = []
        cnn_id = []   
        
        # Valores de cada subvocabulario
        layers = [1, 2, 3]  # Número de capas
        rnn = ['lstm', 'gru']
        units_0 = [8, 16, 32, 64, 128, 256, 300, 512, 700, 900, 1024]
        units_1 = [0, 8, 16, 32, 64, 128, 256, 300, 512, 700] # , 900, 1024]
        units_2 = [0, 8, 16, 32, 64, 128, 256, 300, 512, 700]# , 900, 1024]
        direction = ['unidirectional', 'bidirectional']
        state = ['stateless', 'statefull']
        seq = [3, 6, 9, 12]
        cnn = ['vgg16', 'resnet50', 'inceptionV3']
        
        # Generación de índices acumulativos
        # Primero los que no queremos al final
        for i in range(len(layers)):
            layers_id.append(i)  # layers_id = [0, 1, 2]
        
        for i in range(len(rnn)):
            rnn_id.append(i + len(layers_id))
        
        for i in range(len(units_0)):
            units_0_id.append(i + len(layers_id) + len(rnn_id))
        
        for i in range(len(units_1)):
            units_1_id.append(i + len(layers_id) + len(rnn_id) + len(units_0_id))
        
        for i in range(len(units_2)):
            units_2_id.append(i + len(layers_id) + len(rnn_id) + len(units_0_id) + len(units_1_id))
        
        for i in range(len(direction)):
            direction_id.append(i + len(layers_id) + len(rnn_id) + len(units_0_id) + len(units_1_id) + len(units_2_id))
        
        for i in range(len(state)):
            state_id.append(i + len(layers_id) + len(rnn_id) + len(units_0_id) + len(units_1_id) + len(units_2_id) + len(direction_id))
        
        # Ahora los que queremos al final (seq y cnn)
        for i in range(len(seq)):
            seq_id.append(i + len(layers_id) + len(rnn_id) + len(units_0_id) + len(units_1_id) + len(units_2_id) + len(direction_id) + len(state_id))
        
        for i in range(len(cnn)):
            cnn_id.append(i + len(layers_id) + len(rnn_id) + len(units_0_id) + len(units_1_id) + len(units_2_id) + len(direction_id) + len(state_id) + len(seq_id))
        
        # Crear diccionarios
        vocab_layers = dict(zip(layers_id, layers))
        vocab_rnn = dict(zip(rnn_id, rnn))
        vocab_units_0 = dict(zip(units_0_id, units_0))
        vocab_units_1 = dict(zip(units_1_id, units_1))
        vocab_units_2 = dict(zip(units_2_id, units_2))
        vocab_direction = dict(zip(direction_id, direction))
        vocab_state = dict(zip(state_id, state))
        vocab_seq = dict(zip(seq_id, seq))
        vocab_cnn = dict(zip(cnn_id, cnn))
        
        # Diccionario unificado (el orden no importa aquí ya que es un diccionario)
        vocab = {
            **vocab_layers,
            **vocab_rnn,
            **vocab_units_0,
            **vocab_units_1,
            **vocab_units_2,
            **vocab_direction,
            **vocab_state,
            **vocab_seq,
            **vocab_cnn
        }
        
        return (
            vocab,
            vocab_layers,
            vocab_seq,
            vocab_cnn,
            vocab_rnn,
            vocab_units_0,
            vocab_units_1,
            vocab_units_2,
            vocab_direction,
            vocab_state
        )

# class LSTMSearchSpace(object):

#     def __init__(self):
#         self.vocab, self.vocab_layers, self.vocab_seq, self.vocab_cnn, self.vocab_rnn, self.vocab_units_0, self.vocab_units_1, self.vocab_units_2, self.vocab_direction, self.vocab_state = self.vocab_dict()

#     def vocab_dict(self):
#         layers_id = []
#         rnn_id = []
#         units_0_id = []
#         units_1_id = []
#         units_2_id = []
#         direction_id = []  
#         state_id = []
#         seq_id = []
#         cnn_id = []

#         # Valores de cada subvocabulario
#         layers = [1, 2, 3]
#         rnn = ['lstm', 'gru']
#         units_0 = [8, 16, 32, 64, 128, 256, 300, 512, 700, 900, 1024]
#         units_1 = [0, 8, 16, 32, 64, 128, 256, 300, 512, 700]
#         units_2 = [0, 8, 16, 32, 64, 128, 256, 300, 512, 700]
#         direction = ['unidirectional', 'bidirectional']
#         state = ['stateless', 'statefull']
#         seq = [3, 6, 9, 12]
#         cnn = ['vgg16', 'resnet50', 'inceptionV3']

#         # Primero seq y cnn
#         for i in range(len(seq)):
#             seq_id.append(i)

#         for i in range(len(cnn)):
#             cnn_id.append(i + len(seq_id))

#         # Luego el resto
#         for i in range(len(layers)):
#             layers_id.append(i + len(seq_id) + len(cnn_id))

#         for i in range(len(rnn)):
#             rnn_id.append(i + len(seq_id) + len(cnn_id) + len(layers_id))

#         for i in range(len(units_0)):
#             units_0_id.append(i + len(seq_id) + len(cnn_id) + len(layers_id) + len(rnn_id))

#         for i in range(len(units_1)):
#             units_1_id.append(i + len(seq_id) + len(cnn_id) + len(layers_id) + len(rnn_id) + len(units_0_id))

#         for i in range(len(units_2)):
#             units_2_id.append(i + len(seq_id) + len(cnn_id) + len(layers_id) + len(rnn_id) + len(units_0_id) + len(units_1_id))

#         for i in range(len(direction)):
#             direction_id.append(i + len(seq_id) + len(cnn_id) + len(layers_id) + len(rnn_id) + len(units_0_id) + len(units_1_id) + len(units_2_id))

#         for i in range(len(state)):
#             state_id.append(i + len(seq_id) + len(cnn_id) + len(layers_id) + len(rnn_id) + len(units_0_id) + len(units_1_id) + len(units_2_id) + len(direction_id))

#         # Crear diccionarios
#         vocab_seq = dict(zip(seq_id, seq))
#         vocab_cnn = dict(zip(cnn_id, cnn))
#         vocab_layers = dict(zip(layers_id, layers))
#         vocab_rnn = dict(zip(rnn_id, rnn))
#         vocab_units_0 = dict(zip(units_0_id, units_0))
#         vocab_units_1 = dict(zip(units_1_id, units_1))
#         vocab_units_2 = dict(zip(units_2_id, units_2))
#         vocab_direction = dict(zip(direction_id, direction))
#         vocab_state = dict(zip(state_id, state))

#         # Diccionario unificado
#         vocab = {
#             **vocab_seq,
#             **vocab_cnn,
#             **vocab_layers,
#             **vocab_rnn,
#             **vocab_units_0,
#             **vocab_units_1,
#             **vocab_units_2,
#             **vocab_direction,
#             **vocab_state
#         }

#         return (
#             vocab,
#             vocab_layers,
#             vocab_seq,
#             vocab_cnn,
#             vocab_rnn,
#             vocab_units_0,
#             vocab_units_1,
#             vocab_units_2,
#             vocab_direction,
#             vocab_state
#         )



    def encode_sequence(self, sequence):
        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        encoded_sequence = []
        for value in sequence:
            encoded_sequence.append(keys[values.index(value)])
        return encoded_sequence

    def decode_sequence(self, sequence):

        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        
        decoded_sequence = []
        for key in sequence:
            decoded_sequence.append(values[keys.index(key)])
        
        return decoded_sequence
