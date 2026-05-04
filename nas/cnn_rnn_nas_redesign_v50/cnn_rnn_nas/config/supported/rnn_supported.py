RNN_OPERATIONS = ("train", "test", "deploy")
RNN_TYPES = ("lstm", "gru")
RNN_DIRECTIONS = ("unidirectional", "bidirectional")
RNN_MEMORY_MODES = ("none", "carry_forward")
RNN_VIDEO_DECISIONS = ("average", "majority", "max_prob")
RNN_VIDEO_DECISION_INPUTS = ("clip_logits", "clip_embeddings")
RNN_TEST_STRATEGIES = RNN_VIDEO_DECISIONS
