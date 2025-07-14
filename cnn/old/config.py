class ConfigData():
    def __init__(self,dataset_name=None,data_mode=None,num_clip_frames=None,rescaled_size=None,size=None):
        self.name = dataset_name
        self.mode = data_mode
        self.frames = num_clip_frames
        self.rescaled_size = rescaled_size
        self.size = size
        

class ConfigModelCNN():
    def __init__(self,load_mode=None,device=None,model_architecture=None,config_data=None):
        self.load_mode = load_mode
        self.device = device
        self.model_architecture = model_architecture
        self.model_name ='{}_{}_{}_frames_{}_size'.format(model_architecture,config_data.name,config_data.frames,config_data.size)

class ConfigModelRNN():
    def __init__(self,load_mode=None,seq=None,model=None,direction=None,units=None,device=None,model_architecture=None,config_data=None):
        self.load_mode = load_mode
        self.seq = seq
        self.model = model
        self.direction = direction
        self.units = units
        self.device = device
        self.model_architecture = model_architecture
        self.model_name ='lstm_{}_{}_seq_{}_units_{}_{}_{}_{}_frames_{}_size'.format(model,direction,seq,units,model_architecture,config_data.name,config_data.frames,config_data.size)

class ConfigOp():
    def __init__(self,epochs=None,batch_size=None,learning_rate=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate =learning_rate