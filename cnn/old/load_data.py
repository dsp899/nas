import os
import cv2
import random
import numpy as np
import time
import utils

class ucf101():

    actionSets = dict({'pmi':['Drumming','PlayingGuitar','PlayingPiano','PlayingCello','PlayingFlute','PlayingTabla','PlayingViolin','PlayingDhol','PlayingSitar','PlayingDaf'],
                    'pmi50': ['Drumming','PlayingGuitar','PlayingPiano','PlayingTabla','PlayingViolin'],
                    'bm':['JumpingJack','Lunges','PullUps','PushUps','RockClimbingIndoor','RopeClimbing','Swing','TaiChi','TrampolineJumping', 'WalkingWithDog','BabyCrawling','BlowingCandles','BodyWeightSquats','HandstandPushups','HandstandWalking'],
                    'bm50':['JumpingJack','Lunges','PullUps','PushUps','RockClimbingIndoor','RopeClimbing','Swing','TaiChi','TrampolineJumping', 'WalkingWithDog'],
                    'all':['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', 'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen', 'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting', 'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo'],
                    'all50':['BaseballPitch', 'Basketball',  'BenchPress', 'Biking', 'Billiards', 'BreastStroke', 'CleanAndJerk', 'Diving', 'Drumming', 'Fencing', 'GolfSwing', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Lunges', 'MilitaryParade', 'Mixing',  'Nunchucks', 'PizzaTossing', 'PlayingGuitar', 'PlayingPiano', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Swing', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing', 'VolleyballSpiking', 'WalkingWithDog', 'YoYo']})
    num_classes = None


    def __init__(self, name, split):
        self.name = name
        self.actions = self.actionSets[name]
        self.num_classes = len(self.actions)
        self.videos_dir = '{}/{}/{}/{}'.format(os.path.abspath(os.path.curdir),'data','ucf101','videos')
        self.names_dir = '{}/{}/{}/{}'.format(os.path.abspath(os.path.curdir),'data','ucf101','names')

        dataDict = self.create_datapaths()
        trainDict, testDict = self.split_datapaths(dataDict,split=split)
        self.reduct_datapaths(trainDict, testDict)

    def create_datapaths(self):
        dataDict = dict()
        for index, _class in enumerate(os.listdir(self.videos_dir)):
            classDir = os.path.join(self.videos_dir,_class)
            dataFiles = os.listdir(classDir)
            dataPaths = [os.path.join(classDir,dataFile) for dataFile in dataFiles]
            dataDict[_class] = dataPaths
        return dataDict

    def create_split_datapaths(self, dataDict, files):
        auxdict = {}
        for clave, valor in files:
            for data in dataDict[clave]:
                if clave in dataDict and valor in data:
                    if clave not in auxdict:
                        auxdict[clave] = []
                    auxdict[clave].append(data)
        return auxdict

    def split_datapaths(self, dataDict, split):
        types = dict({'split01': ('trainlist01','testlist01'),'split02': ('trainlist02','testlist02'),'split03': ('trainlist03','testlist03')})
        trainPath = [os.path.join(self.names_dir,_file) for _file in os.listdir(self.names_dir) if types[split][0] in _file] # filepath with filenames associated to train
        testPath = [os.path.join(self.names_dir,_file) for _file in os.listdir(self.names_dir) if types[split][1] in _file] # filepath with filenames associated to test
        with open(trainPath[0], 'r') as file:
            trainFiles = [line for line in file]
            trainSplit = [tuple(trainFile.strip().split(' ')[0].split('/')) for trainFile in trainFiles]
        with open(testPath[0], 'r') as file:
            testFiles = [line for line in file]
            testSplit = [tuple(testFile.strip().split('/')) for testFile in testFiles]
        trainDict = self.create_split_datapaths(dataDict, trainSplit)
        testDict = self.create_split_datapaths(dataDict, testSplit)
        return trainDict, testDict

    def reduct_datapaths(self, trainDict, testDict):
        testDictReduct = {_class: testDict[_class] for _class in self.actions}
        trainDictReduct = {_class: trainDict[_class] for _class in self.actions}
        self.test = [(_file,key) for key, value in testDictReduct.items() for _file in value]
        self.train = [(_file,key) for key, value in trainDictReduct.items() for _file in value]
        random.shuffle(self.test)
        random.shuffle(self.train)
        print("Num videos train: {}, test: {}".format(len(self.train), len(self.test)))


    def load_data(self, video_paths, num_clip_frames, rescaled_size, crop_size):
        
        total_check = False
        video_index = 0
        frames_total = []
        labels_total = []
        videos_total = []
        total_size_bytes = 0
        total_frames_count = 0
        num_cortos = 0
        num_cortos_aux = 0
        while (video_index < len(video_paths)):
            frame_video_count = 0
            video_size_bytes = 0
            frames_video = []
            labels_video = []
            video_id = []
            cap = cv2.VideoCapture(video_paths[video_index][0])
            num_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_freq = num_video_frames // num_clip_frames
            if num_video_frames < num_clip_frames:
                num_cortos +=1
                print("Video demasiado corto nº {}, total frames del video: {}, clase: {}".format(num_cortos, num_video_frames,video_paths[video_index][1]))
                video_index += 1
                cap.release()
                continue

            for i in range(num_video_frames):
                ret, frame = cap.read()
                if (type(frame) != type(None)):
                    if ((i%sample_freq == 0) and (frame_video_count < num_clip_frames)):
                        rescaled_frame = cv2.resize(frame, (rescaled_size, rescaled_size))
                        height, width, _ = rescaled_frame.shape
                        #crop_size = size
                        x = width/2 - crop_size/2
                        y = height/2 - crop_size/2
                        cropped_frame = np.array(rescaled_frame[int(y):int(y+crop_size), int(x):int(x+crop_size)])
                        #print(cropped_frame.shape)
                        #rescaled_frame = cv2.resize(frame, (height, width))
                        #normalized_frame = cv2.normalize(rescaled_frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        video_size_bytes +=  cropped_frame.nbytes
                        frames_video.append(cropped_frame)
                        labels_video.append(video_paths[video_index][1])
                        video_id.append(video_index)
                        frame_video_count += 1
                else:
                    print("He encontrado un frame roto")
            if  frame_video_count < num_clip_frames:
                num_cortos_aux += 1
                print("Video demasiado corto tras carga nº {}, total frames del video: {}, clase: {}".format(num_cortos_aux, frame_video_count,video_paths[video_index][1]))
                video_index += 1   
                cap.release()             
                continue
            cap.release()
            total_frames_count += len(frames_video)
            frames_total.append(frames_video)
            labels_total.extend(labels_video)
            videos_total.extend(video_id)
            total_size_bytes +=  video_size_bytes
            total_check = (total_frames_count % num_clip_frames) == 0
            
            print("Video_index: {}/{}, size: {:.2f}, frames: {}, check: {}".format(video_index+1, len(video_paths), (total_size_bytes/(1024**3)),  total_frames_count, total_check))
            video_index += 1

        frames_total = np.concatenate(frames_total, axis=0)
        print(frames_total.shape)
        return frames_total, labels_total, videos_total


    def create_labelDict(self):
        labelspath = [os.path.join(self.names_dir,_file) for _file in self.names_dir if 'class' in _file]
        labelsDict = {action: i for i, action in enumerate(self.actions)}
        return labelsDict

    def labels_string_to_int(self, labels):
        labelsDict = self.create_labelDict()
        labels = [(int(labelsDict[label])) for label in labels]
        return labels

    def labels_int_to_onehot(self, labels):
        labels = np.eye(len(self.actions))[labels]
        return labels

    def get_data(self, mode, num_clip_frames, rescaled_size, crop_size):
        data_dirname = 'data/ucf101_frames/{}'.format(self.name)
        utils.create_directory(data_dirname)
        dataset_name = '{}_{}_frames_{}_size'.format(self.name,num_clip_frames,crop_size)
        if(mode == 'train'):
            frames_train, labels_train, videos_id_train  = self.load_data(self.train, num_clip_frames, rescaled_size, crop_size) #List of tuples [(frame,label_string),(frame2,label_string)...]
            print("Frames of train dataset: {}, labels of train dataset: {}".format(len(frames_train), len(labels_train)))                        
            labels_train = self.labels_string_to_int(labels_train)
            labels_train = self.labels_int_to_onehot(labels_train)
            np.save('{}/frames_train_{}.npy'.format(data_dirname,dataset_name),frames_train)
            np.save('{}/labels_train_{}.npy'.format(data_dirname,dataset_name),labels_train)
            np.save('{}/videos_id_train_{}.npy'.format(data_dirname,dataset_name),videos_id_train)

            #return (frames_train, labels_train)
        elif(mode == 'test'):
            frames_test, labels_test, videos_test = self.load_data(self.test, num_clip_frames, rescaled_size, crop_size)
            print("Frames of test dataset: {}, labels of train dataset: {}".format(len(frames_test), len(labels_test)))
            labels_test = self.labels_string_to_int(labels_test)
            labels_test = self.labels_int_to_onehot(labels_test)
            np.save('{}/frames_test_{}.npy'.format(data_dirname,dataset_name),frames_test)
            np.save('{}/labels_test_{}.npy'.format(data_dirname,dataset_name),labels_test)
            np.save('{}/videos_id_test_{}.npy'.format(data_dirname,dataset_name),videos_test)
            #return (frames_test, labels_test)
        else:
            data_train = self.load_data(self.train, num_clip_frames, rescaled_size, crop_size) #List of tuples [(frame,label_string),(frame2,label_string)...]
            data_test = self.load_data(self.test, num_clip_frames, rescaled_size, crop_size)
            print("Frames of train dataset: {}".format(len(data_train)))
            print("Frames of test dataset: {}".format(len(data_test)))
            frames_train, labels_train = zip(*data_train) #Two seperate lists [frame1,frame2...], [label1__string,label2_string...]
            frames_test, labels_test = zip(*data_test)
            frames_train = np.array(frames_train)
            frames_test = np.array(frames_test)
            
            labels_train = self.labels_string_to_int(labels_train)
            labels_test = self.labels_string_to_int(labels_test)
            labels_train = self.labels_int_to_onehot(labels_train)
            labels_test = self.labels_int_to_onehot(labels_test)
            np.save('{}/frames_train_{}.npy'.format(data_dirname,dataset_name),frames_train)
            np.save('{}/labels_train_{}.npy'.format(data_dirname,dataset_name),labels_train)
            np.save('{}/frames_test_{}.npy'.format(data_dirname,dataset_name),frames_test)
            np.save('{}/labels_test_{}.npy'.format(data_dirname,dataset_name),labels_test)
            
            #data_train = (frames_train, labels_train)
            #data_test = (frames_test, labels_test)
            #return (data_train,data_test)



class ucf101_frames():
    def __init__(self,name, frames, size):
        self.data_dir = '{}/{}/{}/{}'.format(os.path.abspath(os.path.curdir),'data','ucf101_frames',name)
        self.dataset_name = '{}_{}_frames_{}_size'.format(name,frames,size)
       
    def get_data(self, mode):
        if(mode == 'train'):
            x_train = np.load('{}/frames_train_{}.npy'.format(self.data_dir,self.dataset_name))
            y_train = np.load('{}/labels_train_{}.npy'.format(self.data_dir,self.dataset_name))
            videos_id_train = np.load('{}/videos_id_train_{}.npy'.format(self.data_dir,self.dataset_name))
            self.num_classes = y_train.shape[1]
            return (x_train, y_train, videos_id_train)
        elif(mode == 'test'):            
            x_test = np.load('{}/frames_test_{}.npy'.format(self.data_dir,self.dataset_name))
            y_test = np.load('{}/labels_test_{}.npy'.format(self.data_dir,self.dataset_name))
            videos_id_test = np.load('{}/videos_id_test_{}.npy'.format(self.data_dir,self.dataset_name))
            self.num_classes = y_test.shape[1]
            return (x_test, y_test, videos_id_test)
        else:
            x_train = np.load('{}/frames_train_{}.npy'.format(self.data_dir,self.dataset_name))
            y_train = np.load('{}/labels_train_{}.npy'.format(self.data_dir,self.dataset_name))
            videos_id_train = np.load('{}/videos_id_train_{}.npy'.format(self.data_dir,self.dataset_name))
            x_test = np.load('{}/frames_test_{}.npy'.format(self.data_dir,self.dataset_name))
            y_test = np.load('{}/labels_test_{}.npy'.format(self.data_dir,self.dataset_name))
            videos_id_test = np.load('{}/videos_id_test_{}.npy'.format(self.data_dir,self.dataset_name))
            data_train = (x_train, y_train, videos_id_train)
            data_test = (x_test, y_test, videos_id_test)
            self.num_classes = y_train.shape[1]
            return (data_train,data_test)

    def shuffle_data(self, mode, data):
        if(mode == 'train'):
            frames_train, labels_train, videos_id_train = data
            data_train_index = [i for i in range(len(frames_train))]
            random.shuffle(data_train_index) 
            frames_train = np.array([frames_train[i] for i in data_train_index])
            labels_train = np.array([labels_train[i] for i in data_train_index])
            videos_id_train = np.array([videos_id_train[i] for i in data_train_index])
            return (frames_train, labels_train, videos_id_train)
        elif(mode == 'test'):
            frames_test, labels_test, videos_id_test = data            
            data_test_index = [i for i in range(len(frames_test))]
            random.shuffle(data_test_index)
            frames_test = np.array([frames_test[i] for i in data_test_index])
            labels_test = np.array([labels_test[i] for i in data_test_index])
            videos_id_test = np.array([videos_id_test[i] for i in data_test_index])

            return (frames_test, labels_test, videos_id_test)
        else:
            data_train, data_test = data
            frames_train, labels_train, videos_id_train = data_train
            frames_test, labels_test, videos_id_test = data_test
            data_train_index = [i for i in range(len(frames_train))]
            data_test_index = [i for i in range(len(frames_test))]
            random.shuffle(data_train_index) 
            random.shuffle(data_test_index)
            frames_train = np.array([frames_train[i] for i in data_train_index])
            labels_train = np.array([labels_train[i] for i in data_train_index])
            videos_id_train = np.array([videos_id_train[i] for i in data_train_index])
            frames_test = np.array([frames_test[i] for i in data_test_index])
            labels_test = np.array([labels_test[i] for i in data_test_index])
            videos_id_test = np.array([videos_id_test[i] for i in data_test_index])
            data_train = (frames_train, labels_train, videos_id_train)
            data_test = (frames_test, labels_test, videos_id_test)
            return (data_train,data_test)            

class ucf101_features():
    
    def __init__(self,model_architecture,model_name):
        self.data_dir = '{}/{}/{}'.format(os.path.abspath(os.path.curdir),'data','ucf101_features')
        self.features_dir = '{}/{}/{}'.format(self.data_dir,model_architecture,model_name)
        self.arch = model_architecture
        self.name = model_name

    def get_data(self):
        features_train = np.load('{}/features_train_{}.npy'.format(self.features_dir,self.name))
        features_test = np.load('{}/features_test_{}.npy'.format(self.features_dir,self.name))
        labels_train = np.load('{}/labels_train_{}.npy'.format(self.features_dir,self.name))
        labels_test = np.load('{}/labels_test_{}.npy'.format(self.features_dir,self.name))
        videos_id_train = np.load('{}/features_id_video_train_{}.npy'.format(self.features_dir,self.name))
        videos_id_test = np.load('{}/features_id_video_test_{}.npy'.format(self.features_dir,self.name))        
        print(features_train.shape)
        print(labels_train.shape)
        print(features_test.shape)
        print(labels_test.shape)
        data_train = (features_train, labels_train, videos_id_train)
        data_test = (features_test, labels_test, videos_id_test)
        self.num_classes = labels_test.shape[-1]
        return (data_train, data_test), features_train.shape

    def create_features_window(self, data, seq_length):
        features, labels, videos_id = data
        print(type(videos_id))
        print(labels.shape)
        print(videos_id.shape)
        features = np.reshape(features,[-1,seq_length,features.shape[1], features.shape[2], features.shape[3]])
        labels = np.reshape(labels,[-1,seq_length,labels.shape[1]])
        videos_id = np.reshape(videos_id,[-1,seq_length,1])

        x = np.array(features)
        y = np.array(labels)
        z = np.array(videos_id)
        print(x.shape)
        print(y.shape)
        print(z.shape)
        return (x, y, videos_id)

    def shuffle_data(self, mode, data):
        if(mode == 'train'):
            frames_train, labels_train, videos_id_train = data
            data_train_index = [i for i in range(len(frames_train))]
            random.shuffle(data_train_index) 
            frames_train = np.array([frames_train[i] for i in data_train_index])
            labels_train = np.array([labels_train[i] for i in data_train_index])
            return (frames_train, labels_train, videos_id_train)
        elif(mode == 'test'):
            frames_test, labels_test, videos_id_test = data            
            data_test_index = [i for i in range(len(frames_test))]
            random.shuffle(data_test_index)
            frames_test = np.array([frames_test[i] for i in data_test_index])
            labels_test = np.array([labels_test[i] for i in data_test_index])
            return (frames_test, labels_test, videos_id_test)
        else:
            data_train, data_test = data
            frames_train, labels_train, videos_id_train = data_train
            frames_test, labels_test, videos_id_test = data_test
            data_train_index = [i for i in range(len(frames_train))]
            data_test_index = [i for i in range(len(frames_test))]
            random.shuffle(data_train_index) 
            random.shuffle(data_test_index)
            frames_train = np.array([frames_train[i] for i in data_train_index])
            labels_train = np.array([labels_train[i] for i in data_train_index])
            frames_test = np.array([frames_test[i] for i in data_test_index])
            labels_test = np.array([labels_test[i] for i in data_test_index])
            data_train = (frames_train, labels_train, videos_id_train)
            data_test = (frames_test, labels_test, videos_id_test)
            return (data_train,data_test)
