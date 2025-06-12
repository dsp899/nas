import os
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
import random



actionSets = {
        'pmi': ['Drumming', 'PlayingGuitar', 'PlayingPiano', 'PlayingCello', 'PlayingFlute', 'PlayingTabla', 'PlayingViolin', 'PlayingDhol', 'PlayingSitar', 'PlayingDaf'],
        'pmi50': ['Drumming', 'PlayingGuitar', 'PlayingPiano', 'PlayingTabla', 'PlayingViolin'],
        'bm': ['JumpingJack', 'Lunges', 'PullUps', 'PushUps', 'RockClimbingIndoor', 'RopeClimbing', 'Swing', 'TaiChi', 'TrampolineJumping', 'WalkingWithDog', 'BabyCrawling', 'BlowingCandles', 'BodyWeightSquats', 'HandstandPushups', 'HandstandWalking'],
        'bm50': ['JumpingJack', 'Lunges', 'PullUps', 'PushUps', 'RockClimbingIndoor', 'RopeClimbing', 'Swing', 'TaiChi', 'TrampolineJumping', 'WalkingWithDog'],
        'hoi': ['HulaHoop', 'JugglingBalls', 'JumpRope', 'Mixing', 'Nunchucks', 'PizzaTossing', 'SkateBoarding', 'SoccerJuggling', 'YoYo', 'ApplyEyeMakeup', 'ApplyLipstick', 'BlowDryingHair', 'BrushingTeeth', 'CuttingInKitchen', 'Hammering', 'Knitting', 'MoppingFloor', 'ShavingBeard', 'Typing', 'WritingOnBoard'],
        'hoi50': ['HulaHoop', 'JugglingBalls', 'JumpRope', 'Mixing', 'Nunchucks', 'PizzaTossing', 'SkateBoarding', 'SoccerJuggling', 'YoYo'],
        'hhi': ['MilitaryParade', 'SalsaSpin', 'BandMarching', 'Haircut', 'HeadMassage'],
        'hhi50': ['MilitaryParade', 'SalsaSpin'],
        'sports': ['BaseballPitch', 'Basketball', 'BenchPress', 'Biking', 'Billiards', 'BreastStroke', 'CleanAndJerk', 'Diving', 'Fencing', 'GolfSwing', 'HighJump', 'HorseRace', 'HorseRiding', 'JavelinThrow', 'PoleVault', 'PommelHorse', 'Punch', 'Skiing', 'Skijet', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking', 'Archery', 'BalanceBeam', 'BasketballDunk', 'Bowling', 'Boxing-PunchingBag', 'Boxing-Speed Bag', 'CliffDiving', 'CricketBowling', 'CricketShot', 'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch', 'FrontCrawl', 'HammerThrow', 'IceDancing', 'LongJump', 'ParallelBars', 'Rafting', 'Shotput', 'SkyDiving', 'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'TableTennisShot', 'UnevenBars'],
        'sports50': ['BaseballPitch', 'Basketball', 'BenchPress', 'Biking', 'Billiards', 'BreastStroke', 'CleanAndJerk', 'Diving', 'Fencing', 'GolfSwing', 'HighJump', 'HorseRace', 'HorseRiding', 'JavelinThrow', 'PoleVault', 'PommelHorse', 'Punch', 'Skiing', 'Skijet', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking'],
        'all': ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', 'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen', 'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting', 'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo'],
        'all50': ['BaseballPitch', 'Basketball', 'BenchPress', 'Biking', 'Billiards', 'BreastStroke', 'CleanAndJerk', 'Diving', 'Drumming', 'Fencing', 'GolfSwing', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Lunges', 'MilitaryParade', 'Mixing', 'Nunchucks', 'PizzaTossing', 'PlayingGuitar', 'PlayingPiano', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Swing', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing', 'VolleyballSpiking', 'WalkingWithDog', 'YoYo']
    }

class UCF101:

    def __init__(self, name, split):
        self.name = name
        self.actions = actionSets[name]
        self.num_classes = len(self.actions)
        self.videos_dir = os.path.join(os.path.abspath(os.path.curdir), 'data', 'ucf101', 'videos')
        self.names_dir = os.path.join(os.path.abspath(os.path.curdir), 'data', 'ucf101', 'names')
    
        self.train_split, self.test_split = self.process_videos(split)
        
        # Filtrar los datos a las clases seleccionadas
        self.train_set, self.test_set = self.filter_and_create_dataframe()

        print(f"Num videos train: {len(self.train_set)}, test: {len(self.test_set)}")
        # print(f"Test: {self.test_data}")
        self.num_videos_train = len(self.train_set)
        self.num_videos_test = len(self.test_set)

    def process_videos(self, split):
        """
        Crea el diccionario de videos, asigna IDs y divide los datos en entrenamiento/prueba.
        """
        dataDict = {}
        global_id = 0
        local_ids = {class_name: 0 for class_name in self.actions}
        
        # Cargar videos y asignar IDs locales y globales
        for class_id, _class in enumerate(self.actions):
            classDir = os.path.join(self.videos_dir, _class)
            if not os.path.exists(classDir):
                continue
            for dataFile in os.listdir(classDir):
                video_path = os.path.join(classDir, dataFile)
                local_id = local_ids[_class]
                dataDict[video_path] = [class_id, _class, local_id, global_id]
                local_ids[_class] += 1
                global_id += 1
        
        # Dividir en entrenamiento y prueba
        return self.split_data(dataDict, split)

    def split_data(self, dataDict, split):
        """
        Realiza el split de los datos en sets de entrenamiento y prueba basado en archivos de listas.
        """
        types = {
            'split01': ('trainlist01', 'testlist01'),
            'split02': ('trainlist02', 'testlist02'),
            'split03': ('trainlist03', 'testlist03')
        }

        def get_files(file_type):
            return [tuple(line.strip().split(' ')[0].split('/')) for line in open(
                next(os.path.join(self.names_dir, f) for f in os.listdir(self.names_dir) if file_type in f)
            )]

        trainFiles = get_files(types[split][0])
        testFiles = get_files(types[split][1])

        trainDict = {path: data for path, data in dataDict.items() if any(cl in path and act in path for cl, act in trainFiles)}
        testDict = {path: data for path, data in dataDict.items() if any(cl in path and act in path for cl, act in testFiles)}

        return trainDict, testDict
    
    def filter_and_create_dataframe(self):
        """
        Filtra los diccionarios y crea un dataframe reducido solo con las clases seleccionadas.
        """
        def create_dataframe(dataDict):
            return [{'path': video_path, 'id_label': data[0], 'label': data[1], 'id_local_video': data[2], 'id_global_video': data[3]} for video_path, data in dataDict.items()]
        
        train_data = create_dataframe({k: v for k, v in self.train_split.items() if v[1] in self.actions})
        test_data = create_dataframe({k: v for k, v in self.test_split.items() if v[1] in self.actions})
        
        return train_data, test_data

class Frames():

    def __init__(self, split, config, num_classes):
        self.dataset = pd.DataFrame(split)
        self.frames = config.data['frames'] if config.op == 'train' else config.predict_frames
        self.size = (config.data['size'],config.data['size'])
        self.batch = config.cnn['batch']
        self.num_classes = num_classes
        self.generate()


    def format_frames(self, frame, output_size):
        """
            Pad and resize an image from a video.

            Args:
            frame: Image that needs to resized and padded. 
            output_size: Pixel size of the output frame image.

            Return:
            Formatted frame with padding of specified output size.
        """
        frame = tf.image.convert_image_dtype(frame, tf.float32)
        frame = tf.image.resize_with_pad(frame, *output_size)
        return frame


    def frames_from_video_file(self, video_path, n_frames, output_size = (224,224)):
        result = []
        src = cv2.VideoCapture(str(video_path))  
        video_length = int(src.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_step = max(video_length // n_frames, 1)  # Asegura que al menos sea 1
        
        print(f"Video path:{video_path}, Video length: {video_length}, Frame step: {frame_step}")

        for i in range(n_frames):
            frame_index = i * frame_step
            src.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = src.read()
            
            if ret:
                frame = self.format_frames(frame, output_size)
                result.append(frame)
            else:
                # Si no se puede leer el frame, añade uno negro del mismo tamaño
                if result:
                    result.append(np.zeros_like(result[0]))
                else:
                    result.append(np.zeros((output_size[0], output_size[1], 3), dtype=np.float32))

        src.release()
        result = np.array(result)[..., [2, 1, 0]]  # Convertir de BGR a RGB
        return result

    def read(self):
        pairs = list(zip(self.dataset['path'].tolist(), self.dataset['id_label'].tolist(),self.dataset['id_global_video'].tolist()))
        print(f"Num video: {len(pairs)}")
        random.shuffle(pairs)

        for path, label, video_id in pairs:
            video_frames = self.frames_from_video_file(path, self.frames, self.size)
            for frame in video_frames:
                yield frame, tf.one_hot(label, depth=self.num_classes), video_id
    
    def generate(self):
        self.data = tf.data.Dataset.from_generator(self.read,output_signature = (tf.TensorSpec(shape = (None, None, 3), dtype = tf.float32),tf.TensorSpec(shape = (self.num_classes), dtype = tf.int16),tf.TensorSpec(shape = (), dtype = tf.int16)))
        self.data_train = self.data.map(self.filter_output_train).shuffle(5000).prefetch(buffer_size = tf.data.AUTOTUNE).batch(self.batch) 
        self.data_test = self.data.map(self.filter_output_train).prefetch(buffer_size = tf.data.AUTOTUNE).batch(self.batch)
        self.data_predict = self.data.map(self.filter_output_infer).prefetch(buffer_size=tf.data.AUTOTUNE).batch(self.frames)
        

    def filter_output_train(self, frame, label, video_id):
        return frame, label  # Excluir video_id
        
    def filter_output_infer(self, frame, label, video_id):
        return frame, label, video_id

# def to_gif(images):
#     converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
#     imageio.mimsave('./animation.gif', converted_images, fps=10)
#     return webbrowser.open('file://' + os.path.abspath('./animation.gif'))

if __name__ == '__main__':
    pmi50_paths = UCF101('pmi50', 'split01')
    # initialize the batch size and number of steps
    BS = 64
    NUM_STEPS = 1000
    dataset = Frames(split=pmi50_paths.train_set,frames=15,size=299,mode='train',training=True,batch=BS,num_classes=5)
    datasetGen = iter(dataset.data)



