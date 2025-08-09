import glob
'''
training = glob.glob("/Volumes/StudyNProjects/UnitecFolder/Thesis_Project/WildFireDetection/Training_ds/*.tfrecord")
testing = glob.glob("/Volumes/StudyNProjects/UnitecFolder/Thesis_Project/WildFireDetection/Testing_ds/*.tfrecord")
validation = glob.glob("/Volumes/StudyNProjects/UnitecFolder/Thesis_Project/WildFireDetection/Validation_ds/*.tfrecord")
'''
training = "/Volumes/StudyNProjects/UnitecFolder/Thesis_Project/WildFireDetection/Training_ds/*.tfrecord"
testing = "/Volumes/StudyNProjects/UnitecFolder/Thesis_Project/WildFireDetection/Testing_ds/*.tfrecord"
validation = "/Volumes/StudyNProjects/UnitecFolder/Thesis_Project/WildFireDetection/Evaluation_ds/*.tfrecord"


#Batch Configuration
BATCH_SIZE =128
EPOCHS =100

# Input Features Names
input_features = ['tmmn','NDVI','population','elevation','vs','pdsi','pr','tmmx','sph','th','erc']
prev_fire = 'PrevFireMask'
curr_fire = 'FireMask'
selected_features = ''

# Feature Stats like Min, Max, Mean and std
wildfire_stats = {
    'tmmn': (-444.693, 716.6276, 281.8673, 18.0986),
    'NDVI': (-9567.0, 9966.0, 5297.4717, 2186.9038),
    'FireMask': (-1.0, 1.0, -0.0128, 0.1868),
    'population': (0.0, 27103.605, 29.8874, 214.3112),
    'elevation': (-45.0, 4203.0, 904.5699, 846.5071),
    'vs': (-82.6531, 103.2201, 3.6543, 1.3117),
    'pdsi': (-152.9079, 80.9965, -0.74, 2.4769),
    'pr': (-167.4483, 136.8156, 0.3348, 1.5888),
    'tmmx': (0.0, 1229.8488, 297.742, 19.0823),
    'sph': (-0.129, 0.0855, 0.0065, 0.0037),
    'th': (-505870.1, 37735.63, 154.127, 3163.1426),
    'PrevFireMask': (-1.0, 1.0, -0.0029, 0.1408),
    'erc': (-1196.0886, 2470.8823, 53.6251, 25.2632)
}
