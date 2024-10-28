!git clone https://github.com/aiunderstand/ColabTest.git
!mkdir '/content/ColabTest/alldata_preprocessed'
!mkdir '/content/ColabTest/onnx_models'

# see https://developers.google.com/machine-learning/data-prep/transform/normalization. Normalization before training/eval is important as the different features have different scales resulting in different weighting
# Another good normalization technique is z-score which transform features to zero mean, 1 std deviation distribution
# Do we perhaps need to clip data set of outliers and why (use the data visualisation of step 3)
# Do we need to inter or extrapolate data points to fill in the blanks?
# Do we need stratifed splits? https://www.reddit.com/r/learnmachinelearning/comments/tw8ec5/meaning_of_the_argument_stratify_in_train_test/
# See https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html for tips on multi-column normalisation

### since I saw some negative datapoints, perhaps choose to normalize between -1 and +1. There are some ternary ML papers that we can then reference as this is the optimal quantization.
# note that the normalize function does not quantize to ternary integers and keeps floats.
%pip install tensorboard-plugin-customizable-plots # using https://github.com/abdeladim-s/tensorboard_plugin_customizable_plots since tensorboard has no features to add axis labels and titles
%load_ext tensorboard
%pip install torcheval
%pip install torchinfo
%pip install onnx

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.onnx as onnx_exporter

import random
import math

from torcheval.metrics import BinaryConfusionMatrix
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.utils.tensorboard import SummaryWriter # tensorboard data logger
from torchinfo import summary # show model architecture

writer = SummaryWriter()
sc = MinMaxScaler(feature_range=(-1, 1)).set_output(transform="pandas")
lb = LabelBinarizer()


directory_path = '/content/ColabTest/alldata' #contains data with derived features, not raw features
directory_path_preprocessed = '/content/ColabTest/alldata_preprocessed'
directory_path_onnx_models = '/content/ColabTest/onnx_models' #contains exported onnx models
directory_files = os.listdir(directory_path)
directory_files = sorted(directory_files)
df_files = [[],[],[],[],[],[]] # there are 6 passwords
df_blobs_processed = []

"""
0 - Gigab-exclm-t R3ceiver
1 - Ob-dollar-erv3r
2 - flying automatic monster
3 - gigabit receiver
4 - observer
5 - repetition learn machine thinker
"""

pwd_id = 0 # password id from 0-5 , 4 = "observer"
p_id = 0 # participant id from 0-4, 0 = participant1



batch_size = 1 # Try for several values and see which give best performance for our hardware setup and dataset

debug = True

# Set the random seed manually for reproducibility
torch.manual_seed(42)

# Check if GPU has cuda compatibility
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#set eval/test metric
metric = BinaryConfusionMatrix(device=device) #bug in threshold function, need to cast to int, so do threshold manually



#for each file in directory group them per password
for i in range (len(directory_files)):
  #only parse csv files
  if directory_files[i].endswith(".csv"):
    #read csv into pandas dataframe, first csv row is header which is skipped automatically
    df_file = pd.read_csv(os.path.join(directory_path, directory_files[i]))

    #remove first 2 columns if attack_set
    if directory_files[i].startswith("attack"):
      df_file = df_file.iloc[:,2:]

    #add file to dataset of the same password
    df_files[i %6].append(df_file)

    if (i==pwd_id and debug):
      print ("shape df_file: " , df_file.shape)



#for each password normalize columns
for i in range(len(df_files)):
  #create one big table by stacking the rows and columns of each file
  df_blob = pd.concat(df_files[i], axis=0)

  #for each column run the MIN/MAX scaler between -1 and 1
  df_blob_processed =sc.fit_transform(df_blob)
  df_blobs_processed.append(df_blob_processed)

  if (i==pwd_id and debug):
    print ("shape df_blob: " , df_blob.shape)

#restore individual dataset and save data in ColabTest\alldata_preprocessed. Removing the index is important otherwise the column is included in the trainingsdata


for i in range (len(directory_files)):
  if directory_files[i].endswith(".csv"):
    # retrieve the rows and columns data from the preprocessed data blob and save them
    participant_set_size = 200
    attack_set_size = 340

    # since not all datasets have equal rows, determine which is which
    if (i%6 == 3) or (i%6 == 4):
      attack_set_size = 350

    # determine the section of the blob where the contents of the file is
    start_row = 0

    # the first 6 files are attack set and have different sizes
    if i // 6 > 0:
      start_row = int(attack_set_size + ((i // 6) -1) * participant_set_size )

    end_row = int(attack_set_size + (i // 6) * participant_set_size)

    # retrieve the file from the blow
    df_file = df_blobs_processed[i % 6].iloc[start_row:end_row]

    # save it to the colab storage
    df_file.to_csv(os.path.join(directory_path_preprocessed, directory_files[i]), index=False)

    if (i == pwd_id and debug):
      print ("shape df_file_proc: " , df_file.shape)

# The goal in this paper is password authentication. The use case is a user that types in its username and password as normal. The verification of the password is as normal, for example a hashed and salted password is match. However in addition keyboard dynamic authenticaion is done to AFTER a username/password match. This means that the dataset does not need negative examples of wrong passwords or need to predict the user with a partial password. Only the type timings of the full password is considered. Inherent to the timings is the spatial layout of the keyboard and physical muscle memory and ability of the user and lastly the experience with the password. Other effects like intent to deliberatly type slow or off is ignored. The model is unaware of the actual keys pressed and only receives a timing sequence. This means that the timing sequence depend on the length of the password. With regards to splitting the dataset: in theory can the evaluation set be reduced to only contain participants (POSITIVE EXAMPLES), however with a near perfect score (regardless of small or big dataset) one might wonder if the model can recognize true negatives. Hence both training and eval sets needs positive and negative examples in balanced and normalized fashion.
# The model thus receives a input vector containing a single key press, release and predicts the result after each. If the model architecture is a many-to-one then the result is single node and the continuous variable of that node is then rounded to the closest label, either 0 or 1. If the architecture is many-to-many then each node has a value and the heighest value node is the result with the highest "confidence".
# Linked to the username is a password model that is trained on the input of that user (POSITIVE EXAMPLES) and that of an attack_set (NEGATIVE EXAMPLES). The attack_set is a different dataset and does not contain other user (ie "particpant") data.
# The dataset contains CSV files of 5 participants plus attack_set of each 6 passwords. Since we are interested in password models of participants only (not the attack_set) we need to generate 5 participants * 6 passwords = 30 models.

#https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
def load_dataset(path, label):
    #read CSV into pandas dataframe
    input = pd.read_csv(path)

    labels = np.full(len(input),label)

    return input.to_numpy(), labels

def to_onehot(label_array):
  lb.fit(range(max(label_array)+1))
  enc_data = lb.transform(label_array)
  #enc_data = np.eye(max(label_array)+1)[label_array]   # useful if you need one hot vectors for 2 dimensions which are normally reduced to one
  return enc_data

#important the os.listdir provides unsorted output
directory_files_preprocessed = os.listdir(directory_path_preprocessed)
directory_files_preprocessed = sorted(directory_files_preprocessed)

passwordorder =["Gigab-exclm-t R3ceiver",
                  "Ob-dollar-erv3r",
                  "flying automatic monster",
                  "gigabit receiver",
                  "observer.csv",
                  "repetition learn machine thinker"]
"""
0 - Gigab-exclm-t R3ceiver
1 - Ob-dollar-erv3r
2 - flying automatic monster
3 - gigabit receiver
4 - observer
5 - repetition learn machine thinker
"""

pwd_id = 0 # password id from 0-5 , 4 = "observer"
p_id = 0 # participant id from 0-4, 0 = participant1

k = 5 # k-fold validation


resultsforpaper = [0] * 30
resultscounter = -1

for pwd_id in range(6):
  for p_id in range(5):
    resultscounter += 1
    print("-------------")
    print("-------------")
    print("-------------")
    print("pwd, participant: ", pwd_id, p_id)



    files_attack =["attack - Gigab-exclm-t R3ceiver.csv",
                  "attack - Ob-dollar-erv3r.csv",
                  "attack - flying automatic monster.csv",
                  "attack - gigabit receiver.csv",
                  "attack - observer.csv",
                  "attack - repetition learn machine thinker.csv"]

    #add participant input with label 0
    _inputP, _labelsP = load_dataset(os.path.join(directory_path_preprocessed, directory_files_preprocessed[6 + 6*p_id + pwd_id]), p_id) #directory_files_preprocessed[i]),(i // 6) -1)

    #add attack_set input with label 1 from the same password
    _inputA, _labelsA = load_dataset(os.path.join(directory_path_preprocessed, files_attack[pwd_id]), 5)

    #remove 1 feature (the enter key) from the input column
    _inputP = np.delete(_inputP, _inputP.shape[1] -1, 1)
    _inputA = np.delete(_inputA, _inputA.shape[1] -1, 1)

    #convert to numpy
    _inputA = np.array(_inputA)
    _labelsA = np.array(_labelsA)   # these labels aren't actually used... I make a new array and just fill it with 0 or 1 later in the code when making the training and test set...
    _inputA = np.array(_inputA)
    _labelsA = np.array(_labelsA)   # not used


    print("size of _inputA: ", len(_inputA))
    print("shape of _inputA: ", _inputA.shape)
    print("size of _inputP: ", len(_inputP))
    print("shape of _inputP: ", _inputP.shape)

    # attack comes in groups of 10 from each attack participant, and no data point from one participant must be in both testing and training
    # attack and defence sets need to be split into 5 groups where one is used for testing and the rest for training
    # that means that the number of attack data points in each group must be divisible by 10.
    # which means the size of the attack set in the groups vary in size by 10, since the number of attack participants might not be divisible by k=5
    # this also means the training/test split will be an AVERAGE of 80/20 during k-fold, but may vary a little bit between the 5 folds..
    # the number of datapoints from each participant is 200 which is divisible by k=5


    # Split into groups of 10 lines (one group per attack participant)
    tensplit_inputA = np.stack(np.array_split(_inputA, len(_inputA)/10))  #splits into nr of attack participants
    #print("Shape of tensplit_inputA:", tensplit_inputA.shape)
    np.random.shuffle(tensplit_inputA)  # shuffles by ATTACK PARTICIPANT, not by line
    # Split into 5 groups, then merge the attack parcitipants in each group
    #tenfivesplit_inputA = np.stack(np.array_split(tensplit_inputA, 5))  #splits into 5 groups (does not work)
    k_inputA = np.array_split(tensplit_inputA, k)  # This gives a LIST of 2D arrays.
                                                              #The 2D arrays can vary in length since the number of attack participants are not necessarily dibisible by 5
    # merging the participants in each k group
    for i, arr in enumerate(k_inputA):
      #print(f"FOOBAR Shape of array {i+1}: {arr.shape}")
      k_inputA[i] = k_inputA[i].reshape(-1, k_inputA[i].shape[-1])
      #print(f"FOOBAR NEW Shape of array {i+1}: {k_inputA[i].shape}")

    #randomize the order of the defence data
    np.random.shuffle(_inputP)
    k_inputP = np.array_split(_inputP, k) #k_inputP



    #K-FOLD VALIDATION

    for k_nr in range(k): #K-FOLD VALIDATION IN THIS LOOP
      torch.manual_seed(42 + k_nr)

      print("@@@")
      print("K-fold nr ", k_nr)
      #create the training and test sets with labels for this iteration of k-fold
      
      
      _trainingset = np.empty(0)
      _traininglabels = np.empty(0)
      _testset = np.empty(0)
      _testlabels = np.empty(0)
      for i in range(5):
        attacklabel_array = np.full(len(k_inputA[i]), 5, dtype=int)       #array of the value 5 (the label for attack), the length of the number of attack datapoints in the i k-fold group
        defencelabel_array = np.full(len(k_inputP[i]), p_id, dtype=int)   #array of the value p_id (the label for defence), the length of the number of defence datapoints in the i k-fold group
        if i == k_nr:
          _testset =  np.concatenate((k_inputA[i], k_inputP[i]))
          _testlabels = np.concatenate((attacklabel_array, defencelabel_array))
        else:
          if len(_trainingset) == 0:
            _trainingset = np.concatenate((k_inputA[i], k_inputP[i]))
            _traininglabels = np.concatenate((attacklabel_array, defencelabel_array))
          else:
            _trainingset = np.concatenate((_trainingset, k_inputA[i], k_inputP[i]))
            _traininglabels = np.concatenate((_traininglabels, attacklabel_array, defencelabel_array))

      # Shuffle training data and labels
      num_samples = len(_traininglabels)
      indices = np.random.permutation(num_samples)
      _trainingset = _trainingset[indices]
      _labels_train = _traininglabels[indices]

      # Shuffle test data and labels
      num_samples = len(_testlabels)
      indices = np.random.permutation(num_samples)
      _testset = _testset[indices]
      _labels_eval = _testlabels[indices]

      #print("size of training set: ", len(_trainingset))
      #print("shape of training set: ", _trainingset.shape)
      #print("size of test set: ", len(_testset))
      #print("shape of test set: ", _testset.shape)

      mean_vals = np.mean(_trainingset, axis=0)
      std_devs = np.std(_trainingset, axis=0)

      # Normalize each feature using mean and standard deviation
      _input_train = (_trainingset - mean_vals) / std_devs
      _input_eval = (_testset - mean_vals) / std_devs      # test set is also normalized, using the mean and std dev from the -->training set<-- !
                                                      # they both need to be normalized using the same values, and the values must be from the training set only!


      #_input_train er trainingset
      # seq_length er antall kolonner delt på 3
      # Password length -1, pga enter, men den har blitt fjernet fra trainingset...

      #reshape the input and output to sequences. The sequences are chosen to be raw or derived features of physical sequences (key press/release or hold, down to down, down to up) so the input shape becomes N samples * Password length -1 (since the enter has 1 feature and not 3) * 3 features per password letter
      seq_length = _input_train.shape[1] /3
      input_train_torch = _input_train.reshape(_input_train.shape[0], int(seq_length), -1)
      input_eval_torch = _input_eval.reshape(_input_eval.shape[0], int(seq_length), -1)

      #if (i == pwd_id+6 and debug):
      #print ("input_train: ", _input_train.shape)
      #print ("labels_train: ", _labels_train.shape)
      #print ("input_eval: ", _input_eval.shape)
      #print ("labels_eval: ", _labels_eval.shape)


      _labels_train = to_onehot(_labels_train)
      _labels_eval = to_onehot(_labels_eval)

      trainingset = []
      for arr, label in zip(input_train_torch, _labels_train):
        x_tensor = torch.tensor([arr], dtype=torch.float32)
        y_tensor = torch.tensor([label], dtype=torch.float32)
        trainingset.append({'X_batch': x_tensor, 'y_batch': y_tensor})
      testset = []
      for arr, label in zip(input_eval_torch, _labels_eval):
        x_tensor = torch.tensor([arr], dtype=torch.float32)
        y_tensor = torch.tensor([label], dtype=torch.float32)
        testset.append({'X_batch': x_tensor, 'y_batch': y_tensor})








      """
      # #copy data into a tensor type based on CUDA availability
      if torch.cuda.is_available():

        train_dataset = TensorDataset(torch.tensor(_input_train, dtype = torch.long).cuda(),
                                      torch.tensor(to_onehot(_labels_train), dtype=torch.long).cuda()) # since these are one hot vectors of ints couldnt this also be of int64?


        eval_dataset = TensorDataset(torch.tensor(_input_eval, dtype=torch.long).cuda(),
                                    torch.tensor(to_onehot(_labels_eval), dtype=torch.long).cuda())


        # test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.long).cuda(),
        #                            torch.tensor(y_test, dtype=torch.long).cuda())


        #Batch_size, how many samples per batch to load
        #Shuffle = True, have the data reshuffled at every epoch
        #Drop_last = True, drop the last incomplete batch, if the dataset size is not divisible by the batch size. If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
        train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, drop_last=True, shuffle = True)
        eval_loader = DataLoader(dataset = eval_dataset, batch_size = batch_size, drop_last=True, shuffle = True)

        
        #test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, drop_last=True, shuffle = False)
      else:
        train_dataset = TensorDataset(torch.tensor(input_train_torch, dtype = torch.float32),
                                      torch.tensor(to_onehot(_labels_train), dtype=torch.float32)) # since these are one hot vectors of ints couldnt this also be of int64?


        eval_dataset = TensorDataset(torch.tensor(input_eval_torch, dtype=torch.float32),
                                    torch.tensor(to_onehot(_labels_eval), dtype=torch.float32))


        # test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.long),
        #                            torch.tensor(y_test, dtype=torch.long))

        #train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, drop_last=True, shuffle = True)
        #eval_loader = DataLoader(dataset = eval_dataset, batch_size = batch_size, drop_last=True, shuffle = True)
        #test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, drop_last=True, shuffle = False)
      """

      #add splitted dataset to array[passwordId][participantId] (eg [0][0] = Gigab-exclm-t R3ceiver, participant 1 )
      """
      input_train_list[i % 6].append(_input_train)
      input_eval_list[i % 6].append(_input_eval)
      labels_train_list[i % 6].append(_labels_train)
      labels_eval_list[i % 6].append(_labels_eval)
      input_loader_list[i % 6].append(train_loader)
      eval_loader_list[i % 6].append(eval_loader)
      """


      """
      input_train_list[pwd_id].append(_input_train)
      input_eval_list[pwd_id].append(_input_eval)
      labels_train_list[pwd_id].append(_labels_train)
      labels_eval_list[pwd_id].append(_labels_eval)
      input_loader_list[pwd_id].append(train_loader)
      eval_loader_list[pwd_id].append(eval_loader)
      """

      """
      input_train_list[pwd_id][p_id] = (_input_train)
      input_eval_list[pwd_id][p_id] = (_input_eval)
      labels_train_list[pwd_id][p_id] = (_labels_train)
      labels_eval_list[pwd_id][p_id] = (_labels_eval)
      input_loader_list[pwd_id][p_id] = (train_loader)
      eval_loader_list[pwd_id][p_id] = (eval_loader)
      """

      #for i in range (6, len(directory_files_preprocessed)):
      # => 6 + p_id + pwd_id
      #for pwd_id in range(6):
        #for p_id in range(5):


      #sanity check (password 'Gigab-exclm-t R3ceiver' (index 0) and participant 1 (index 0) -> 320+80=400 entries)
      #if debug:

        #print ("train input (#batches): ", len(input_loader_list[pwd_id][p_id]))
        #print ("train eval (#batches): ", len(eval_loader_list[pwd_id][p_id]))
        #print ("train input: ", input_train_list[pwd_id][p_id].shape)
        #print ("train labels (regular): ", labels_train_list[pwd_id][p_id].shape)

        #inputs, labels = next(iter(input_loader_list[pwd_id][p_id]))
        #print ("train labels (one hot): ", labels.shape)
        #print ("sample one-hot label: ", labels[0])
        #print ("sample input shape: ", inputs.shape)
        #print ("sample input : ", inputs[0])


      # implementation examples of the LSTM can be seen here: https://github.com/pytorch/pytorch/blob/main/benchmarks/fastrnns/custom_lstms.py
      # colab we used as research: https://colab.research.google.com/github/dlmacedo/starter-academic/blob/master/content/courses/deeplearning/notebooks/pytorch/Time_Series_Prediction_with_LSTM_Using_PyTorch.ipynb#scrollTo=_BcDEjcABRVz
      # paper we use as benchmark: [Soni & Prabakar, 2022] "KeyNet: Enhancing Cybersecurity with Deep Learning-Based LSTM on Keystroke Dynamics for Authentication" https://link.springer.com/chapter/10.1007/978-3-030-98404-5_67
      # the paper main claim is to detect which of the 51 persons from [Killourhy and Maxion, 2009]'s keystroke dynamics dataset has typed a single password  .tie5Roanl + enter key with 98% accuracy adn loss of 0.05%
      # the LSTM uses one-hot encoding of output labels, a 70-30 train-eval distribution, 4 LSTM layers x 96 nodes with 4 dropout layers and 1 fully connected layer to the output, K-fold cross-validation (K = 10), ReLU activation for hidden layers and Softmax activation for fully connected layer, Adam optimizer for dynamic learning rate, categorical cross entropy loss function, batch-size 32 using mini-batch gradient descent, epoch 100. For the LSTM cell the memory gate uses tanh activation, for the other (input, output, forget) the sigmoid activiation function was used
      # About data preprocessing: "To feed the data to the LSTM network, we transformed the feature set into timed windowed sequences of input and output.". For evaluation precision/recall and EER was used
      # About batching: they found 32 to be the best batch_size. This is dependent on the GPU or CPU (how much can be put in memory) but also data set: smaller batch_size allows updating the weights after the batch_size instead of updating it after seeing the entire dataset first (once or even multiple times). Academic paper mentioned in https://github.com/rantsandruse/pytorch_lstm_02minibatch

      # Things that could be better in [Soni & Prabaker] paper:
      # They use ReLU activation between hidden layers which is weird, see https://discuss.pytorch.org/t/trying-to-understand-the-use-of-relu-in-a-lstm-network/143300/4, https://stats.stackexchange.com/questions/444923/activation-function-between-lstm-layers and https://datascience.stackexchange.com/questions/66594/activation-function-between-lstm-layers.IT could also be that they used rely for the LSTM instead of Tanh, but then why did they mention tanh in the paper. https://stackoverflow.com/questions/40761185/what-is-the-intuition-of-using-tanh-in-lstm
      # They did not report any normalizing of the data (alhough the data is very much in the same range).
      # They did not report how much a feature contributed to the end result (some featrures are redundant), the just said transformed features set into a timed windowed sequence of input and output, did they use all features? Probably because input vector is 31 features
      # They did not report why they choose 4 LSTM layers and 96 nodes and what dropout probablity per layer and what learning_rate
      # They did not share their data or source code despite it being a colab file.
      # They got good result, but unclear if they really had unseen data in the test set or if they trained on the whole set using different folds. Another thing is they use 31 as input vector in sequence which is incorrect. some sequences overlap. some features belong to the same key and should be combined.
      # They did not share performance result GPU vs CPU (perhaps with power usage  or symbol/sec/watt)
      # The data structure is None x 31 x1, meaning according to doc: N,L,H or [batch size, sequence size, input size] but in text they say batch size 32 is used. In addition Sequence size means that each feature is considered a time step which is wrong (there is redudancy as they are derived features). It might be more correct to use time steps = keys in password and features = t_v (t_press) and t_^ (t_release) so a [batch] x [password length + 1] x [2 features]
      # Minor: not reported that they are doing a stateless LSTM architecture of type many-input (which is one input sequence) -to-one-output ( which has binary classes) classificaiton
      # Minor: the dataset is only 1 password, so it is unclear how well the architecture generalizes to different lengths
      # Minor: also mention other metrics like F1 or ROC (https://deepchecks.com/f1-score-accuracy-roc-auc-and-pr-auc-metrics-for-models/) and perhaps do a confusion matrix when doing multiclass so we can see which of the 51 classes are alike according to classifier.
      # Minor: show application how fast the network can do a forward pass on a trained NN and demonstrate actually usability for password authentication
      # Remark: They did not need stratified selection since samples for each class were equal,LSTM with sequences seems to be not smart (https://github.com/rantsandruse/pytorch_lstm_02minibatch/tree/main)
      # Remark: They seem to split in train, evaluation, test https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7 with k-fold cross validation
      # Remark: Did they use Drop_last with batching and ignored some data since their training set /32 is not an integer

      # Halvor's raw training data is sequential in nature, each row of the input and label set contains a complete input-output sequence. Interesting to compare relative time vs absolutime
      # we could also use the nn.LSTM model without any custom init and forward functions eg. https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

      class LSTM(nn.Module):

          # mandatory constructor since we inherit from nn.Module
          def __init__(self, num_classes, input_size, hidden_size, num_layers, batch_size):
              # initialize the parent class nn.Module
              super().__init__()

              # variables are reused in the forward pass function, hence stored
              self.num_classes = num_classes
              self.input_size = input_size
              self.hidden_size = hidden_size
              self.num_layers = num_layers
              self.num_directions = 1 # if bidirectional=TRUE should be 2 else 1
              self.batch_size = batch_size

              # create the pytorch LSTM model
              self.lstm = nn.LSTM(input_size=input_size,    # INPUT_SIZE features in the input layer
                                  hidden_size=hidden_size,  # HIDDEN_SIZE features in the hidden layer
                                  num_layers=num_layers,    # NUM_LAYERS number of hidden layers
                                  batch_first=True,         # BATCH_FIRST, input in format (batch, sequence, token)
                                  bidirectional=False,      # DIRECTIONS, bidirectional LSTM or not (also need to update num_directions if you change this parameter)
                                  bias=True,                # BIAS use trainable bias node for each hidden layer (which is the default) although the input range and output range are very similar having a bias node might improve performance
                                  #dropout = 0.33)           # DROPOUT adds a dropout layer after each hidden layer (only during training). Use dropout probability for each node of eg. 33% to force the learning to distribute it over the entire network and prevent overfitting, but at the cost of slower training
                                  dropout = 0)

              # create a fully connected layer output layer with NUM_CLASSES being the output node(s), in this case 2 (0 and 1). If it is 1 class, it is a regression problem and not a classification problem
              self.fc = nn.Linear(hidden_size, num_classes)


          def init_hidden(self):
              '''
              Initiate hidden states.
              '''
              # Shape for hidden state and cell state: num_layers * num_directions, batch, hidden_size
              h_0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size)
              c_0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size)

              # The Variable API is now semi-deprecated, so we use nn.Parameter instead.
              # Note: For Variable API requires_grad=False by default;
              # For Parameter API requires_grad=True by default.
              h_0 = nn.Parameter(h_0, requires_grad=True)
              c_0 = nn.Parameter(c_0, requires_grad=True)

              return h_0, c_0

          def forward(self, x):
              # debug.append("1")
              #debug.append(print (x.shape))
              # debug.append("2")
              # debug.append("3")
              #debug.append(print (self.num_layers))
              #debug.append(print (x.size(0)))
              #debug.append(print (self.hidden_size))

              hidden_0 = self.init_hidden()
              #embeds = self.word_embeddings(sentences)
              #...
              #lstm_out, _ = self.lstm(embeds, hidden_0)

              #h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

              #c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

              # debug.append("5")
              # debug.append(print (h_0.shape))
              # debug.append("6")
              # debug.append(print (c_0.shape))

              # Propagate input through LSTM
              ula, (h_out, _) = self.lstm(x, hidden_0)
              #ula, (h_out, _) = self.lstm(x, (h_0, c_0))
              #ula, (h_out, _) = self.lstm(x)

              h_out = h_out.view(-1, self.hidden_size)

              out = self.fc(h_out)

              return out

          # mandatory method since we inherit from nn.Module. Each input sequence needs a one forward pass to evaluate.
          # note that we can change everything here, including how the LSTM cell is working. This includes the activation functions for each gate and how they are combined. See https://discuss.pytorch.org/t/custom-lstm-cell-implementation/64566 for an example
          # def forward(self, x):

          #     # hidden state vector set to zero. Needed to reset prior memory https://www.mathworks.com/matlabcentral/answers/2031634-why-are-hidden-state-and-cell-state-vectors-zero-after-training-an-lstm-model-with-trainnetwork-func
          #     h_0 = Variable(torch.zeros(self.num_layers, self.hidden_size))

          #     # cell state vector set to zero, Needed to reset prior memory
          #     c_0 = Variable(torch.zeros(self.num_layers, self.hidden_size))

          #     # propagate input through LSTM, x is the input sequence, h and c are the prior knowledge states. The output is a uniform linear array (ULA) and the hidden and cell state vectors of the final layer, see https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html and https://discuss.pytorch.org/t/in-lstm-which-layer-should-i-use-as-output/121201. As shown in the last link, the output (ula) vector contains output of all time steps and we are only interested in the last time step in this exercise. The H_out vector does not contain all time steps but contains the last hidden state layer information. Not sure why the h_out is used over ula since they have the same information in this case. Perhaps easier to filter?
          #     ula, (h_out, _) = self.lstm(x, (h_0, c_0))

          #     # Split num_layers and num_directions (useful if your LSTM is bidirectional), see https://discuss.pytorch.org/t/in-lstm-which-layer-should-i-use-as-output/121201/3
          #     h_out = h_out.view(self.num_layers, self.num_directions, self.batch_size, self.hidden_size)

          #     # Get the last layer with respect to num_layers
          #     h_out = h_out[-1]

          #     # Handle num_directions dimension (assume bidirectional=False)
          #     h_out = h_out.squeeze(0)

          #     # run the output through the fully connected (and trained) layer to retrieve the final output, the shape of h_out is (batch, hidden_size)
          #     out = self.fc(h_out)

          #     #return the output vector which contains 2 values with probablities. The highest probablity is the chosen class eg. [0.8 0.2] means 80% label 0, 20% label 1
          #     return out


      input_size = 3 # These are the features per time step Ideally: password length * 2 features Tv (T_press) and T^ (T_release), currently it is (password length * 3 features: hold, down-to-down, up-to-down) -2 (since enter has not sequence afterwards)
      hidden_size = 96 # Estimate this from formula eg. https://www.kaggle.com/code/kmkarakaya/lstm-understanding-the-number-of-parameters
      num_layers = 1 # Estimate this from formula
      num_classes = 6 # 2 output labels, participant or attack_set or 6 if each has an output node
      num_epochs = 200 # Try for several epochs, typically early stop is used when eval metric start to become worse since training set has no knowledge of eval and can thus overfit on training set
      learning_rate = 0.001 # Try for several numbers. If a dynamic learning rate like ADAM (Adaptive Moment Estimation) is used this is only an initial value and 0.001 is often used.

      debug = []

      # Define & init the model
      lstm = LSTM(num_classes, input_size, hidden_size, num_layers, batch_size)


      #See an overview of loss functions https://pytorch.org/docs/stable/nn.html#loss-functions and https://gombru.github.io/2018/05/23/cross_entropy_loss/ for an idea of which type are useful for this dataset. The [Soni & Prabaker] paper paper uses "categorical cross entropy loss" also called "softmax loss" which is a softmax activation function + cross entropy loss function. This is a function distributes the result over the available categories (labels). However, other types like binary cross entropy loss (sigmoid activation function + cross entropy loss function) is also interesting because it doesnt force distribution over the categories (eg. they all add to 1.0) but rather treats each category as independent (eg. it is possible to have .8 and .7 for two categories). This is important for multi-label classification when a single example can contains results of multiple persons (eg. the first part of the password by person A and the rest by person B). In this case Softmax loss makes sense to use, which is called CrossEntropyLoss in Pytorch
      #criterion = torch.nn.CrossEntropyLoss()

      # BCE seems to give better results with 6 classes why?
      #criterion = torch.nn.BCEWithLogitsLoss()
      
      attackcount = 0
      defencecount = 0
      for d in trainingset:
        temp = d['y_batch']
        #print(temp)
        #print(temp[0][5])
        temp_array = temp[0].numpy()
        #print(temp_array[5])
        if (temp_array[5] == 1):
          attackcount +=1
        else:
          defencecount +=1

      print("defence count ", defencecount)
      print("attack count ", attackcount)

      lossweight = attackcount / defencecount
      print("lossweight ", lossweight)
      #pos_weight = torch.Tensor(np.full(num_classes,lossweight)) #?
      pos_weight_array = [lossweight,lossweight,lossweight,lossweight,lossweight,1]
      pos_weight = torch.tensor(pos_weight_array)
      criterion = torch.nn.BCEWithLogitsLoss(pos_weight= pos_weight)
      
      #SGD is static learning rate, all others are dynamic using momentum, ADAM is popular. ADAM main features are dynamic individual node learning rates with momentum, bias correction and low memory due to storage of a small history (https://www.analyticsvidhya.com/blog/2023/09/what-is-adam-optimizer/). Note that since we use ADAM we dont need to call scheduler.step()
      optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

      import math
      def sigmoid(x):
        return 1 / (1 + math.exp(-x))

      # Train the model, based on https://pytorch.org/docs/stable/optim.html

      #print model architecture for a particular password and participant
      #inputs, labels = next(iter(input_loader_list[pwd_id][p_id]))
      #stats = summary(lstm, input_data=inputs, device= device, col_names=("input_size", "output_size", "num_params"))
      #print (stats)


      #print("")
      #print("Start training model")
      batches_processed_train = 0 #ie. samples seen = batches * batch_size
      batches_processed_eval = 0 #ie. samples seen = batches * batch_size


      #to get the x most recent eval acc avgeraged
      eval_acc_list = []
      recent_avg_acc = 0
      prev_avg_acc = 0


      for epoch in range(num_epochs):
        #explicitely set the model in training mode (and thus activate dropout and use batchnorm per-batch statistics) see https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
        lstm.train()
        #for X_batch, y_batch in input_loader_list[pwd_id][p_id]:
        #for X_batch, y_batch in trainingset:
        for d in trainingset:
          X_batch = d['X_batch']
          y_batch = d['y_batch']

          #print("X_batch:")
          #print(X_batch)
          #print("y_batch:")
          #print(y_batch)

          #reset the history of gradients of any previous training
          optimizer.zero_grad()



          # once = True

          # if once == True:
          #   print (X_batch.shape)
          #   print (y_batch.shape)
          #   print (X_lens_batch.shape)

          # h_0 = torch.randn(num_layers, batch_size, hidden_size)

          # print (X_batch.shape)
          # print (X_lens_batch.shape)
          # print (h_0.shape)

          # Do a forward pass
          outputs = lstm(X_batch)



          # if once == True:
          #   print (outputs.shape)

          once = False;
          # obtain the loss score
          train_loss = criterion(outputs, y_batch)
          #batches_processed_train += 1
          #writer.add_scalar('Loss/training', train_loss, batches_processed_train * batch_size)
          writer.add_scalar('Loss/training', train_loss, epoch)

          # Do a backpropagation pass (compute the gradients (step direction or sign and its quantity per individual weight) without updating the weights)
          #note, if it has to be smaller, the result will never be 100%
          train_loss.backward()

          # Actually do the updating of the weights using the computed gradients
          optimizer.step()


        # Eval
        lstm.eval()
        metric.reset()
        with torch.no_grad():



          predictions = []
          correctlabels = []

          #model makes prediction values from 0 (attack) to 1 (defend), and stores it in a list. Compared with threshold later.
          #for X_val, y_val in eval_loader_list[pwd_id][p_id]:
          for d in testset:
            X_val = d['X_batch']
            y_val = d['y_batch']
            outputs = lstm(X_val)
            for n in range(batch_size):

              predict = sigmoid(outputs[n][p_id] - outputs[n][5])
              predictions.append(predict)
              correctlabels.append(y_val[n][p_id])
              # small (negative) value for attack prediction, large (positive) for user prediction
              #print("\n")
              #print(y_val[n][p_id])
              #print(predict)

          #it should be done like this because same LSTM model produces different result with same input each time you run it


          foobar = False #EER troubleshooting
          foodbar = False #Confucius: Man run in front of car get tired, man run behind car get exhausted
                          #I mean, confusion matrix values


          #Finding the accuracy, at the EQUAL ERROR RATE
          #it does this by doing a binary search on the threshold value
          EERiterations = 15  ##  how many rounds of binary search splits
          threshold = 0.5
          maxthresh = 1
          minthresh = 0
          for i in range(EERiterations):
            TN = 0
            FP = 0
            FN = 0
            TP = 0

            #for X_val, y_val in eval_loader_list[pwd_id][p_id]:
              #for n in range(batch_size):
            for p in range(len(predictions)):
              if (predictions[p] > minthresh and predictions[p] < maxthresh and foobar and epoch > 20):
                print(predictions[p])
              if foodbar and epoch > 20:
                print("\nIn EER: ")
                print(predictions[p])
                print(correctlabels[p])
              if predictions[p] > threshold:  #predicts correct label is p_id
                if correctlabels[p] == 1: #if the correct label is p_id
                  TP +=1
                  if foodbar and epoch > 20:
                    print("TP")
                else:
                  FP +=1
                  if foodbar and epoch > 20:
                    print("FP")
              else:
                if correctlabels[p]:
                  FN +=1
                  if foodbar and epoch > 20:
                    print("FN")
                else:
                  TN +=1
                  if foodbar and epoch > 20:
                    print("TN")


            if (FN/ (TP+FN)) < (FP / (TN + FP)):
              minthresh = threshold
            else:
              maxthresh = threshold
            threshold = (minthresh + maxthresh) / 2
            if epoch > 20:
              if foobar:
                print("\n")
                print(i)
                print("FN: ", FN/ (TP+FN))
                print("FP: ", FP / (TN + FP))
                print("minthresh: ", minthresh)
                print("maxthresh: ", maxthresh)
                print("threshold: ", threshold)
          ## end of EER search



          eval_loss = criterion(outputs, y_val) #torch.nn.CrossEntropyLoss(outputs, y_val) # Shouldn't this only use [p_id] and [5]? the other labels don't matter.
          writer.add_scalar('Loss/test', eval_loss, epoch)


          acc = (TP+TN) / (TP+TN+FN+FP)
          eval_acc_list.append(acc)
          writer.add_scalar('Acc/FOOBAR', acc, epoch)


          # Calculate the average of the last (avg_width) accuracies
          recent_avg_acc = 1
          prev_avg_acc  = 0
          avg_width = 10

          if epoch > avg_width:
            recent_avg_acc = eval_acc_list[-avg_width:]
            recent_avg_acc = sum(recent_avg_acc) / len(recent_avg_acc)
          if epoch > avg_width*2:
            prev_avg_acc = eval_acc_list[-2*avg_width:-avg_width]
            prev_avg_acc = sum(prev_avg_acc) / len(prev_avg_acc)


        #print("Epoch: ", epoch, " \tTraining Loss: ", train_loss.item(), " \tEval Loss: ", eval_loss.item(), "\tEval Acc: ", acc.item(), "\t test: ", recent_avg_acc.item())
        #print("Epoch: ", epoch, " N= ",  (TP+TN+FN+FP), "Count=", counter)
        #print("Epoch: ", epoch, " \tTrain/Eval Loss: {:.3f}".format(train_loss.item()), " / {:.3f}".format(eval_loss.item()), "\tEval Acc: {:.4f}".format(acc), "\tmost recent eval acc: {:.4f}".format(recent_avg_acc), " / {:.4f}".format(prev_avg_acc), "\tFRR/FAR: {:.4f}".format(100*FN/ (TP+FN)),  " / {:.4f}".format(100*FP/ (TN+FP)),"\t FR: ", FN, "\tFA:", FP, "\tTR: ", TN, "\tTA:", TP)

        if recent_avg_acc < prev_avg_acc:
          if epoch > 40:
            print("Final Epoch: ", epoch, " \tTrain/Eval Loss: {:.3f}".format(train_loss.item()), " / {:.3f}".format(eval_loss.item()), "\tEval Acc: {:.4f}".format(acc), "\tmost recent eval acc: {:.4f}".format(recent_avg_acc), " / {:.4f}".format(prev_avg_acc), "\tFRR/FAR: {:.4f}".format(100*FN/ (TP+FN)),  " / {:.4f}".format(100*FP/ (TN+FP)),"\t FR: ", FN, "\tFA:", FP, "\tTR: ", TN, "\tTA:", TP)
            break
        metric.reset()
      print("Epoch ", epoch - avg_width+1, ", ", avg_width+1, "before final epoch: ", eval_acc_list[epoch - avg_width+1])  #oldest one in the worst average, if it chooses the one that just left the worst, it could have been the one keeping the average from being the worst one of the two (especially high)
      #print(2*avg_width, "epochs ago: ", eval_acc_list[epoch - 2*avg_width])
      resultsforpaper[resultscounter] += eval_acc_list[epoch - avg_width+1]

      #sanity check
      #print("Model output at last index of the last batch:")
      #print("Pred: ", outputs[batch_size -1], " True: ", y_val[batch_size -1])
      #print("Input: ", X_val[batch_size -1])


print()
print()
print("final results for paper: ")
for i in range(len(resultsforpaper)):
  resultsforpaper[i] /= 5
  print("---")
  print("pwd_id: ", passwordorder[i//5])
  print("participant: ", (i%6) + 1)
  print(resultsforpaper[i])




