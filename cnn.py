import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import csv


avg_width = 10
random.seed(4242)
resultsforpaper = [0.0] * 30

k = 5
    


def load_data(fn, trimfirstn=0, trimlastn=0):
    f = open(fn)
    lines = [l for l in f.readlines()]
    lines = lines[1:]
    all_values = []
    for l in lines:
        values = []
        for n in l.split(","):
            values.append(n)
        if len(values) == 0:
            continue
        values = values[trimfirstn:]
        values = values[: len(values) - trimlastn]
        assert len(values) > 0
        values = [float(v) for v in values]
        all_values.append(torch.Tensor(values).cuda())
    return all_values


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        layersize = 128
        self.input = nn.Conv1d(3, layersize, kernel_size=3, padding=1)

        #self.convs = nn.ModuleList(nn.Conv1d(layersize, layersize, kernel_size=3, padding=1) for _ in range(4))
        self.convs = nn.ModuleList()
        for _ in range(4):
            self.convs.append(nn.Conv1d(layersize, layersize, kernel_size=3, padding=1))

        self.output = nn.Linear(layersize, 1)

        



    def forward(self, x):
        
        # 3, 8
        x = self.input(x)
        # 128, 8
        for c in self.convs:
            x = c(x).relu()
        # 128, 8
        x = x.amax(-1)
        # 128
        x = self.output(x).sigmoid()
        # 1

        return x


defence_data = []
attack_data = []
train_data = [] 
test_data = []

def add_data(data, label):
    for x in data:
        x = x[:-1]
        x = x.view(-1, 3).t()
        d = {"x": x, "y": torch.Tensor([label]).cuda()}
        defence_data.append(d)

def add_attack_data(data, label):
    for x in data:
        x = x[:-1]
        x = x.view(-1, 3).t()
        d = {"x": x, "y": torch.Tensor([label]).cuda()}
        attack_data.append(d)

passwords = ["observer", "Ob-dollar-erv3r", "gigabit receiver", "Gigab-exclm-t R3ceiver", "flying automatic monster", "repetition learn machine thinker"]

scorecounter = 0
threshold = 0.5 #gets adjusted for EER
epochs = 200

trainloss = []
testloss = []
testFR = []
testFA = []
testacc = []

threshold = 0.5
TA = 0.0
TR = 0.0
FA = 0.0
FR = 0.0

epochcounter = 0

printgraphs = False #True
resultcounter = -1

iterationloss = [0.0] * epochs




for passnr in range(0, 6):  
    print()
    print()
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print()
    print (passwords[passnr])
    for participantnr in range(1,6):
        resultcounter += 1
        print()
        print("participant%d" % participantnr)

        

        #split data into 5 groups here
        defence_data = []
        attack_data = []
        
        attacker = f"data/attack - {passwords[passnr]}.csv"
        defender = f"data/participant{participantnr} - {passwords[passnr]}.csv"

        add_attack_data(load_data(attacker, 2), 1)
        add_data(load_data(defender), 0)

        #defence_data = np.array(defence_data)
        #attack_data = np.array(attack_data)
        #defence_data = np.array([tensor.numpy() for tensor in defence_data])
        #attack_data = np.array([tensor.numpy() for tensor in attack_data])
        #print("attack data: ", attack_data)
        #print("shape of defence data: ", attack_data.shape)

        #print(attack_data)
        attack_buffer = np.stack([d['x'].cpu().numpy() for d in attack_data])
        defence_buffer = np.stack([d['x'].cpu().numpy() for d in defence_data])
        #print("attack buffer shape: ", attack_buffer.shape)
        #print("defence buffer shape: ", defence_buffer.shape)
        #print(attack_buffer)
        
        # convert attack and defence data to 2d or 3d array
        #use that, do the thing
        # convert training and test sets to list of tensors, with the labels
        
        # Split into groups of 10 lines (one group per attack participant)
        tensplit_inputA = np.stack(np.array_split(attack_buffer, len(attack_buffer)/10))  #splits into nr of attack participants
        #print("Shape of tensplit_inputA:", tensplit_inputA.shape)
        np.random.shuffle(tensplit_inputA)  # shuffles by ATTACK PARTICIPANT, not by line
        # Split into 5 groups, then merge the attack parcitipants in each group
        #tenfivesplit_inputA = np.stack(np.array_split(tensplit_inputA, 5))  #splits into 5 groups (does not work)
        k_inputA = np.array_split(tensplit_inputA, k)  # This gives a LIST of 2D arrays. 
        #The 2D arrays can vary in length since the number of attack participants are not necessarily divisible by 5
        # merging the participants in each k group 
        for i, arr in enumerate(k_inputA):
          #print(f"FOOBAR Shape of array {i+1}: {arr.shape}")
          #k_inputA[i] = k_inputA[i].reshape(-1, k_inputA[i].shape[-1])
          k_inputA[i] = k_inputA[i].reshape(-1, arr.shape[-2], arr.shape[-1]) 
          #print(f"FOOBAR NEW Shape of array {i+1}: {k_inputA[i].shape}")

        #randomize the order of the defence data
        np.random.shuffle(defence_buffer)
        
        k_inputP = np.array_split(defence_buffer, k) 
        #print("k_inputA shape: ", k_inputA[0].shape)    #list of 5 attack data groups
        #print("k_inputP shape: ", k_inputP[0].shape)    #list of 5 defence data groups
        
        #K-FOLD VALIDATION

        for k_nr in range(k): #K-FOLD VALIDATION IN THIS LOOP
            random.seed(42 + k)
            #trainingset = []
            #testset = []
            print("@@@")
            print("K-fold nr ", k_nr)
            #create the training and test sets with labels for this iteration of k-fold
            buffer_trainingset = np.empty((0, 0, 0))
            traininglabels = np.empty(0)
            buffer_testset = np.empty((0, 0, 0))
            testlabels = np.empty(0)
            
            for i in range(5):
                if i == k_nr:   #for one of k groups, use as test set 
                  buffer_testset =  np.concatenate((k_inputA[i], k_inputP[i]))
                  ones_array = np.ones(len(k_inputA[i]), dtype=int)
                  zeros_array = np.zeros(len(k_inputP[i]), dtype=int)
                  testlabels = np.concatenate((ones_array, zeros_array))
                else:
                  ones_array = np.ones(len(k_inputA[i]), dtype=int)
                  zeros_array = np.zeros(len(k_inputP[i]), dtype=int)
                  if len(buffer_trainingset) == 0:
                    buffer_trainingset = np.concatenate((k_inputA[i], k_inputP[i]))
                    traininglabels = np.concatenate((ones_array, zeros_array))
                  else:
                    buffer_trainingset = np.concatenate((buffer_trainingset, k_inputA[i], k_inputP[i]))
                    traininglabels = np.concatenate((traininglabels, ones_array, zeros_array))

            #trainingset = np.array(trainingset)
            #traininglabels = np.array(traininglabels)
            #testset = np.array(testset)
            #testlabels = np.array(testlabels)
            # Shuffle training data and labels
            num_samples = len(traininglabels)
            indices = np.random.permutation(num_samples)
            buffer_trainingset = buffer_trainingset[indices]
            traininglabels = traininglabels[indices]

            # Shuffle test data and labels
            num_samples = len(testlabels)
            indices = np.random.permutation(num_samples)
            buffer_testset = buffer_testset[indices]
            testlabels = testlabels[indices]

            print("size of training set: ", len(buffer_trainingset))
            #print("shape of training set: ", buffer_trainingset.shape)
            print("size of test set: ", len(buffer_testset))
            #print("shape of test set: ", buffer_testset.shape)

            mean_vals = np.mean(buffer_trainingset, axis=0)
            std_devs = np.std(buffer_trainingset, axis=0)

            # Normalize each feature using mean and standard deviation
            buffer_trainingset = (buffer_trainingset - mean_vals) / std_devs
            buffer_testset = (buffer_testset - mean_vals) / std_devs      # test set is also normalized, using the mean and std dev from the -->training set<-- !
                                                          # they both need to be normalized using the same values, and the values must be from the training set only!

            #print("shape of buffer_trainingset: ", buffer_trainingset.shape)
            #print("shape of buffer_testset: ", buffer_testset.shape)
            
            trainingset = []
            for arr, label in zip(buffer_trainingset, traininglabels):
                # Convert numpy array to tensor and send to device
                x_tensor = torch.tensor(arr).to('cuda:0')
                # Convert label to tensor and send to device
                y_tensor = torch.tensor([label], dtype=torch.float32).to('cuda:0')
                # Create dictionary and append to list
                trainingset.append({'x': x_tensor, 'y': y_tensor})
            testset = []
            for arr, label in zip(buffer_testset, testlabels):
                # Convert numpy array to tensor and send to device
                x_tensor = torch.tensor(arr).to('cuda:0')
                # Convert label to tensor and send to device
                y_tensor = torch.tensor([label], dtype=torch.float32).to('cuda:0')
                # Create dictionary and append to list
                testset.append({'x': x_tensor, 'y': y_tensor})

            

            model = Model()
            model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
            #optimizer = torch.optim.SGD(model.parameters(), lr=2e-3)
            #lr LEARNING RATE
            #step size

            #num_params = 0
            #for p in model.parameters():
            #    num_params += p.numel()
            #print("Number of parameters: %d" % num_params)

            stop = False
            
            epochcounter = - 1
            testacc = []
            for i in range(epochs): #epochs, no batches
                
                epochcounter += 1
                print()
                print("epoch ", epochcounter)
                avgloss = 0.0
                losscounter = 0
                #print("iteration %d" % i)
                #random.shuffle(all_data)

                defencecount = 0 #find exact numbers y == 0
                attackcount = 0 #y == 1
                
                
                for d in trainingset:
                    if d["y"] == 1:
                        attackcount += 1
                    else:
                        defencecount += 1
                neglossweight = attackcount / defencecount 
                
                for d in trainingset:

                    x = d["x"].clone()
                    #x += torch.randn_like(x) * 0.01
                    
                    prob = model(x)
                    y = d["y"]
                    
                    loss = torch.nn.functional.binary_cross_entropy(prob, y)
                    if (d["y"]==0): #defence, minority class
                        loss*=neglossweight
                        #use weights proportional to amounts

                                    
                    #loss = torch.nn.functional.mse_loss(prob, y)
                    #mean square error, litt mindre effektivt av en grunn
                    #det gir loss mellom 0 og 1
                    #binary cross entropy er laget for å være bra med sigmoid
                    #binary cross entropy gir uendelig loss når det er maks forskjell
                    avgloss += loss
                    losscounter +=1
                    # print("loss: %g" % loss)

                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                iterationloss[i] = (avgloss/losscounter)
                trainloss.append(iterationloss[i])
              

                with torch.no_grad(): ##not sure if I'm using this correctly???
                    testlosses = []
                    for d in testset:
                        prob = model(d["x"])
                        y = d["y"]
                        
                        loss = torch.nn.functional.binary_cross_entropy(prob, y)
                        
                        testlosses.append(loss.cpu())
                    testloss.append(sum(testlosses) / len(testlosses))
            

                #if stop:
                    #break
                
                #GETTING THE EER
                threshold = 0.5     #starting threshold
                maxiterations = 50  #for binary search
                minthresh = 0.0 
                maxthresh = 1.0
                stepsize = 0.01


                TA = 0.0
                TR = 0.0
                FA = 0.0
                FR = 0.0

                
                for count in range (maxiterations):
                    threshold = (minthresh + maxthresh)/2

                    with torch.no_grad():
                        losses = []
                        score = 0
                        n = 0
                        m = 0
                        TA = 0.0
                        TR = 0.0
                        FA = 0.0
                        FR = 0.0
                        correct = 0.0

                        
                        for d in testset:

                            #with noise
                            x = d["x"].clone()
                            #x += torch.randn_like(x) * 0.01
                            prob = model(x)
                            y = d["y"]

                            
                            #prob = model(d["x"])
                            #y = d["y"]

                            
                            # print("y %g, pred %g" % (y, prob))
                            if y == 0:
                                n +=1
                            else:
                                m +=1


                            correct = (prob > (threshold)) == (y > (threshold)) #y > threshold only checks if y is 0 or 1 since threshold is between 0 and 1

                            score += 1 if correct else 0
                            
                            if correct:
                                if y == 0:
                                    TA +=1
                                else:
                                    TR +=1
                            else:
                                if y == 0:
                                    FR +=1
                                else:
                                    FA +=1

                        if (100*FA/(FA+TR)  <  100*FR/(TA+FR)):
                            minthresh = threshold
                        else:
                            maxthresh = threshold

                testFA.append(100*FA/(FA+TR))
                testFR.append(100*FR/(TA+FR))
                print("threshold: ", threshold)
                print("EER FA: ", 100*FA/(FA+TR))
                print("EER FR: ", 100*FR/(TA+FR))
                #testacc.append((TA+TR) / (TA+TR+FR+FA))
                ## EER is not completely exact due to different resolutions,
                #    due to different number of positive/negative data
                # Using average of the two errors for accuracy is about the same as acc on average:
                testacc.append(  100 - (((100*FA/(FA+TR)) + (100*FR/(TA+FR)))/2) )
                print("testacc {:.4f}".format(testacc[epochcounter]))
                recent_avg_acc = 1
                prev_avg_acc  = 0

                if epochcounter +2 > avg_width: #+2 because equal or larger, and epoch is zero indexed
                  recent_avg_acc = testacc[-avg_width:]
                  recent_avg_acc = sum(recent_avg_acc) / len(recent_avg_acc)
                if epochcounter +2 > avg_width*2:
                  prev_avg_acc = testacc[-2*avg_width:-avg_width]
                  prev_avg_acc = sum(prev_avg_acc) / len(prev_avg_acc)

                print("prev avg {:.4f}".format(prev_avg_acc))
                print("recent avg {:.4f}".format(recent_avg_acc))
                if recent_avg_acc <= prev_avg_acc:
                    break
                
            #end of epoch loop



            trainloss_cpu = [] #hack fix for weird error

            for i in range (epochcounter+1):
                
                trainloss_cpu.append(iterationloss[i].cpu().item())


      
            """
            if printgraphs:
                xaxis = [i for i in range(epochs)]
                plt.plot(xaxis, testloss, label="test loss")
                plt.plot(xaxis, trainloss_cpu, label="train loss")
                plt.plot()
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Training and Test loss per Epochs")
                plt.legend()
                plt.show()

                plt.plot(xaxis, testFA, label="False Accept")
                plt.plot(xaxis, testFR, label="False Reject")
                plt.plot()
                plt.xlabel("Epoch")
                plt.ylabel("Percent Error")
                plt.title("EER per Epochs")
                plt.legend()
                plt.show()
            """


            trainloss = []
            testlosses = []
            testloss = []
            testFR = []
            testFA = []



            print("")

            #print("threshold %f" %threshold)
            print("test correct: %d/%d" % (score, n+m))
            print("%f %%" % (100*score / (n+m) ))
            print("TA: %f FR: %f" %(100*TA/(TA+FR),100*FR/(TA+FR)))
            print("FA: %f TR: %f" %(100*FA/(FA+TR),100*TR/(FA+TR)))

            scorecounter +=1
            print("%d / 150" %scorecounter)
            print("k fold ", k_nr +1, " / ", k)


            print("Final epoch at early stopping: ", epochcounter)
            print("epoch for paper result: ", epochcounter - avg_width +1)
            print("Use this one:", testacc[epochcounter - avg_width +1], "@@@")
            resultsforpaper[resultcounter] += testacc[epochcounter - avg_width +1]





    

for i in range(len(resultsforpaper)):
    resultsforpaper[i] /= k
    print("\nparticipant ", (i%5)+1)
    print("pwd",(i//6)+1)
    print(i)
    print("result {:.4f}".format(resultsforpaper[i]), "%")

