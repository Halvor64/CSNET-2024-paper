import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import csv



k = 5


resultsforpaper = [0.0] * 30
resultscounter = 0
random.seed(44)


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
        
        self.linear1 = nn.Linear(feats, layersize)    #stort nettverk overfitter raskere på lite datasett
        self.linear2 = nn.Linear(layersize, layersize)
        #self.linear2 = nn.Linear(layersize, 1)
        self.linear3 = nn.Linear(layersize, 1)
        
        
        #self.linear1 = nn.Linear(feats, layersize)
        #self.linear2 = nn.Linear(layersize, layersize)
        #self.linear3 = nn.Linear(layersize, layersize)
        #self.linear4 = nn.Linear(layersize, 1)
        

        #er antall output per linear samme som antall nevroner?
        #antall parameter er input x output


    def forward(self, x):
        x = self.linear1(x).relu() #non-linear for INTERNAL value

        #x = self.linear2(x).sigmoid()   #sigmoid for output



        x = self.linear2(x).relu()
        #x = self.linear3(x).relu()
        x = self.linear3(x).sigmoid()
        #x = self.linear4(x).sigmoid()

        return x


trainingset = []
testset = []
test_data = []
attack_data = []


def add_data(data, label):
    for x in data:
        d = {"x": x, "y": torch.Tensor([label]).cuda()}
        defence_data.append(d)

def add_attack_data(data, label):
    for x in data:
        d = {"x": x, "y": torch.Tensor([label]).cuda()}
        attack_data.append(d)

#passwords = ["Gigab-exclm-t R3ceiver", "Ob-dollar-erv3r", "gigabit receiver", "Gigab-exclm-t R3ceiver", "flying automatic monster", "repetition learn machine thinker"]
passwords = ["observer", "Ob-dollar-erv3r", "gigabit receiver", "Gigab-exclm-t R3ceiver", "flying automatic monster", "repetition learn machine thinker"]

totalscore = 0.0
fullscore = 0.0
scorecounter = 0

threshold = 0.5 #gets adjusted for EER

epochs = 200



trainloss = []
testloss = []
testFR = []
testFA = []
testFRsmooth = []
testFAsmooth = []

testacc = []


printgraphs = 0 #True
iterationloss = [0.0] * epochs

avg_width = 10

for passnr in range(0, 6):  ## MAIN LOOP
    print()
    print()
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print()
    print (passwords[passnr])
    for participantnr in range(1,6):
    #for participantnr in range(1,2):
        print()
        print("participant%d" % participantnr)
        #split data into 5 groups here
        defence_data = []
        attack_data = []
        
        attacker = f"data/attack - {passwords[passnr]}.csv"
        defender = f"data/participant{participantnr} - {passwords[passnr]}.csv"

        add_attack_data(load_data(attacker, 2), 1)
        add_data(load_data(defender), 0)

        feats = len(defence_data[0]["x"])
        for d in defence_data:
            assert len(d["x"]) == feats

        
        #print(attack_data)
        attack_buffer = np.stack([d['x'].cpu().numpy() for d in attack_data])
        defence_buffer = np.stack([d['x'].cpu().numpy() for d in defence_data])
        print("attack buffer shape: ", attack_buffer.shape)
        print("defence buffer shape: ", defence_buffer.shape)
        #print(attack_buffer)
        
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
          k_inputA[i] = k_inputA[i].reshape(-1, k_inputA[i].shape[-1])  
          #print(f"FOOBAR NEW Shape of array {i+1}: {k_inputA[i].shape}")

        #randomize the order of the defence data
        np.random.shuffle(defence_buffer)
        
        k_inputP = np.array_split(defence_buffer, k)
        for i in range(k):
            print("k_nr =",i)
            print("k_inputA shape: ", k_inputA[i].shape)    #list of 5 attack data groups
            print("k_inputP shape: ", k_inputP[i].shape)    #list of 5 defence data groups   
        
        
        #K-FOLD VALIDATION

        for k_nr in range(k): #K-FOLD VALIDATION IN THIS LOOP
            random.seed(42 + k_nr)

            print("@@@")
            print("K-fold nr ", k_nr)
            #create the training and test sets with labels for this iteration of k-fold
            buffer_trainingset = np.empty((0, 0))
            traininglabels = np.empty(0)
            buffer_testset = np.empty((0, 0))
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

            #print("size of training set: ", len(buffer_trainingset))
            #print("shape of training set: ", buffer_trainingset.shape)
            #print("size of test set: ", len(buffer_testset))
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

            stop = False

            ########

            attackcount = 0
            defencecount = 0
            for d in trainingset:
                if d["y"] == 1:
                    attackcount += 1
                else:
                    defencecount += 1
            neglossweight = attackcount / defencecount
            epochcounter = 0
            testacc = []
            for i in range(epochs): #epochs, no batches
                epochcounter += 1
                avgloss = 0.0
                losscounter = 0
                #print("iteration %d" % i)
                random.shuffle(trainingset)

                for d in trainingset:
                    #with noise
                    x = d["x"].clone()
                    #x += torch.randn_like(x) * 0.01
                    prob = model(x)
                    y = d["y"]

                    loss = torch.nn.functional.binary_cross_entropy(prob, y)
                    if (d["y"]==0): #defence, minority class
                        loss*=neglossweight
                        #use weights for negative data proportional to amounts
                    #neglossweight

                    
                    #loss = torch.nn.functional.mse_loss(prob, y)

                    avgloss += loss
                    losscounter +=1
                    # print("loss: %g" % loss)

                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    # if loss.item() <= 0.001:
                    #     stop = True
                    #     break



                iterationloss[i] = (avgloss/losscounter)
                trainloss.append(iterationloss[i])
              
                
                

              
              
                with torch.no_grad(): 
                    testlosses = []
                    for d in testset:
                        prob = model(d["x"])
                        y = d["y"]
                        loss = torch.nn.functional.binary_cross_entropy(prob, y)
                        testlosses.append(loss.cpu())
                    testloss.append(sum(testlosses) / len(testlosses))
                
                
                
                #GETTING THE EER
                threshold = 0.5     #starting threshold (not important)
                maxiterations = 50
                minthresh = 0.0 #for binary search
                maxthresh = 1.0


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

                testacc.append((TA+TR) / (TA+TR+FR+FA))
                recent_avg_acc = 1
                prev_avg_acc  = 0
                avg_width = 10

                if epochcounter > avg_width:
                  recent_avg_acc = testacc[-avg_width:]
                  recent_avg_acc = sum(recent_avg_acc) / len(recent_avg_acc)
                if epochcounter > avg_width*2:
                  prev_avg_acc = testacc[-2*avg_width:-avg_width]
                  prev_avg_acc = sum(prev_avg_acc) / len(prev_avg_acc)


                if recent_avg_acc < prev_avg_acc:
                    break
                
            testFAsmooth = []
            testFRsmooth = []
            for i in range(len(testFA)):
                smoothFA = 0
                smoothFR = 0
                count = 0
                width = 3 #SHOULD be odd 

                #for n in range(-2,3): #width = 5
                for n in range(0-math.floor(width/2),math.ceil(width/2)):
                    if i >= n and i+n<len(testFA):
                        smoothFA += testFA[i+n] 
                        smoothFR += testFR[i+n]
                        count +=1

                smoothFA /= count
                smoothFR /= count
                
                testFAsmooth.append(smoothFA)
                testFRsmooth.append(smoothFR)

            
            trainloss_cpu = [] #hack fix for weird error

            for i in range (epochcounter):#(len(iterationloss)):
                
                trainloss_cpu.append(iterationloss[i].cpu().item())
                #megaTrainloss[i] += (iterationloss[i].cpu().item()) / 30


            
            
            
            if printgraphs:
                xaxis = [i for i in range(epochcounter)]
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

                plt.plot(xaxis, testFAsmooth, label="Smooth False Accept")
                plt.plot(xaxis, testFRsmooth, label="Smooth False Reject")
                plt.plot()
                plt.xlabel("Epoch")
                plt.ylabel("Percent Error")
                plt.title("EER per Epochs")
                plt.legend()
                plt.show()
                
            


            trainloss = []
            testlosses = []
            testloss = []
            testFR = []
            testFA = []


            #GETTING THE EER
            threshold = 0.5     #starting threshold (not important)
            maxiterations = 50
            minthresh = 0.0 #for binary search
            maxthresh = 1.0


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
                        prob = model(d["x"])
                        y = d["y"]
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



            print("")
            print("final epoch", epochcounter)
            print("threshold %f" %threshold)
            #print("test correct: %d/%d" % (score, n+m))
            print("%f %%" % (100*score / (n+m) ))
            print("TA: %f FR: %f" %(100*TA/(TA+FR),100*FR/(TA+FR)))
            print("FA: %f TR: %f" %(100*FA/(FA+TR),100*TR/(FA+TR)))
            #print("totalscore increase: %f" % ((100*TA/(TA+FR) + 100*TR/(FA+TR))/2))
            totalscore = totalscore + (100*TA/(TA+FR) + 100*TR/(FA+TR))/2
            scorecounter +=1
            print("Epoch ", epochcounter - avg_width +1)
            print("Use this one:", testacc[epochcounter - avg_width +1], "@@@")
            print("%d / 30\n\n" %scorecounter)
            #print("totalscore increase: %f" % (100*float(score)/(n+m)))
            #totalscore = totalscore + (100*float(score)/(n+m))


            resultsforpaper[resultscounter] += testacc[epochcounter - avg_width +1]
        resultscounter += 1
        

print()
print()


totalscore = (totalscore) / scorecounter #(5 participants x 6 passwords)
fullscore = (float(100)*fullscore) #/ 30 #(5 participants x 6 passwords)
print ("Average test score: %f" % totalscore)



for i in range(len(resultsforpaper)):
    resultsforpaper[i] /= k
    print("\nparticipant ", (i%5)+1)
    print("pwd",(i//6)+1)
    print(i)
    print(resultsforpaper[i])




