-- Date: 28th Dec, 2017
-- Def.: MIL with 4 AuxArm

-- Data: 26th feb, 2018
-- Testing MaxPool Layer at FC-Layer of Dim-4096 
-- SoftMax and Negative_Log_Likelihood_Criterion as the Instance Level Classifiers.

require 'nn'
require 'cunn'
require 'cudnn'
require 'torch'
require 'loadcaffe'
require 'math'
require 'optim'

pl = require('pl.import_into')()
local t = require '/transforms'

model = loadcaffe.load('VGG_ILSVRC_19_layers_deploy.prototxt', 'VGG_ILSVRC_19_layers.caffemodel', 'cudnn')

local auxArm_1 = nn.Sequential()
auxArm_1: add(cudnn.SpatialMaxPooling(56,56,1,1))
auxArm_1: add(nn.Squeeze())
auxArm_1: add(nn.Linear(256,2))
auxArm_1: add(nn.LogSoftMax())
-- auxArm_1: add(nn.Tanh())

local auxArm_2 = nn.Sequential()
auxArm_2: add(cudnn.SpatialMaxPooling(28,28,1,1))
auxArm_2: add(nn.Squeeze())
auxArm_2: add(nn.Linear(512,2))
auxArm_2: add(nn.LogSoftMax())
-- auxArm_2: add(nn.Tanh())

local auxArm_3 = nn.Sequential()
auxArm_3: add(cudnn.SpatialMaxPooling(14,14,1,1))
auxArm_3: add(nn.Squeeze())
auxArm_3: add(nn.Linear(512,2))
auxArm_3: add(nn.LogSoftMax())
-- auxArm_3: add(nn.Tanh())

local auxArm_4 = nn.Sequential()
auxArm_4: add(nn.Linear(4096,1000))
auxArm_4: add(nn.ReLU(true))
auxArm_4: add(nn.Dropout(0.5))
auxArm_4: add(nn.Linear(1000,2))
auxArm_4: add(nn.LogSoftMax())
-- auxArm_4: add(nn.Tanh())

model_1 = nn.Sequential()
for layer = 1, 17 do 
	model_1: add(model:get(layer))
end

local model_2 = nn.Sequential()
for layer = 18, 27 do
	model_2: add(model:get(layer))
end

local model_3 = nn.Sequential()
for layer = 28, 36 do
	model_3: add(model:get(layer))
end

local model_4 = nn.Sequential()
for layer = 37, 41 do
	model_4: add(model:get(layer))
end
model_4:remove(2) -- Removing (nn.View(-1)) Layer
model_4:insert(nn.View(-1):setNumInputDims(3), 2) -- Inserting Reshaping Layer -- 

local modelMax_1 = nn.Sequential()
modelMax_1: add(nn.Max(1))
modelMax_1: add(nn.Linear(4096,1000))
modelMax_1: add(nn.ReLU(true))
modelMax_1: add(nn.Dropout(0.5))
modelMax_1: add(nn.Linear(1000,2))
modelMax_1: add(nn.LogSoftMax())

local conModel_1 = nn.ConcatTable()
conModel_1: add(auxArm_4)
conModel_1: add(modelMax_1)

model_4: add(conModel_1)

local conModel_2 = nn.ConcatTable()
conModel_2: add(auxArm_3)
conModel_2: add(model_4)

model_3: add(conModel_2)

local conModel_3 = nn.ConcatTable()
conModel_3: add(auxArm_2)
conModel_3: add(model_3)

model_2: add(conModel_3) 

local conModel_4 = nn.ConcatTable()
conModel_4: add(auxArm_1)
conModel_4: add(model_2)

model_1: add(conModel_4): add(nn.FlattenTable())
model_1 = model_1:cuda()

model = nil
collectgarbage()

local count = 0
for i, m in ipairs(model_1.modules) do
   	if count == 4 then break end
   	if torch.type(m):find('Convolution') then
      	m.accGradParameters = function() end
      	m.updateParameters = function() end
      	count = count + 1
   	end
end

marginCri = nn.MarginCriterion(0.4)
crossEntropyCri = nn.ClassNLLCriterion()
marginCri = marginCri:cuda()
crossEntropyCri = crossEntropyCri:cuda()

criterion = nn.ParallelCriterion():add(crossEntropyCri,0.001):add(crossEntropyCri,0.001)
								  :add(crossEntropyCri,0.001):add(crossEntropyCri,0.001)
								  :add(crossEntropyCri,0.5)

criterion = criterion:cuda()
--[[
local count = 0
for i, m in ipairs(model_1.modules) do
   	if count == 4 then break end
   	if torch.type(m):find('Convolution') then
      	m.accGradParameters = function() end
      	m.updateParameters = function() end
      	count = count + 1
   	end
end
--]]

--[[
marginCri = nn.MarginCriterion(0.4)
crossEntropyCri = nn.ClassNLLCriterion()
marginCri = marginCri:cuda()
crossEntropyCri = crossEntropyCri:cuda()

criterion = nn.ParallelCriterion():add(marginCri,0.1):add(marginCri,0.1):add(marginCri,0.1):add(marginCri,0.1):add(crossEntropyCri,1)
criterion = criterion:cuda()

criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()
--]]

func = function(x)
        if x ~= parameters then
              parameters:copy(x)
        end

        local norm = torch.norm
		local sign = torch.sign

        gradParameters:zero()
        
        output = model_1:forward(inputs)
		-- output[1] = sign(output[1]-0.75)
        
        f = criterion:forward(output,outputs)
        df_do = criterion:backward(output,outputs)
        model_1:backward(inputs, df_do)

		-- Add L1 Normalization
		if coefL1 ~= 0 then
			f = f + coefL1 * norm(parameters,1)
			gradParameters:add( sign(parameters):mul(coefL1) )
		end

		-- Add L2 Normalization
		if coefL2 ~= 0 then
			f = f + coefL2 * norm(parameters,2)^2/2
			gradParameters: add(parameters:clone():mul(coefL2))
 	    end

 	    collectgarbage()
        return f,gradParameters	--f/batch,gradParameters:div(batch)
end

--sys:tic()
state = {
        -- maxIter = 50,
        learningRate = 1e-5,}
    	-- momentum 	 = 0.99,}
        -- learningRateDecay = 5,}

--momentum = 0.9,
--dampening = 0,
--weightDecay = 1e-6,
--nesterov = true,}

optimMethod = optim.sgd  --adadelta--cg--adam--adagrad--
-- neval = 0
-- batch = 64
coefL1 = 0
coefL2 = 0
parameters,gradParameters = model_1:getParameters()

----------------------------------------------------
--------------- Training Parameter -----------------
meanstd = {
   mean = { 202.76, 176.93, 221.01 },
   std  = { 23.78,  23.78,  23.78  },
}

transform = t.Compose{
			 	-- t.Scale(230),	
		     	-- t.CenterCrop(224),
		     --[[		     		     
		     t.ColorJitter({
		        brightness = 0.4,
		        contrast = 0.4,
		        saturation = 0.4,
		     }),
		     --]]		 
		     -- t.Lighting(0.1, pca.eigval, pca.eigvec),
             -- t.Rotation   		     
             t.ColorNormalize(meanstd),
		     t.HorizontalFlip(0.5),
      	}

transformTest = t.Compose{
				--	t.Scale(230),
				--	t.CenterCrop(224),
				t.ColorNormalize(meanstd),
			  }

----------------------------------------------------
------------------- Load Data Set ------------------
maxNumIns = 50

numBenTrainBag = 23
numMalTrainBag = 19

NumBenTestBag = 9
NumMalTestBag = 7

-- maxNumBag = 1347
	trainTensorBen  = torch.Tensor(numBenTrainBag) :fill(0)
	trainTensorMal  = torch.Tensor(numMalTrainBag) :fill(0)
-- benTrainTensor = torch.Tensor(maxNumBag):fill(0)
-- malTrainTensor = torch.Tensor(maxNumBag):fill(0)
 
-- benTrainTable = torch.totable(trainTensor)
-- malTrainTable = torch.totable(trainTensor)

---------------------------------------------------
---------------------------------------------------

	local shuffleBen  = torch.randperm(numBenTrainBag)
	local shuffleMal  = torch.randperm(numMalTrainBag)

	-- for i = 1, torch.ceil(numMalTrain/maxNumBag) do

		BenTrainTable  = torch.totable(trainTensorBen)
		MalTrainTable  = torch.totable(trainTensorMal)

		for j = 1, numBenTrainBag do 

			dir = './Biseque/Fold 1/Train/Benign/'..shuffleBen[j]
			numOfIns = #pl.dir.getallfiles(dir, '*.png')      

		   	if(numOfIns >= maxNumIns) then
				inputs = torch.Tensor(maxNumIns, 3, 224, 224):fill(0)
			else
				inputs = torch.Tensor(numOfIns,  3, 224, 224):fill(0)
			end

			count = 0
			for i,f in ipairs(pl.dir.getallfiles(dir, '*.png')) do
				-- img = (image.load(f): mul(256.0))
				inputs[{{i},{},{},{}}] = transform(image.load(f): mul(256.0))
				count = count + 1
				if(count >= maxNumIns) then
					break;
				end
			end      
			BenTrainTable[j] = inputs;    					
		end

		for j = 1, numMalTrainBag do 

			dir = './Biseque/Fold 1/Train/Malignant/'..shuffleMal[j]
			numOfIns = #pl.dir.getallfiles(dir, '*.png')      

		   	if(numOfIns >= maxNumIns) then
				inputs = torch.Tensor(maxNumIns, 3, 224, 224):fill(0)
			else
				inputs = torch.Tensor(numOfIns,  3, 224, 224):fill(0)
			end

			count = 0
			for i,f in ipairs(pl.dir.getallfiles(dir, '*.png')) do
				-- img = image.load(f): mul(256.0)
				inputs[{{i},{},{},{}}] = transform(image.load(f): mul(256.0))
				count = count + 1
				if(count >= maxNumIns) then
					break;
				end
			end      
			MalTrainTable[j] = inputs;    				
		end
	
		print('Bag Image Loading Done')


--- train at multiple bag level ---
for epoch = 1, 150 do 

	print('epoch', epoch)
	totErrPerEpoch = 0

	local shuffleBen  = torch.cat(torch.randperm(numBenTrainBag), torch.randperm(numBenTrainBag))
	local shuffleMal  = torch.cat(torch.randperm(numMalTrainBag), torch.randperm(numMalTrainBag))

	model_1:training()

	totErr = 0		
	--- Push for Training 
	for i = 1, numBenTrainBag do					

		-- Push DCIS/cancerous Term
		inputs  = MalTrainTable[shuffleMal[i]];
		inputs  = inputs:cuda()

		--[[
		outputs = { torch.Tensor(inputs:size(1)):fill(1):cuda(), torch.Tensor(inputs:size(1)):fill(1):cuda(), 
					torch.Tensor(inputs:size(1)):fill(1):cuda(), torch.Tensor(inputs:size(1)):fill(1):cuda(),
					torch.Tensor{1}:cuda() }
		--]]
		-- outputs = torch.Tensor{1}:cuda()
		outputs = { torch.Tensor(inputs:size(1)):fill(1):cuda(), torch.Tensor(inputs:size(1)):fill(1):cuda(), 
					torch.Tensor(inputs:size(1)):fill(1):cuda(), torch.Tensor(inputs:size(1)):fill(1):cuda(),
					torch.Tensor{1}:cuda() }
        
		e_1 = 0
		_, e_1 = optimMethod(func,parameters,state)
		totErr = e_1[1] + totErr

		-- Push UDH/non-cancerous Term
		inputs  = BenTrainTable[shuffleBen[i]];
		inputs  = inputs:cuda()
		
		--[[
		outputs = { torch.Tensor(inputs:size(1)):fill(-1):cuda(), torch.Tensor(inputs:size(1)):fill(-1):cuda(), 
					torch.Tensor(inputs:size(1)):fill(-1):cuda(), torch.Tensor(inputs:size(1)):fill(-1):cuda(),
					torch.Tensor{2}:cuda() }
		--]]
		-- outputs = torch.Tensor{2}:cuda()
		outputs = { torch.Tensor(inputs:size(1)):fill(2):cuda(), torch.Tensor(inputs:size(1)):fill(2):cuda(), 
					torch.Tensor(inputs:size(1)):fill(2):cuda(), torch.Tensor(inputs:size(1)):fill(2):cuda(),
					torch.Tensor{2}:cuda() }

		e_1 = 0
		_, e_1 = optimMethod(func,parameters,state)
		totErr = e_1[1] + totErr
	end

	totErr = totErr/(2*numBenTrainBag)
	print('totErrPerEpoch--', totErr)
	collectgarbage()		
	
	--------------------------------------------------
	------------------- Testing ----------------------	

	if (epoch%10 == 0) then

		model_1:evaluate()

		print('Inside Testing')
		print('epoch', epoch)

		totalUDHTrue   = 0
		totalUDHAcc    = 0
		totalDCISTrue  = 0
		totalDCISAcc   = 0

		out_test = torch.Tensor(NumBenTestBag):fill(0)
		for x = 1, NumBenTestBag do 

			dir = './Biseque/Fold 1/Test/Benign/'..x
			numOfIns = #pl.dir.getallfiles(dir, '*.png')      

		   	if(numOfIns >= maxNumIns) then
				inputs = torch.Tensor(maxNumIns, 3, 224, 224):fill(0)
			else
				inputs = torch.Tensor(numOfIns,  3, 224, 224):fill(0)
			end

			count = 0
			for k,f in ipairs(pl.dir.getallfiles(dir, '*.png')) do
				-- img = (image.load(f): mul(256.0))
				inputs[{{k},{},{},{}}] = transformTest(image.load(f): mul(256.0))
				count = count + 1
				if(count >= maxNumIns) then
					break;
				end
			end      

			output = model_1:forward(inputs:cuda())
			oo,out_test[x] = torch.max(output[5]:float(),1)
		end

		totalBenTrue = torch.sum(torch.eq(out_test:float(), torch.FloatTensor(NumBenTestBag):fill(2)))
		totalBenAcc  = totalBenTrue/NumBenTestBag

		print('Benign Total TruePositive --', 	  	totalBenTrue)
		print('Benign Total Num Test--', 			NumBenTestBag)
		print('Benign Acc  -- ',   		 			totalBenAcc)

		out_test = torch.Tensor(NumMalTestBag):fill(0)
		for x = 1, NumMalTestBag do 

			dir = './Biseque/Fold 1/Test/Malignant/'..x
			numOfIns = #pl.dir.getallfiles(dir, '*.png')      

		   	if(numOfIns >= maxNumIns) then
				inputs = torch.Tensor(maxNumIns, 3, 224, 224):fill(0)
			else
				inputs = torch.Tensor(numOfIns,  3, 224, 224):fill(0)
			end

			count = 0
			for k,f in ipairs(pl.dir.getallfiles(dir, '*.png')) do
				-- img = image.load(f): mul(256.0)
				inputs[{{k},{},{},{}}] = transformTest(image.load(f): mul(256.0))
				count = count + 1
				if(count >= maxNumIns) then
					break;
				end
			end    

			output = model_1:forward(inputs:cuda())
			oo,out_test[x] = torch.max(output[5]:float(),1)
		end

		totalMalTrue = torch.sum(torch.eq(out_test:float(), torch.FloatTensor(NumMalTestBag):fill(1)))
		totalMalAcc  = totalMalTrue/NumMalTestBag

		print('Malignant Total TruePositive --' , 	totalMalTrue)
		print('Malignant Total Num Test--', 		NumMalTestBag)
		print('Malignant Acc  -- ',   				totalMalAcc)

		print('Total Acc for Fold_1 BL_5--', (totalBenTrue+totalMalTrue)/(NumBenTestBag+NumMalTestBag))

		out_test = nil
		output 	 = nil
		inputs   = nil

		torch.save('Biseque_BL_5_Fold_1.t7', model_1:clearState())
		collectgarbage()
	end

	collectgarbage()
end



--------------  calculate mean//std  ---------------

----------------------------------------------------


-- perm = torch.LongTensor{3, 2, 1}

--[[
for epoch = 1, 300 do

	print('epoch', epoch)
	totErr = 0
	local shuffleBenign    = torch.cat(torch.randperm(numBenTrain),torch.randperm(numBenTrain))
	local shuffleMalignant = (torch.randperm(numMaligTrain) + 8960)

	model_1:training()	-- activate training mode

	for bag = 1, numMaligTrain do

		print('bag ID -- ', bag)
		-- load malignant data time 
		dir = '/home/deepkliv/Desktop/Kausik/MIL2/DDSM/Fold 1/Malignant/Train/'..shuffleMalignant[bag]
		numOfIns = #pl.dir.getallfiles(dir, '*.jpg')

		if(numOfIns >= maxNumIns) then
			inputs = torch.Tensor(maxNumIns, 3, 224, 224):fill(0)
		else
			inputs = torch.Tensor(numOfIns,  3, 224, 224):fill(0)
		end

		count = 0

		for i,f in ipairs(pl.dir.getallfiles(dir, '*.jpg')) do
			img = transform(image.load(f): mul(256.0))
			inputs[{{i},{},{},{}}] = torch.cat(torch.cat(img,img,1),img,1)
			count = count + 1
			if(count >= maxNumIns) then
				break;
			end
		end


		inputs = inputs:cuda()
		outputs = { torch.Tensor(inputs:size(1)):fill(1):cuda(), torch.Tensor(inputs:size(1)):fill(1):cuda(), 
					torch.Tensor(inputs:size(1)):fill(1):cuda(), torch.Tensor(inputs:size(1)):fill(1):cuda(),
					torch.Tensor{1}:cuda() }

		-- outputs = outputs:cuda()
		e_1 = 0
		_, e_1 = optimMethod(func,parameters,state)
		totErr = e_1[1] + totErr

		-- load benign data time 
		dir = '/home/deepkliv/Desktop/Kausik/MIL2/DDSM/Fold 1/Benign/Train/'..shuffleBenign[bag]
		numOfIns = #pl.dir.getallfiles(dir, '*.jpg')

		if(numOfIns >= maxNumIns) then
			inputs = torch.Tensor(maxNumIns, 3, 224, 224):fill(0)
		else
			inputs = torch.Tensor(numOfIns,  3, 224, 224):fill(0)
		end

		count = 0

		for i,f in ipairs(pl.dir.getallfiles(dir, '*.jpg')) do
			img = transform(image.load(f): mul(256.0))
			inputs[{{i},{},{},{}}] = torch.cat(torch.cat(img,img,1),img,1)
			count = count + 1
			if(count >= maxNumIns) then
				break;
			end
		end

		inputs = inputs:cuda()
		outputs = { torch.Tensor(inputs:size(1)):fill(-1):cuda(), torch.Tensor(inputs:size(1)):fill(-1):cuda(), 
					torch.Tensor(inputs:size(1)):fill(-1):cuda(), torch.Tensor(inputs:size(1)):fill(-1):cuda(),
					torch.Tensor{2}:cuda() }

		e_1 = 0
		_,e_1 = optimMethod(func,parameters,state)
		totErr = e_1[1] + totErr

	end

	print('Total Bag Training Error -- ', (totErr/2*numMaligTrain))

	------------------------------------------------------
	---------------------- Testing -----------------------
--[[
	numBenTest = 6
	numMaligTest = 18

	model_1:evaluate()
	
	out_test = torch.zeros(numMaligTest)
	for x = 1, numMaligTest do
		dir = '/home/deepkliv/Desktop/Kausik/Data Folds/Fold 2/Malignant/Test/'..x..'/40X'
		numOfIns = #pl.dir.getallfiles(dir, '*.png')
		
		if(numOfIns >= maxNumIns) then
			inputs = torch.Tensor(maxNumIns, 3, 224, 224):fill(0)
		else
			inputs = torch.Tensor(numOfIns, 3, 224, 224):fill(0)
		end	

		count = 0
		for i,f in ipairs(pl.dir.getallfiles(dir, '*.png')) do
			inputs[{{i},{},{},{}}] = transformTest(image.load(f): mul(256.0))
			count = count + 1
			if(count >= maxNumIns) then
				break;
			end
		end
		output = model_1:forward(inputs:cuda())
		oo,out_test[x] = torch.max(output[5]:float(),1)
	end
	maligAcc = torch.sum(torch.eq(out_test:float(), torch.FloatTensor(numMaligTest):fill(1)))/numMaligTest			
	print('Malignant Acc -- ', maligAcc)
	
	out_test = torch.zeros(numBenTest)
	for x = 1, numBenTest do
		dir = '/home/deepkliv/Desktop/Kausik/Data Folds/Fold 2/Benign/Test/'..x..'/40X'
		numOfIns = #pl.dir.getallfiles(dir, '*.png')
		
		if(numOfIns >= maxNumIns) then
			inputs = torch.Tensor(maxNumIns, 3, 224, 224):fill(0)
		else
			inputs = torch.Tensor(numOfIns, 3, 224, 224):fill(0)
		end	

		count = 0
		for i,f in ipairs(pl.dir.getallfiles(dir, '*.png')) do
			inputs[{{i},{},{},{}}] = transformTest(image.load(f): mul(256.0))
			count = count + 1
			if(count >= maxNumIns) then
				break;
			end
		end
		output = model_1:forward(inputs:cuda())
		oo,out_test[x] = torch.max(output[5]:float(),1)
	end
	benAcc = torch.sum(torch.eq(out_test:float(), torch.FloatTensor(numBenTest):fill(2)))/numBenTest
	print('Benign Acc -- ', benAcc)
	
	totalAcc = (benAcc + maligAcc)/2
	print('Total Acc 40X (Fold 2) (SoftMax+NLL at Bag + MarginCri(0.4) at Instances+4AuxArm+MaxPool_Dim4096)-- --', totalAcc)

--]]

--[[
	if ((epoch%150) == 0) then
		torch.save('Proposed_Fold_2_40X_MaxPoolDim4096_NLL_AtBagLevel.t7', model_1:clearState())	
	end		

--]]

-- end
--]]
-- torch.save('DDSM.t7', model_1:clearState())	
