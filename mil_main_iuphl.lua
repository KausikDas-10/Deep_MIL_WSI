-- Date: 28th Dec, 2017
-- Def.: MIL with 4 AuxArm

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
auxArm_1: add(nn.Linear(256,1))
auxArm_1: add(nn.Tanh())

local auxArm_2 = nn.Sequential()
auxArm_2: add(cudnn.SpatialMaxPooling(28,28,1,1))
auxArm_2: add(nn.Squeeze())
auxArm_2: add(nn.Linear(512,1))
auxArm_2: add(nn.Tanh())

local auxArm_3 = nn.Sequential()
auxArm_3: add(cudnn.SpatialMaxPooling(14,14,1,1))
auxArm_3: add(nn.Squeeze())
auxArm_3: add(nn.Linear(512,1))
auxArm_3: add(nn.Tanh())

local auxArm_4 = nn.Sequential()
auxArm_4: add(nn.Linear(4096,512))
auxArm_4: add(nn.ReLU(true))
auxArm_4: add(nn.Dropout(0.5))
auxArm_4: add(nn.Linear(512,1))
auxArm_4: add(nn.Tanh())

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
modelMax_1: add(nn.Linear(4096,512))
modelMax_1: add(nn.ReLU(true))
modelMax_1: add(nn.Dropout(0.5))
modelMax_1: add(nn.Linear(512,1))
modelMax_1: add(nn.Sigmoid())

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
crossEntropyCri = nn.BCECriterion()
marginCri = marginCri:cuda()
crossEntropyCri = crossEntropyCri:cuda()

criterion = nn.ParallelCriterion():add(marginCri,0.1):add(marginCri,0.1):add(marginCri,0.1):add(marginCri,0.1):add(crossEntropyCri,1)
criterion = criterion:cuda()

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
------------------- Load Data Set ------------------

meanstd = {
   mean = { 183.4845, 143.6125, 179.0315 },
   std  = { 1.4234,   1.2885,   0.6378   },
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
		     	t.ColorNormalize(meanstd),
		     	t.HorizontalFlip(0.5),
      	}

transformTest = t.Compose{
					-- t.Scale(230),
					-- t.CenterCrop(224),
					t.ColorNormalize(meanstd),
			  }

-- perm = torch.LongTensor{3, 2, 1}
maxNumIns = 60
numDCIS = 275
numUDH = 254

for epoch = 1, 150 do

	print('epoch', epoch)
	totErr = 0
	local shuffleUDH  =  torch.cat(torch.randperm(numUDH),torch.randperm(numUDH))
	local shuffleDCIS =  torch.randperm(numDCIS)

	model_1:training()	-- activate training mode

	for bag = 1, numDCIS do
		-- print('bag ', bag)
		-- load malignant data time 
		dir = '/home/deepkliv/Desktop/Kausik/IUPHL/DCIS_3/Fold 3/Train/'..shuffleDCIS[bag]
		numOfIns = #pl.dir.getallfiles(dir, '*.jpg')

		inputs = torch.Tensor(numOfIns, 3, 224, 224):fill(0)
		count = 0

		for i,f in ipairs(pl.dir.getallfiles(dir, '*.jpg')) do
			inputs[{{i},{},{},{}}] = transform(image.load(f): mul(256.0))
			count = count + 1
			if(count >= maxNumIns) then
				break;
			end
		end

		inputs = inputs:cuda()
		-- outputs = { torch.Tensor(inputs:size(1)):fill(1):cuda(), torch.Tensor{1}:cuda() }
		outputs = { torch.Tensor(inputs:size(1)):fill(1):cuda(), torch.Tensor(inputs:size(1)):fill(1):cuda(), torch.Tensor(inputs:size(1)):fill(1):cuda(), torch.Tensor(inputs:size(1)):fill(1):cuda(),torch.Tensor{1}:cuda()}

		e_1 = 0
		_, e_1 = optimMethod(func,parameters,state)
		totErr = e_1[1] + totErr

		-- load benign data time 
		dir = '/home/deepkliv/Desktop/Kausik/IUPHL/UDH_3/Fold 3/Train/'..shuffleUDH[bag]
		numOfIns = #pl.dir.getallfiles(dir, '*.jpg')

		inputs = torch.Tensor(numOfIns, 3, 224, 224):fill(0)
		count = 0

		for i,f in ipairs(pl.dir.getallfiles(dir, '*.jpg')) do
			inputs[{{i},{},{},{}}] = transform(image.load(f): mul(256.0))
			count = count + 1
			if(count >= maxNumIns) then
				break;
			end
		end

		inputs = inputs:cuda()
		-- outputs = { torch.Tensor(inputs:size(1)):fill(-1):cuda(), torch.Tensor{0}:cuda() }
		outputs = { torch.Tensor(inputs:size(1)):fill(-1):cuda(), torch.Tensor(inputs:size(1)):fill(-1):cuda(), torch.Tensor(inputs:size(1)):fill(-1):cuda(), torch.Tensor(inputs:size(1)):fill(-1):cuda(),torch.Tensor{0}:cuda()}
		-- outputs = outputs:cuda()

		e_1 = 0
		_,e_1 = optimMethod(func,parameters,state)
		totErr = e_1[1] + totErr
		collectgarbage()
	end

	print('Total Bag Training Error -- ', (totErr/2*numDCIS))
	------------------------------------------------------
	---------------------- Testing -----------------------

	numUDHTest = 110
	numDCISTest = 116
	UDH_Acc = 0
	model_1:evaluate()

	for x = 1, numUDHTest do
		dir = '/home/deepkliv/Desktop/Kausik/IUPHL/UDH_3/Fold 3/Test/'..x
		numOfIns = #pl.dir.getallfiles(dir, '*.jpg')
		
		inputs = torch.Tensor(numOfIns, 3, 224, 224):fill(0)
		count = 0
		
		for i,f in ipairs(pl.dir.getallfiles(dir, '*.jpg')) do
			inputs[{{i},{},{},{}}] = transformTest(image.load(f): mul(256.0))
			count = count + 1
			if(count >= maxNumIns) then
				break;
			end
		end
		
		output = model_1: forward(inputs:cuda())
		if torch.lt(output[5], 0.5)[1]==1 then
			-- print('UDH with prob.', output[5])
			UDH_Acc = UDH_Acc + 1
		else
			print('not UDH', output[5])
		end
	end
	UDH_AccPer = UDH_Acc/numUDHTest
	print('UDH Acc -- ', UDH_AccPer)

	DCIS_Acc = 0
	for x = 1, numDCISTest do
		dir = '/home/deepkliv/Desktop/Kausik/IUPHL/DCIS_3/Fold 3/Test/'..x
		numOfIns = #pl.dir.getallfiles(dir, '*.jpg')
		
		inputs = torch.Tensor(numOfIns, 3, 224, 224):fill(0)
		count = 0

		for i,f in ipairs(pl.dir.getallfiles(dir, '*.jpg')) do
			inputs[{{i},{},{},{}}] = transformTest(image.load(f): mul(256.0))
			count = count + 1
			if(count >= maxNumIns) then
				break;
			end
		end
		
		output = model_1: forward(inputs:cuda())
		if torch.gt(output[5], 0.5)[1]==1 then
			-- print('DCIS with prob.', output[5])
			DCIS_Acc = DCIS_Acc + 1
		else
			print('not DCIS', output[5])
		end
	end
	DCIS_AccPer = DCIS_Acc/numDCISTest
	print('DCIS Acc -- ', DCIS_AccPer)
	
	totalAcc = (UDH_Acc+DCIS_Acc)/(numUDHTest+numDCISTest)
	print(' Total Acc With AUX ARM Fold 3 (CrossEntropy on Bag + MarginCri(0.4) on Instances +#SideLevelSupervision4AuxArm)-- ', totalAcc)
	collectgarbage()
end

torch.save('IUPHL_Fold_3.t7', model_1:clearState())	