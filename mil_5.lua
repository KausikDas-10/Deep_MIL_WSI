--[[

	Date: 25/05/2017
	Multiple Instance Network with-out Auxilary Arms.
	BL-3  
--]]



require 'nn'
require 'cunn'
require 'cudnn'
require 'torch'
require 'loadcaffe'
require 'math'
require 'optim'

pl = require('pl.import_into')()
local t = require '/transforms'

-- MI network with Max-pooling layer at the end

-- MI Network without Auxilary Arm
model = loadcaffe.load('VGG_ILSVRC_19_layers_deploy.prototxt', 'VGG_ILSVRC_19_layers.caffemodel', 'cudnn')--:float()
model: remove(46)
model: remove(45)

--[[
-- AuxilaryArm
auxArm = nn.Sequential()
auxArm: add(nn.Identity())
auxArm: add(nn.Linear(512,2))
auxArm: add(nn.LogSoftMax())
--]]

-- add next max pooling layer
-- modelMax = nn.Sequential()
model: add(nn.Max(1))
model: add(nn.Linear(4096,512))
model: add(nn.ReLU(true))
model: add(nn.Dropout(0.5))
model: add(nn.Linear(512,2))
model: add(nn.LogSoftMax())



--[[
concatModel = nn.ConcatTable()
concatModel: add(auxArm)
concatModel: add(modelMax)

model: add(concatModel)
--]]

--[[
    --Initialization following ResNet
    local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nInputPlane
         v.weight:normal(0,math.sqrt(2/n))
		 if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else	
         	v.bias:zero()
       	 end
      end
    end

    local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
    end

    ConvInit('cudnn.SpatialConvolution')
    BNInit('cudnn.SpatialBatchNormalization')
    for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
    end
--]]

	model:cuda()	
	-- maxModel:cuda()

local count = 0
for i, m in ipairs(model.modules) do
   if count == 4 then break end
   if torch.type(m):find('Convolution') then
      m.accGradParameters = function() end
      m.updateParameters = function() end
      count = count + 1
   end
end

-- cost function 
criterion = nn.ClassNLLCriterion()
criterion:cuda()

func = function(x)
        if x ~= parameters then
              parameters:copy(x)
        end
        gradParameters:zero()
        --f_test = 0
        --neval = neval + 1
        output = model:forward(inputs)
        f = criterion:forward(output,outputs)
        df_do = criterion:backward(output,outputs)
        model:backward(inputs, df_do)

		local norm = torch.norm
		local sign = torch.sign

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
parameters,gradParameters = model:getParameters()

----------------------------------------------------
------------------- Load Data Set ------------------

meanstd = {
   mean = { 198.181, 160.657, 193.945 },
   std 	= { 32.387, 45.069, 27.001 },
}

transform = t.Compose{
			 t.Scale(230),	
		     t.CenterCrop(224),

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
					t.Scale(230),
					t.CenterCrop(224),
					t.ColorNormalize(meanstd),
			  }

-- perm = torch.LongTensor{3, 2, 1}
maxNumIns = 60
numMaligTrain = 40
numBenTrain = 18

for epoch = 1, 300 do
	
	--[[
	if(epoch%50 == 0) then
		-- step-wise decreasing of Learning Rate.
		state.learningRate = state.learningRate*0.1
		print(state)
	end
	--]]

	print('epoch', epoch)
	totErr = 0
	local shuffleBenign    = torch.cat(torch.randperm(numBenTrain),torch.cat(torch.randperm(numBenTrain),torch.randperm(numBenTrain)))
	local shuffleMalignant = torch.randperm(numMaligTrain)
  				
	model:training()

	for bag = 1, numMaligTrain do

		-- load benign data time 
		-- randIns = math.random(1,17)
		dir = '/home/deepkliv/Desktop/Kausik/Data Folds/Fold 2/Malignant/Train/'..shuffleMalignant[bag]..'/200X'
		numOfIns = #pl.dir.getallfiles(dir, '*.png')
		
		if(numOfIns >= maxNumIns) then
			inputs = torch.Tensor(maxNumIns, 3, 224, 224):fill(0)
		else
			inputs = torch.Tensor(numOfIns, 3, 224, 224):fill(0)
		end

		count = 0	

		for i,f in ipairs(pl.dir.getallfiles(dir, '*.png')) do
			inputs[{{i},{},{},{}}] = transform(image.load(f): mul(256.0))
			count = count + 1
			if(count >= maxNumIns) then
				break;
			end
		end
		inputs = inputs:cuda()
		outputs = torch.Tensor{1}:cuda()
		-- outputs = outputs:cuda()
		e_1 = 0
		_,e_1 = optimMethod(func,parameters,state)
		totErr = e_1[1] + totErr

		-- load malignant data time 
		-- randIns = math.random(1,17)--43
		dir = '/home/deepkliv/Desktop/Kausik/Data Folds/Fold 2/Benign/Train/'..shuffleBenign[bag]..'/200X'
		numOfIns = #pl.dir.getallfiles(dir, '*.png')

		if(numOfIns >= maxNumIns) then
			inputs = torch.Tensor(maxNumIns, 3, 224, 224):fill(0)
		else
			inputs = torch.Tensor(numOfIns, 3, 224, 224):fill(0)
		end		

		count = 0
		for i,f in ipairs(pl.dir.getallfiles(dir, '*.png')) do
			inputs[{{i},{},{},{}}] = transform(image.load(f): mul(256.0))
			count = count + 1
			if(count >= maxNumIns) then
				break;
			end
		end
		inputs 	= inputs:cuda()
		outputs = torch.Tensor{2}:cuda()
		-- outputs = outputs:cuda()
		e_1 	= 0
		_,e_1 	= optimMethod(func,parameters,state)
		totErr 	= e_1[1] + totErr

--[[
		--load benign data time 
		randIns = math.random(1,17)
		dir = '/home/deepkliv/Desktop/Kausik/Benign/Train/'..randIns..'/40X'
		numOfIns = #pl.dir.getallfiles(dir, '*.png')

		inputs = torch.Tensor(numOfIns, 3, 224, 224):fill(0)

		for i,f in ipairs(pl.dir.getallfiles(dir, '*.png')) do
			inputs[{{i},{},{},{}}] = transform(image.load(f))
		end
		inputs = inputs:cuda()
		outputs = { torch.Tensor(numOfIns):fill(1), torch.Tensor{1}}:cuda()
		e_1 = 0
		_,e_1 = optimMethod(func,parameters,state)
		totErr = e_1[1] + totErr
--]]
		-- print('trainErr -- ', totErr/2)
	end

	print('Total Bag Training Error -- ', (totErr/(numMaligTrain*2)))

	--------------------------------------------------------
	---------------------- Testing -------------------------
	numBenTest = 6
	numMaligTest = 18

	model:evaluate()
	
	out_test = torch.zeros(numMaligTest)
	for x = 1, numMaligTest do
		dir = '/home/deepkliv/Desktop/Kausik/Data Folds/Fold 2/Malignant/Test/'..x..'/200X'
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
		output = model:forward(inputs:cuda())
		oo,out_test[x] = torch.max(output:float(),1)
	end
	maligAcc = torch.sum(torch.eq(out_test:float(), torch.FloatTensor(numMaligTest):fill(1)))/numMaligTest			
	print('Malignant Acc -- ', maligAcc)

	
	out_test = torch.zeros(numBenTest)
	for x = 1, numBenTest do
		dir = '/home/deepkliv/Desktop/Kausik/Data Folds/Fold 2/Benign/Test/'..x..'/200X'
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
		output = model:forward(inputs:cuda())
		oo,out_test[x] = torch.max(output:float(),1)
	end
	benAcc = torch.sum(torch.eq(out_test:float(), torch.FloatTensor(numBenTest):fill(2)))/numBenTest
	print('Benign Acc -- ', benAcc)
	
	totalAcc = (benAcc + maligAcc)/2
	print('Total Acc BL-3 200X Fold 2 --', totalAcc)

	if ((epoch%150) == 0) then
		torch.save('BL-3_Fold_2_200X.t7', model:clearState())	
	end		

end
torch.save('BL-3_Fold_2_200X.t7', model:clearState())	