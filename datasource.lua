require 'torch'

local Datasource = torch.class('Datasource')

function Datasource:__init()
   
end

function Datasource:nextSample(set)
   return self:nextBatch(1, set)[1]
end

-- set in [train|test|val]
function Datasource:nextBatch(batchSize, set)
   error('This is an abstract class')
end

function Datasource:type()
   error('This is an abstract class')
end

function Datasource:cuda()
   self:type('torch.CudaTensor')
end

function Datasource:float()
   self:type('torch.FloatTensor')
end

function Datasource:double()
   self:type('torch.DoubleTensor')
end